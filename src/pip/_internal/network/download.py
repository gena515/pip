"""Download files with progress indicators.
"""

import email.message
import logging
import mimetypes
import os
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from threading import Event, Semaphore, Thread
from typing import Iterable, Iterator, List, Mapping, Optional, Tuple, Union, cast

from pip._vendor.requests.models import Response
from pip._vendor.rich.progress import TaskID

from pip._internal.cli.progress_bars import (
    BatchedProgress,
    ProgressBarType,
    get_download_progress_renderer,
)
from pip._internal.exceptions import CommandError, NetworkConnectionError
from pip._internal.models.index import PyPI
from pip._internal.models.link import Link
from pip._internal.network.cache import is_from_cache
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks
from pip._internal.utils.misc import format_size, redact_auth_from_url, splitext

logger = logging.getLogger(__name__)


def _get_http_response_size(resp: Response) -> Optional[int]:
    try:
        return int(resp.headers["content-length"])
    except (ValueError, KeyError, TypeError):
        return None


def _format_download_log_url(link: Link) -> str:
    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    return redact_auth_from_url(url)


def _log_download_link(
    link: Link,
    total_length: Optional[int],
    link_is_from_cache: bool = False,
) -> None:
    logged_url = _format_download_log_url(link)

    if total_length:
        logged_url = f"{logged_url} ({format_size(total_length)})"

    if link_is_from_cache:
        logger.info("Using cached %s", logged_url)
    else:
        logger.info("Downloading %s", logged_url)


def _prepare_download(
    resp: Response,
    link: Link,
    progress_bar: ProgressBarType,
    quiet: bool = False,
    color: bool = True,
) -> Iterable[bytes]:
    total_length = _get_http_response_size(resp)

    _log_download_link(link, total_length, is_from_cache(resp))

    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif is_from_cache(resp):
        show_progress = False
    elif not total_length:
        show_progress = True
    elif total_length > (512 * 1024):
        show_progress = True
    else:
        show_progress = False

    chunks = response_chunks(resp)

    if not show_progress:
        return chunks

    renderer = get_download_progress_renderer(
        bar_type=progress_bar,
        size=total_length,
        quiet=quiet,
        color=color,
    )
    return renderer(chunks)


def sanitize_content_filename(filename: str) -> str:
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition: str, default_filename: str) -> str:
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    m = email.message.Message()
    m["content-type"] = content_disposition
    filename = m.get_param("filename")
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(str(filename))
    return filename or default_filename


def _get_http_response_filename(
    headers: Mapping[str, str], resp_url: str, link: Link
) -> str:
    """Get an ideal filename from the given HTTP response, falling back to
    the link filename if not provided.
    """
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = headers.get("content-disposition", None)
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext: Optional[str] = splitext(filename)[1]
    if not ext:
        ext = mimetypes.guess_extension(headers.get("content-type", ""))
        if ext:
            filename += ext
    if not ext and link.url != resp_url:
        ext = os.path.splitext(resp_url)[1]
        if ext:
            filename += ext
    return filename


def _maybe_log_http_error(response: Response, link: Link) -> Response:
    try:
        raise_for_status(response)
        return response
    except NetworkConnectionError as e:
        assert e.response is not None
        logger.critical("HTTP error %s while getting %s", e.response.status_code, link)
        raise


def _http_get_download(session: PipSession, link: Link) -> Response:
    target_url = link.url.split("#", 1)[0]
    resp = session.get(target_url, headers=HEADERS, stream=True)
    return _maybe_log_http_error(resp, link)


def _http_head_content_info(
    session: PipSession,
    link: Link,
) -> Tuple[Optional[int], str]:
    target_url = link.url.split("#", 1)[0]
    resp = _maybe_log_http_error(session.head(target_url), link)

    if length := resp.headers.get("content-length"):
        content_length = int(length)
    else:
        content_length = None

    filename = _get_http_response_filename(resp.headers, resp.url, link)
    return content_length, filename


class Downloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: ProgressBarType,
        quiet: bool = False,
        color: bool = True,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar
        self._quiet = quiet
        self._color = color

    def __call__(self, link: Link, location: str) -> Tuple[str, str]:
        """Download the file given by link into location."""
        resp = _http_get_download(self._session, link)

        filename = _get_http_response_filename(resp.headers, resp.url, link)
        filepath = os.path.join(location, filename)

        chunks = _prepare_download(
            resp, link, self._progress_bar, quiet=self._quiet, color=self._color
        )
        with open(filepath, "wb") as content_file:
            for chunk in chunks:
                content_file.write(chunk)
        content_type = resp.headers.get("Content-Type", "")
        return filepath, content_type


class _ErrorReceiver:
    def __init__(self, error_flag: Event) -> None:
        self._error_flag = error_flag
        self._thread_exception: Optional[BaseException] = None

    def receive_error(self, exc: BaseException) -> None:
        self._error_flag.set()
        self._thread_exception = exc

    def stored_error(self) -> Optional[BaseException]:
        return self._thread_exception


@contextmanager
def _spawn_workers(
    workers: List[Thread], error_flag: Event
) -> Iterator[_ErrorReceiver]:
    err_recv = _ErrorReceiver(error_flag)
    try:
        for w in workers:
            w.start()
            # We've sorted the list of worker threads so they correspond to the largest
            # downloads first. Each thread immediately waits upon a semaphore to limit
            # maximum parallel downloads, and we would like the semaphore's internal
            # wait queue to retain the same order we established earlier (otherwise, we
            # would end up nondeterministically downloading files out of our desired
            # order). Yielding to the scheduler here is intended to give the thread we
            # just started time to jump into the semaphore, either to execute further
            # (until the semaphore is full) or to jump into the queue at the desired
            # position. We seem to get the ordering reliably even without this explicit
            # yield, and ideally we would like to somehow ensure this deterministically,
            # but this is relatively idiomatic and lets us lean on much fewer
            # synchronization constructs. We can revisit this if users find the ordering
            # is unreliable. It's easy to see if we've messed up, as the rich progress
            # table prominently shows each download size and which ones are executing.
            time.sleep(0)
        yield err_recv
    except BaseException as e:
        err_recv.receive_error(e)
    finally:
        thread_exception = err_recv.stored_error()
        if thread_exception is not None:
            logger.critical("Received exception, shutting down downloader threads...")

        # Ensure each thread is complete by the time the queue has exited, either by
        # writing the full request contents, or by checking the Event from an exception.
        for w in workers:
            # If the user (reasonably) wants to hit ^C again to try to make it close
            # faster, we want to avoid spewing out a ton of error text, but at least
            # let's let them know we hear them and we're trying to shut down!
            while w.is_alive():
                try:
                    w.join()
                except BaseException:
                    logger.critical("Shutting down worker threads, please wait...")

        if thread_exception is not None:
            raise thread_exception


def _copy_chunks(
    output_queue: "Queue[Union[Tuple[Link, Path, Optional[str]], BaseException]]",
    error_flag: Event,
    semaphore: Semaphore,
    session: PipSession,
    location: Path,
    batched_progress: BatchedProgress,
    download_info: Tuple[Link, TaskID, str],
) -> None:
    link, task_id, filename = download_info

    with semaphore:
        # Check if another thread exited with an exception before we started.
        if error_flag.is_set():
            return
        try:
            resp = _http_get_download(session, link)

            filepath = location / filename
            content_type = resp.headers.get("Content-Type")
            # TODO: different chunk size for batched downloads?
            chunks = response_chunks(resp)
            with filepath.open("wb") as output_file:
                # Notify that the current task has begun.
                batched_progress.start_subtask(task_id)
                for chunk in chunks:
                    # Check if another thread exited with an exception between chunks.
                    if error_flag.is_set():
                        return
                    # Copy chunk directly to output file, without any
                    # additional buffering.
                    output_file.write(chunk)
                    # Update progress.
                    batched_progress.advance_subtask(task_id, len(chunk))

            output_queue.put((link, filepath, content_type))
        except BaseException as e:
            output_queue.put(e)
        finally:
            batched_progress.finish_subtask(task_id)


class BatchDownloader:
    def __init__(
        self,
        session: PipSession,
        progress_bar: ProgressBarType,
        quiet: bool = False,
        color: bool = True,
        max_parallelism: Optional[int] = None,
    ) -> None:
        self._session = session
        self._progress_bar = progress_bar
        self._quiet = quiet
        self._color = color

        if max_parallelism is None:
            max_parallelism = 1
        if max_parallelism < 1:
            raise CommandError(
                f"invalid batch download parallelism {max_parallelism}: must be >=1"
            )
        self._max_parallelism: int = max_parallelism

    def __call__(
        self, links: Iterable[Link], location: Path
    ) -> Iterable[Tuple[Link, Tuple[Path, Optional[str]]]]:
        """Download the files given by links into location."""
        # Calculate the byte length for each file, if available.
        links_with_lengths: List[Tuple[Link, Tuple[Optional[int], str]]] = [
            (link, _http_head_content_info(self._session, link)) for link in links
        ]
        # Sum up the total length we'll be downloading.
        # TODO: filter out responses from cache from total download size?
        total_length: Optional[int] = 0
        for _link, (maybe_len, _filename) in links_with_lengths:
            if maybe_len is None:
                total_length = None
                break
            assert total_length is not None
            total_length += maybe_len
        # If lengths are available, sort downloads to perform larger downloads first.
        if total_length is not None:
            # Extract the length from each tuple entry.
            links_with_lengths.sort(key=lambda t: cast(int, t[1][0]), reverse=True)

        # Set up state to track thread progress, including inner exceptions.
        total_downloads: int = len(links_with_lengths)
        completed_downloads: int = 0
        q: "Queue[Union[Tuple[Link, Path, Optional[str]], BaseException]]" = Queue()
        error_flag = Event()
        # Limit downloads to 10 at a time so we can reuse our connection pool.
        semaphore = Semaphore(value=self._max_parallelism)
        batched_progress = BatchedProgress.select_progress_bar(
            self._progress_bar
        ).create(
            num_tasks=total_downloads,
            known_total_length=total_length,
            quiet=self._quiet,
            color=self._color,
        )

        # Log the link we're about to download, and add it to the progress table.
        link_tasks: List[Tuple[Link, TaskID, str]] = []
        for link, (maybe_len, filename) in links_with_lengths:
            _log_download_link(link, maybe_len)
            task_id = batched_progress.add_subtask(filename, maybe_len)
            link_tasks.append((link, task_id, filename))

        # Distribute request i/o across equivalent threads.
        # NB: event-based/async is likely a better model than thread-per-request, but
        #     (1) pip doesn't use async anywhere else yet,
        #     (2) this is at most one thread per dependency in the graph (less if any
        #         are cached)
        #     (3) pip is fundamentally run in a synchronous context with a clear start
        #         and end, instead of e.g. as a server which needs to process
        #         arbitrary further requests at the same time.
        # For these reasons, thread-per-request should be sufficient for our needs.
        workers = [
            Thread(
                target=_copy_chunks,
                args=(
                    q,
                    error_flag,
                    semaphore,
                    self._session,
                    location,
                    batched_progress,
                    download_info,
                ),
            )
            for download_info in link_tasks
        ]

        with batched_progress:
            with _spawn_workers(workers, error_flag) as err_recv:
                # Read completed downloads from queue, or extract the exception.
                while completed_downloads < total_downloads:
                    # Get item from queue, but also check for ^C from user!
                    try:
                        item = q.get()
                    except BaseException as e:
                        err_recv.receive_error(e)
                        break
                    # Now see if the worker thread failed with an exception (unlikely).
                    if isinstance(item, BaseException):
                        err_recv.receive_error(item)
                        break
                    # Otherwise, the thread succeeded, and we can pass it to
                    # the preparer!
                    link, filepath, content_type = item
                    completed_downloads += 1
                    yield link, (filepath, content_type)
