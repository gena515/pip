"""Prepares a distribution for installation
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False
# mypy: disallow-untyped-defs=False

import cgi
import logging
import mimetypes
import os
import shutil
import sys

from pip._vendor import requests
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._vendor.six import PY2

from pip._internal.distributions import (
    make_distribution_for_install_requirement,
)
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
    DirectoryUrlHashUnsupported,
    HashMismatch,
    HashUnpinned,
    InstallationError,
    PreviousBuildDirError,
    VcsHashUnsupported,
)
from pip._internal.models.index import PyPI
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import expanduser
from pip._internal.utils.filesystem import copy2_fixed
from pip._internal.utils.hashes import MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.marker_files import write_delete_marker_file
from pip._internal.utils.misc import (
    ask_path_exists,
    backup_dir,
    consume,
    display_path,
    format_size,
    hide_url,
    normalize_path,
    path_to_display,
    rmtree,
    splitext,
)
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.ui import DownloadProgressProvider
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs

if MYPY_CHECK_RUNNING:
    from typing import (
        Callable, IO, List, Optional, Tuple,
    )

    from mypy_extensions import TypedDict

    from pip._internal.distributions import AbstractDistribution
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.models.link import Link
    from pip._internal.req.req_install import InstallRequirement
    from pip._internal.req.req_tracker import RequirementTracker
    from pip._internal.utils.hashes import Hashes

    if PY2:
        CopytreeKwargs = TypedDict(
            'CopytreeKwargs',
            {
                'ignore': Callable[[str, List[str]], List[str]],
                'symlinks': bool,
            },
            total=False,
        )
    else:
        CopytreeKwargs = TypedDict(
            'CopytreeKwargs',
            {
                'copy_function': Callable[[str, str], None],
                'ignore': Callable[[str, List[str]], List[str]],
                'ignore_dangling_symlinks': bool,
                'symlinks': bool,
            },
            total=False,
        )

logger = logging.getLogger(__name__)


def _get_prepared_distribution(req, req_tracker, finder, build_isolation):
    """Prepare a distribution for installation.
    """
    abstract_dist = make_distribution_for_install_requirement(req)
    with req_tracker.track(req):
        abstract_dist.prepare_distribution_metadata(finder, build_isolation)
    return abstract_dist


def unpack_vcs_link(link, location):
    # type: (Link, str) -> None
    vcs_backend = vcs.get_backend_for_scheme(link.scheme)
    assert vcs_backend is not None
    vcs_backend.unpack(location, url=hide_url(link.url))


def _progress_indicator(iterable, *args, **kwargs):
    return iterable


def _download_url(
    resp,  # type: Response
    link,  # type: Link
    content_file,  # type: IO
    hashes,  # type: Optional[Hashes]
    progress_bar  # type: str
):
    # type: (...) -> None
    try:
        total_length = int(resp.headers['content-length'])
    except (ValueError, KeyError, TypeError):
        total_length = 0

    cached_resp = getattr(resp, "from_cache", False)
    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif cached_resp:
        show_progress = False
    elif total_length > (40 * 1000):
        show_progress = True
    elif not total_length:
        show_progress = True
    else:
        show_progress = False

    def resp_read(chunk_size):
        try:
            # Special case for urllib3.
            for chunk in resp.raw.stream(
                    chunk_size,
                    # We use decode_content=False here because we don't
                    # want urllib3 to mess with the raw bytes we get
                    # from the server. If we decompress inside of
                    # urllib3 then we cannot verify the checksum
                    # because the checksum will be of the compressed
                    # file. This breakage will only occur if the
                    # server adds a Content-Encoding header, which
                    # depends on how the server was configured:
                    # - Some servers will notice that the file isn't a
                    #   compressible file and will leave the file alone
                    #   and with an empty Content-Encoding
                    # - Some servers will notice that the file is
                    #   already compressed and will leave the file
                    #   alone and will add a Content-Encoding: gzip
                    #   header
                    # - Some servers won't notice anything at all and
                    #   will take a file that's already been compressed
                    #   and compress it again and set the
                    #   Content-Encoding: gzip header
                    #
                    # By setting this not to decode automatically we
                    # hope to eliminate problems with the second case.
                    decode_content=False):
                yield chunk
        except AttributeError:
            # Standard file-like object.
            while True:
                chunk = resp.raw.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def written_chunks(chunks):
        for chunk in chunks:
            content_file.write(chunk)
            yield chunk

    progress_indicator = _progress_indicator

    if link.netloc == PyPI.file_storage_domain:
        url = link.show_url
    else:
        url = link.url_without_fragment

    if show_progress:  # We don't show progress on cached responses
        progress_indicator = DownloadProgressProvider(progress_bar,
                                                      max=total_length)
        if total_length:
            logger.info("Downloading %s (%s)", url, format_size(total_length))
        else:
            logger.info("Downloading %s", url)
    elif cached_resp:
        logger.info("Using cached %s", url)
    else:
        logger.info("Downloading %s", url)

    downloaded_chunks = written_chunks(
        progress_indicator(
            resp_read(CONTENT_CHUNK_SIZE),
            CONTENT_CHUNK_SIZE
        )
    )
    if hashes:
        hashes.check_against_chunks(downloaded_chunks)
    else:
        consume(downloaded_chunks)


def _copy_file(filename, location, link):
    copy = True
    download_location = os.path.join(location, link.filename)
    if os.path.exists(download_location):
        response = ask_path_exists(
            'The file %s exists. (i)gnore, (w)ipe, (b)ackup, (a)abort' %
            display_path(download_location), ('i', 'w', 'b', 'a'))
        if response == 'i':
            copy = False
        elif response == 'w':
            logger.warning('Deleting %s', display_path(download_location))
            os.remove(download_location)
        elif response == 'b':
            dest_file = backup_dir(download_location)
            logger.warning(
                'Backing up %s to %s',
                display_path(download_location),
                display_path(dest_file),
            )
            shutil.move(download_location, dest_file)
        elif response == 'a':
            sys.exit(-1)
    if copy:
        shutil.copy(filename, download_location)
        logger.info('Saved %s', display_path(download_location))


def unpack_http_url(
    link,  # type: Link
    location,  # type: str
    download_dir=None,  # type: Optional[str]
    session=None,  # type: Optional[PipSession]
    hashes=None,  # type: Optional[Hashes]
    progress_bar="on"  # type: str
):
    # type: (...) -> None
    if session is None:
        raise TypeError(
            "unpack_http_url() missing 1 required keyword argument: 'session'"
        )

    with TempDirectory(kind="unpack") as temp_dir:
        # If a download dir is specified, is the file already downloaded there?
        already_downloaded_path = None
        if download_dir:
            already_downloaded_path = _check_download_dir(link,
                                                          download_dir,
                                                          hashes)

        if already_downloaded_path:
            from_path = already_downloaded_path
            content_type = mimetypes.guess_type(from_path)[0]
        else:
            # let's download to a tmp dir
            from_path, content_type = _download_http_url(link,
                                                         session,
                                                         temp_dir.path,
                                                         hashes,
                                                         progress_bar)

        # unpack the archive to the build dir location. even when only
        # downloading archives, they have to be unpacked to parse dependencies
        unpack_file(from_path, location, content_type)

        # a download dir is specified; let's copy the archive there
        if download_dir and not already_downloaded_path:
            _copy_file(from_path, download_dir, link)

        if not already_downloaded_path:
            os.unlink(from_path)


def _copy2_ignoring_special_files(src, dest):
    # type: (str, str) -> None
    """Copying special files is not supported, but as a convenience to users
    we skip errors copying them. This supports tools that may create e.g.
    socket files in the project source directory.
    """
    try:
        copy2_fixed(src, dest)
    except shutil.SpecialFileError as e:
        # SpecialFileError may be raised due to either the source or
        # destination. If the destination was the cause then we would actually
        # care, but since the destination directory is deleted prior to
        # copy we ignore all of them assuming it is caused by the source.
        logger.warning(
            "Ignoring special file error '%s' encountered copying %s to %s.",
            str(e),
            path_to_display(src),
            path_to_display(dest),
        )


def _copy_source_tree(source, target):
    # type: (str, str) -> None
    def ignore(d, names):
        # Pulling in those directories can potentially be very slow,
        # exclude the following directories if they appear in the top
        # level dir (and only it).
        # See discussion at https://github.com/pypa/pip/pull/6770
        return ['.tox', '.nox'] if d == source else []

    kwargs = dict(ignore=ignore, symlinks=True)  # type: CopytreeKwargs

    if not PY2:
        # Python 2 does not support copy_function, so we only ignore
        # errors on special file copy in Python 3.
        kwargs['copy_function'] = _copy2_ignoring_special_files

    shutil.copytree(source, target, **kwargs)


def unpack_file_url(
    link,  # type: Link
    location,  # type: str
    download_dir=None,  # type: Optional[str]
    hashes=None  # type: Optional[Hashes]
):
    # type: (...) -> None
    """Unpack link into location.

    If download_dir is provided and link points to a file, make a copy
    of the link file inside download_dir.
    """
    link_path = link.file_path
    # If it's a url to a local directory
    if link.is_existing_dir():
        if os.path.isdir(location):
            rmtree(location)
        _copy_source_tree(link_path, location)
        if download_dir:
            logger.info('Link is a directory, ignoring download_dir')
        return

    # If --require-hashes is off, `hashes` is either empty, the
    # link's embedded hash, or MissingHashes; it is required to
    # match. If --require-hashes is on, we are satisfied by any
    # hash in `hashes` matching: a URL-based or an option-based
    # one; no internet-sourced hash will be in `hashes`.
    if hashes:
        hashes.check_against_path(link_path)

    # If a download dir is specified, is the file already there and valid?
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link,
                                                      download_dir,
                                                      hashes)

    if already_downloaded_path:
        from_path = already_downloaded_path
    else:
        from_path = link_path

    content_type = mimetypes.guess_type(from_path)[0]

    # unpack the archive to the build dir location. even when only downloading
    # archives, they have to be unpacked to parse dependencies
    unpack_file(from_path, location, content_type)

    # a download dir is specified and not already downloaded
    if download_dir and not already_downloaded_path:
        _copy_file(from_path, download_dir, link)


def unpack_url(
    link,  # type: Link
    location,  # type: str
    download_dir=None,  # type: Optional[str]
    session=None,  # type: Optional[PipSession]
    hashes=None,  # type: Optional[Hashes]
    progress_bar="on"  # type: str
):
    # type: (...) -> None
    """Unpack link.
       If link is a VCS link:
         if only_download, export into download_dir and ignore location
          else unpack into location
       for other types of link:
         - unpack into location
         - if download_dir, copy the file into download_dir
         - if only_download, mark location for deletion

    :param hashes: A Hashes object, one of whose embedded hashes must match,
        or HashMismatch will be raised. If the Hashes is empty, no matches are
        required, and unhashable types of requirements (like VCS ones, which
        would ordinarily raise HashUnsupported) are allowed.
    """
    # non-editable vcs urls
    if link.is_vcs:
        unpack_vcs_link(link, location)

    # file urls
    elif link.is_file:
        unpack_file_url(link, location, download_dir, hashes=hashes)

    # http urls
    else:
        if session is None:
            session = PipSession()

        unpack_http_url(
            link,
            location,
            download_dir,
            session,
            hashes=hashes,
            progress_bar=progress_bar
        )


def sanitize_content_filename(filename):
    # type: (str) -> str
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition, default_filename):
    # type: (str, str) -> str
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    _type, params = cgi.parse_header(content_disposition)
    filename = params.get('filename')
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(filename)
    return filename or default_filename


def _download_http_url(
    link,  # type: Link
    session,  # type: PipSession
    temp_dir,  # type: str
    hashes,  # type: Optional[Hashes]
    progress_bar  # type: str
):
    # type: (...) -> Tuple[str, str]
    """Download link url into temp_dir using provided session"""
    target_url = link.url.split('#', 1)[0]
    try:
        resp = session.get(
            target_url,
            # We use Accept-Encoding: identity here because requests
            # defaults to accepting compressed responses. This breaks in
            # a variety of ways depending on how the server is configured.
            # - Some servers will notice that the file isn't a compressible
            #   file and will leave the file alone and with an empty
            #   Content-Encoding
            # - Some servers will notice that the file is already
            #   compressed and will leave the file alone and will add a
            #   Content-Encoding: gzip header
            # - Some servers won't notice anything at all and will take
            #   a file that's already been compressed and compress it again
            #   and set the Content-Encoding: gzip header
            # By setting this to request only the identity encoding We're
            # hoping to eliminate the third case. Hopefully there does not
            # exist a server which when given a file will notice it is
            # already compressed and that you're not asking for a
            # compressed file and will then decompress it before sending
            # because if that's the case I don't think it'll ever be
            # possible to make this work.
            headers={"Accept-Encoding": "identity"},
            stream=True,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.critical(
            "HTTP error %s while getting %s", exc.response.status_code, link,
        )
        raise

    content_type = resp.headers.get('content-type', '')
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get('content-disposition')
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext = splitext(filename)[1]  # type: Optional[str]
    if not ext:
        ext = mimetypes.guess_extension(content_type)
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, 'wb') as content_file:
        _download_url(resp, link, content_file, hashes, progress_bar)
    return file_path, content_type


def _check_download_dir(link, download_dir, hashes):
    # type: (Link, str, Optional[Hashes]) -> Optional[str]
    """ Check download_dir for previously downloaded file with correct hash
        If a correct file is found return its path else None
    """
    download_path = os.path.join(download_dir, link.filename)

    if not os.path.exists(download_path):
        return None

    # If already downloaded, does its hash match?
    logger.info('File was already downloaded %s', download_path)
    if hashes:
        try:
            hashes.check_against_path(download_path)
        except HashMismatch:
            logger.warning(
                'Previously-downloaded file %s has bad hash. '
                'Re-downloading.',
                download_path
            )
            os.unlink(download_path)
            return None
    return download_path


class RequirementPreparer(object):
    """Prepares a Requirement
    """

    def __init__(
        self,
        build_dir,  # type: str
        download_dir,  # type: Optional[str]
        src_dir,  # type: str
        wheel_download_dir,  # type: Optional[str]
        progress_bar,  # type: str
        build_isolation,  # type: bool
        req_tracker  # type: RequirementTracker
    ):
        # type: (...) -> None
        super(RequirementPreparer, self).__init__()

        self.src_dir = src_dir
        self.build_dir = build_dir
        self.req_tracker = req_tracker

        # Where still-packed archives should be written to. If None, they are
        # not saved, and are deleted immediately after unpacking.
        if download_dir:
            download_dir = expanduser(download_dir)
        self.download_dir = download_dir

        # Where still-packed .whl files should be written to. If None, they are
        # written to the download_dir parameter. Separate to download_dir to
        # permit only keeping wheel archives for pip wheel.
        if wheel_download_dir:
            wheel_download_dir = normalize_path(wheel_download_dir)
        self.wheel_download_dir = wheel_download_dir

        # NOTE
        # download_dir and wheel_download_dir overlap semantically and may
        # be combined if we're willing to have non-wheel archives present in
        # the wheelhouse output by 'pip wheel'.

        self.progress_bar = progress_bar

        # Is build isolation allowed?
        self.build_isolation = build_isolation

    @property
    def _download_should_save(self):
        # type: () -> bool
        if not self.download_dir:
            return False

        if os.path.exists(self.download_dir):
            return True

        logger.critical('Could not find download directory')
        raise InstallationError(
            "Could not find or access download directory '{}'"
            .format(self.download_dir))

    def prepare_linked_requirement(
        self,
        req,  # type: InstallRequirement
        session,  # type: PipSession
        finder,  # type: PackageFinder
        require_hashes,  # type: bool
    ):
        # type: (...) -> AbstractDistribution
        """Prepare a requirement that would be obtained from req.link
        """
        assert req.link
        link = req.link

        # TODO: Breakup into smaller functions
        if link.scheme == 'file':
            path = link.file_path
            logger.info('Processing %s', display_path(path))
        else:
            logger.info('Collecting %s', req.req or req)

        with indent_log():
            # @@ if filesystem packages are not marked
            # editable in a req, a non deterministic error
            # occurs when the script attempts to unpack the
            # build directory
            req.ensure_has_source_dir(self.build_dir)
            # If a checkout exists, it's unwise to keep going.  version
            # inconsistencies are logged later, but do not fail the
            # installation.
            # FIXME: this won't upgrade when there's an existing
            # package unpacked in `req.source_dir`
            if os.path.exists(os.path.join(req.source_dir, 'setup.py')):
                raise PreviousBuildDirError(
                    "pip can't proceed with requirements '{}' due to a"
                    " pre-existing build directory ({}). This is "
                    "likely due to a previous installation that failed"
                    ". pip is being responsible and not assuming it "
                    "can delete this. Please delete it and try again."
                    .format(req, req.source_dir)
                )

            # Now that we have the real link, we can tell what kind of
            # requirements we have and raise some more informative errors
            # than otherwise. (For example, we can raise VcsHashUnsupported
            # for a VCS URL rather than HashMissing.)
            if require_hashes:
                # We could check these first 2 conditions inside
                # unpack_url and save repetition of conditions, but then
                # we would report less-useful error messages for
                # unhashable requirements, complaining that there's no
                # hash provided.
                if link.is_vcs:
                    raise VcsHashUnsupported()
                elif link.is_existing_dir():
                    raise DirectoryUrlHashUnsupported()
                if not req.original_link and not req.is_pinned:
                    # Unpinned packages are asking for trouble when a new
                    # version is uploaded. This isn't a security check, but
                    # it saves users a surprising hash mismatch in the
                    # future.
                    #
                    # file:/// URLs aren't pinnable, so don't complain
                    # about them not being pinned.
                    raise HashUnpinned()

            hashes = req.hashes(trust_internet=not require_hashes)
            if require_hashes and not hashes:
                # Known-good hashes are missing for this requirement, so
                # shim it with a facade object that will provoke hash
                # computation and then raise a HashMissing exception
                # showing the user what the hash should be.
                hashes = MissingHashes()

            download_dir = self.download_dir
            if link.is_wheel and self.wheel_download_dir:
                # when doing 'pip wheel` we download wheels to a
                # dedicated dir.
                download_dir = self.wheel_download_dir

            try:
                unpack_url(
                    link, req.source_dir, download_dir,
                    session=session, hashes=hashes,
                    progress_bar=self.progress_bar
                )
            except requests.HTTPError as exc:
                logger.critical(
                    'Could not install requirement %s because of error %s',
                    req,
                    exc,
                )
                raise InstallationError(
                    'Could not install requirement {} because of HTTP '
                    'error {} for URL {}'.format(req, exc, link)
                )

            if link.is_wheel:
                if download_dir:
                    # When downloading, we only unpack wheels to get
                    # metadata.
                    autodelete_unpacked = True
                else:
                    # When installing a wheel, we use the unpacked
                    # wheel.
                    autodelete_unpacked = False
            else:
                # We always delete unpacked sdists after pip runs.
                autodelete_unpacked = True
            if autodelete_unpacked:
                write_delete_marker_file(req.source_dir)

            abstract_dist = _get_prepared_distribution(
                req, self.req_tracker, finder, self.build_isolation,
            )

            if self._download_should_save:
                # Make a .zip of the source_dir we already created.
                if link.is_vcs:
                    req.archive(self.download_dir)
        return abstract_dist

    def prepare_editable_requirement(
        self,
        req,  # type: InstallRequirement
        require_hashes,  # type: bool
        use_user_site,  # type: bool
        finder  # type: PackageFinder
    ):
        # type: (...) -> AbstractDistribution
        """Prepare an editable requirement
        """
        assert req.editable, "cannot prepare a non-editable req as editable"

        logger.info('Obtaining %s', req)

        with indent_log():
            if require_hashes:
                raise InstallationError(
                    'The editable requirement {} cannot be installed when '
                    'requiring hashes, because there is no single file to '
                    'hash.'.format(req)
                )
            req.ensure_has_source_dir(self.src_dir)
            req.update_editable(not self._download_should_save)

            abstract_dist = _get_prepared_distribution(
                req, self.req_tracker, finder, self.build_isolation,
            )

            if self._download_should_save:
                req.archive(self.download_dir)
            req.check_if_exists(use_user_site)

        return abstract_dist

    def prepare_installed_requirement(
        self,
        req,  # type: InstallRequirement
        require_hashes,  # type: bool
        skip_reason  # type: str
    ):
        # type: (...) -> AbstractDistribution
        """Prepare an already-installed requirement
        """
        assert req.satisfied_by, "req should have been satisfied but isn't"
        assert skip_reason is not None, (
            "did not get skip reason skipped but req.satisfied_by "
            "is set to {}".format(req.satisfied_by)
        )
        logger.info(
            'Requirement %s: %s (%s)',
            skip_reason, req, req.satisfied_by.version
        )
        with indent_log():
            if require_hashes:
                logger.debug(
                    'Since it is already installed, we are trusting this '
                    'package without checking its hash. To ensure a '
                    'completely repeatable environment, install into an '
                    'empty virtualenv.'
                )
            abstract_dist = InstalledDistribution(req)

        return abstract_dist
