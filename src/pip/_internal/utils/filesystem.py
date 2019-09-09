import os
import os.path
import random
import shutil
import stat
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

# NOTE: retrying is not annotated in typeshed as on 2017-07-17, which is
#       why we ignore the type on this import.
from pip._vendor.retrying import retry  # type: ignore
from pip._vendor.six import PY2

from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import cast
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import BinaryIO, Iterator

    class NamedTemporaryFileResult(BinaryIO):
        @property
        def file(self):
            # type: () -> BinaryIO
            pass


def check_path_owner(path):
    # type: (str) -> bool
    # If we don't have a way to check the effective uid of this process, then
    # we'll just assume that we own the directory.
    if not hasattr(os, "geteuid"):
        return True

    previous = None
    while path != previous:
        if os.path.lexists(path):
            # Check if path is writable by current user.
            if os.geteuid() == 0:
                # Special handling for root user in order to handle properly
                # cases where users use sudo without -H flag.
                try:
                    path_uid = get_path_uid(path)
                except OSError:
                    return False
                return path_uid == 0
            else:
                return os.access(path, os.W_OK)
        else:
            previous, path = path, os.path.dirname(path)
    return False  # assume we don't own the path


def copy2_fixed(src, dest):
    # type: (str, str) -> None
    """Wrap shutil.copy2() but map errors copying socket files to
    SpecialFileError as expected.

    See also https://bugs.python.org/issue37700.
    """
    try:
        shutil.copy2(src, dest)
    except (OSError, IOError):
        for f in [src, dest]:
            try:
                is_socket_file = is_socket(f)
            except OSError:
                # An error has already occurred. Another error here is not
                # a problem and we can ignore it.
                pass
            else:
                if is_socket_file:
                    raise shutil.SpecialFileError("`%s` is a socket" % f)

        raise


def is_socket(path):
    # type: (str) -> bool
    return stat.S_ISSOCK(os.lstat(path).st_mode)


@contextmanager
def adjacent_tmp_file(path):
    # type: (str) -> Iterator[NamedTemporaryFileResult]
    """Given a path to a file, open a temp file next to it securely and ensure
    it is written to disk after the context reaches its end.
    """
    with NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(path),
        prefix=os.path.basename(path),
        suffix='.tmp',
    ) as f:
        result = cast('NamedTemporaryFileResult', f)
        try:
            yield result
        finally:
            result.file.flush()
            os.fsync(result.file.fileno())


_replace_retry = retry(stop_max_delay=1000, wait_fixed=250)

if PY2:
    @_replace_retry
    def replace(src, dest):
        # type: (str, str) -> None
        try:
            os.rename(src, dest)
        except OSError:
            os.remove(dest)
            os.rename(src, dest)

else:
    replace = _replace_retry(os.replace)


# test_writable_dir and _test_writable_dir_win are copied from Flit,
# with the author's agreement to also place them under pip's license.
def test_writable_dir(path):
    """Check if a directory is writable.

    Uses os.access() on POSIX, tries creating files on Windows.
    """
    if os.name == 'posix':
        return os.access(path, os.W_OK)

    return _test_writable_dir_win(path)


def _test_writable_dir_win(path):
    # os.access doesn't work on Windows: http://bugs.python.org/issue2528
    # and we can't use tempfile: http://bugs.python.org/issue22107
    basename = 'accesstest_deleteme_fishfingers_custard_'
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    for i in range(10):
        name = basename + ''.join(random.choice(alphabet) for _ in range(6))
        file = os.path.join(path, name)
        try:
            with open(file, mode='xb'):
                pass
        except FileExistsError:
            continue
        except PermissionError:
            # This could be because there's a directory with the same name.
            # But it's highly unlikely there's a directory called that,
            # so we'll assume it's because the parent directory is not writable.
            return False
        else:
            os.unlink(file)
            return True

    # This should never be reached
    raise EnvironmentError('Unexpected condition testing for writable directory')
