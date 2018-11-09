import os
import os.path

from pip._internal.utils.compat import get_path_uid


def check_path_owner(path):
    # type: (str) -> bool
    # If we don't have a way to check the effective uid of this process, then
    # we'll just assume that we own the directory.
    if not hasattr(os, "geteuid"):
        return True

    previous = None
    while path != previous:
        if not os.path.lexists(path):
            previous, path = path, os.path.dirname(path)
        else:
            # Check if path is not writable by current user.
            if os.geteuid() != 0:
                return os.access(path, os.W_OK)

            # Special handling for root user in order to handle properly
            # cases where users use sudo without -H flag.
            try:
                path_uid = get_path_uid(path)
            except OSError:
                return False
            else:
                return path_uid == 0
    return False  # assume we don't own the path
