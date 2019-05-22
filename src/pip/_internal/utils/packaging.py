from __future__ import absolute_import

import logging
from email.parser import FeedParser

from pip._vendor import pkg_resources
from pip._vendor.packaging import specifiers, version

from pip._internal import exceptions
from pip._internal.utils.misc import display_path
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Optional, Tuple
    from email.message import Message
    from pip._vendor.pkg_resources import Distribution


logger = logging.getLogger(__name__)


def check_requires_python(requires_python, version_info):
    # type: (Optional[str], Tuple[int, ...]) -> bool
    """
    Check if the given Python version matches a `requires_python` specifier.

    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check (e.g. `sys.version_info[:3]`).

    Returns `True` if the version of python in use matches the requirement.
    Returns `False` if the version of python in use does not matches the
    requirement.

    Raises an InvalidSpecifier if `requires_python` have an invalid format.
    """
    if requires_python is None:
        # The package provides no information
        return True
    requires_python_specifier = specifiers.SpecifierSet(requires_python)

    python_version = version.parse('.'.join(map(str, version_info)))
    return python_version in requires_python_specifier


def get_metadata(dist):
    # type: (Distribution) -> Message
    if (isinstance(dist, pkg_resources.DistInfoDistribution) and
            dist.has_metadata('METADATA')):
        metadata = dist.get_metadata('METADATA')
    elif dist.has_metadata('PKG-INFO'):
        metadata = dist.get_metadata('PKG-INFO')
    else:
        logger.warning("No metadata found in %s", display_path(dist.location))
        metadata = ''

    feed_parser = FeedParser()
    feed_parser.feed(metadata)
    return feed_parser.close()


def check_dist_requires_python(dist, version_info):
    """
    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check (e.g. `sys.version_info[:3]`).
    """
    pkg_info_dict = get_metadata(dist)
    requires_python = pkg_info_dict.get('Requires-Python')
    try:
        if not check_requires_python(
            requires_python, version_info=version_info,
        ):
            raise exceptions.UnsupportedPythonVersion(
                "%s requires Python '%s' but the running Python is %s" % (
                    dist.project_name,
                    requires_python,
                    '.'.join(map(str, version_info)),)
            )
    except specifiers.InvalidSpecifier as e:
        logger.warning(
            "Package %s has an invalid Requires-Python entry %s - %s",
            dist.project_name, requires_python, e,
        )
        return


def get_installer(dist):
    # type: (Distribution) -> str
    if dist.has_metadata('INSTALLER'):
        for line in dist.get_metadata_lines('INSTALLER'):
            if line.strip():
                return line.strip()
    return ''
