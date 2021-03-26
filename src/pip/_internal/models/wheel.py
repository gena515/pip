"""Represents a wheel file and provides access to the various parts of the
name that have meaning.
"""
import re
from typing import List

from pip._vendor.packaging.tags import Tag

from pip._internal.exceptions import InvalidWheelFilename


class Wheel:
    """A wheel file"""

    wheel_file_re = re.compile(
        r"""^(?P<namever>(?P<name>.+?)-(?P<ver>.*?))
        ((-(?P<build>\d[^-]*?))?-(?P<pyver>.+?)-(?P<abi>.+?)-(?P<plat>.+?)
        \.whl|\.dist-info)$""",
        re.VERBOSE
    )

    def __init__(self, filename):
        # type: (str) -> None
        """
        :raises InvalidWheelFilename: when the filename is invalid for a wheel
        """
        wheel_info = self.wheel_file_re.match(filename)
        if not wheel_info:
            raise InvalidWheelFilename(
                f"{filename} is not a valid wheel filename."
            )
        self.filename = filename
        self.name = wheel_info.group('name').replace('_', '-')
        # we'll assume "_" means "-" due to wheel naming scheme
        # (https://github.com/pypa/pip/issues/1150)
        self.version = wheel_info.group('ver').replace('_', '-')
        self.build_tag = wheel_info.group('build')
        self.pyversions = wheel_info.group('pyver').split('.')
        self.abis = wheel_info.group('abi').split('.')
        self.plats = wheel_info.group('plat').split('.')

        # All the tag combinations from this file
        self.file_tags = {
            Tag(x, y, z) for x in self.pyversions
            for y in self.abis for z in self.plats
        }

    def get_formatted_file_tags(self):
        # type: () -> List[str]
        """Return the wheel's tags as a sorted list of strings."""
        return sorted(str(tag) for tag in self.file_tags)

    def support_index_min(self, tags):
        # type: (List[Tag]) -> int
        """Return the lowest index that one of the wheel's file_tag combinations
        achieves in the given list of supported tags.

        For example, if there are 8 supported tags and one of the file tags
        is first in the list, then return 0.

        :param tags: the PEP 425 tags to check the wheel against, in order
            with most preferred first.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        return min(tags.index(tag) for tag in self.file_tags if tag in tags)

    def support_index_min_fast(self, tags, tag_to_idx):
        return min(tag_to_idx[tag] for tag in self.file_tags if tag in tag_to_idx)

    def supported(self, tags):
        # type: (List[Tag]) -> bool
        """Return whether the wheel is compatible with one of the given tags.

        :param tags: the PEP 425 tags to check the wheel against.
        """
        # not disjoint means has some overlap
        # tags is a list (and long)
        # file tags is a set (and short)
        # print("len(tags)", len(tags), type(tags))
        # print("len(file_tags)", len(self.file_tags), type(self.file_tags))
        try:
            return bool(self.file_tags & tags)
        except TypeError:
            return bool(self.file_tags & set(tags))
