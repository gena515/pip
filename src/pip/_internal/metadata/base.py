import csv
import email.message
import json
import logging
import pathlib
import re
import zipfile
from typing import (
    IO,
    TYPE_CHECKING,
    Collection,
    Container,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pip._vendor.packaging.version import LegacyVersion, Version

from pip._internal.models.direct_url import (
    DIRECT_URL_METADATA_NAME,
    DirectUrl,
    DirectUrlValidationError,
)
from pip._internal.utils.misc import stdlib_pkgs  # TODO: Move definition here.

if TYPE_CHECKING:
    from typing import Protocol

    from pip._vendor.packaging.utils import NormalizedName
else:
    Protocol = object

DistributionVersion = Union[LegacyVersion, Version]

logger = logging.getLogger(__name__)


def _convert_legacy_file(entry: Tuple[str, ...], info: Tuple[str, ...]) -> pathlib.Path:
    """Convert a legacy installed-files.txt path into modern RECORD path.

    The legacy format stores paths relative to the info directory, while the
    modern format stores paths relative to the package root, e.g. the
    site-packages directory.

    :param entry: Path parts of the installed-files.txt entry.
    :param info: Path parts of the egg-info directory relative to package root.
    :returns: The converted entry.

    For best compatibility with symlinks, this does not use ``abspath()`` or
    ``Path.resolve()``, but tries to work with path parts:

    1. While ``entry`` starts with ``..``, remove the equal amounts of parts
       from ``info``; if ``info`` is empty, start appending ``..`` instead.
    2. Join the two directly.
    """
    while entry and entry[0] == "..":
        if not info or info[-1] == "..":
            info += ("..",)
        else:
            info = info[:-1]
        entry = entry[1:]
    return pathlib.Path(*info, *entry)


class BaseEntryPoint(Protocol):
    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def value(self) -> str:
        raise NotImplementedError()

    @property
    def group(self) -> str:
        raise NotImplementedError()


class BaseDistribution(Protocol):
    def __repr__(self) -> str:
        return f"{self.raw_name} {self.version} ({self.location})"

    def __str__(self) -> str:
        return f"{self.raw_name} {self.version}"

    @property
    def location(self) -> Optional[str]:
        """Where the distribution is loaded from.

        A string value is not necessarily a filesystem path, since distributions
        can be loaded from other sources, e.g. arbitrary zip archives. ``None``
        means the distribution is created in-memory.

        Do not canonicalize this value with e.g. ``pathlib.Path.resolve()``. If
        this is a symbolic link, we want to preserve the relative path between
        it and files in the distribution.
        """
        raise NotImplementedError()

    @property
    def info_directory(self) -> Optional[str]:
        """Location of the .[egg|dist]-info directory.

        Similarly to ``location``, a string value is not necessarily a
        filesystem path. ``None`` means the distribution is created in-memory.

        For a modern .dist-info installation on disk, this should be something
        like ``{location}/{raw_name}-{version}.dist-info``.

        Do not canonicalize this value with e.g. ``pathlib.Path.resolve()``. If
        this is a symbolic link, we want to preserve the relative path between
        it and other files in the distribution.
        """
        raise NotImplementedError()

    @property
    def canonical_name(self) -> "NormalizedName":
        raise NotImplementedError()

    @property
    def version(self) -> DistributionVersion:
        raise NotImplementedError()

    @property
    def direct_url(self) -> Optional[DirectUrl]:
        """Obtain a DirectUrl from this distribution.

        Returns None if the distribution has no `direct_url.json` metadata,
        or if `direct_url.json` is invalid.
        """
        try:
            content = self.read_text(DIRECT_URL_METADATA_NAME)
        except FileNotFoundError:
            return None
        try:
            return DirectUrl.from_json(content)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            DirectUrlValidationError,
        ) as e:
            logger.warning(
                "Error parsing %s for %s: %s",
                DIRECT_URL_METADATA_NAME,
                self.canonical_name,
                e,
            )
            return None

    @property
    def installer(self) -> str:
        raise NotImplementedError()

    @property
    def egg_link(self) -> Optional[str]:
        """Location of the ``.egg-link`` for this distribution.

        If there's not a matching file, None is returned. Note that finding this
        file does not necessarily mean the currently-installed distribution is
        editable since the ``.egg-link`` can still be shadowed by a
        non-editable installation located in front of it in ``sys.path``.
        """
        raise NotImplementedError()

    @property
    def editable(self) -> bool:
        raise NotImplementedError()

    @property
    def local(self) -> bool:
        raise NotImplementedError()

    @property
    def in_usersite(self) -> bool:
        raise NotImplementedError()

    @property
    def in_site_packages(self) -> bool:
        raise NotImplementedError()

    def read_text(self, name: str) -> str:
        """Read a file in the .dist-info (or .egg-info) directory.

        :raises FileNotFoundError: ``name`` does not exist in the info directory.
        """
        raise NotImplementedError()

    def iterdir(self, name: str) -> Iterable[pathlib.PurePosixPath]:
        """Iterate through a directory in the info directory.

        Each item is a path relative to the info directory.

        :raises FileNotFoundError: ``name`` does not exist in the info directory.
        :raises NotADirectoryError: ``name`` exists in the info directory, but
            is not a directory.
        """

    def iter_entry_points(self) -> Iterable[BaseEntryPoint]:
        raise NotImplementedError()

    @property
    def metadata(self) -> email.message.Message:
        """Metadata of distribution parsed from e.g. METADATA or PKG-INFO."""
        raise NotImplementedError()

    @property
    def metadata_version(self) -> Optional[str]:
        """Value of "Metadata-Version:" in distribution metadata, if available."""
        return self.metadata.get("Metadata-Version")

    @property
    def raw_name(self) -> str:
        """Value of "Name:" in distribution metadata."""
        # The metadata should NEVER be missing the Name: key, but if it somehow
        # does, fall back to the known canonical name.
        return self.metadata.get("Name", self.canonical_name)

    @property
    def requires_python(self) -> SpecifierSet:
        """Value of "Requires-Python:" in distribution metadata.

        If the key does not exist or contains an invalid value, an empty
        SpecifierSet should be returned.
        """
        value = self.metadata.get("Requires-Python")
        if value is None:
            return SpecifierSet()
        try:
            # Convert to str to satisfy the type checker; this can be a Header object.
            spec = SpecifierSet(str(value))
        except InvalidSpecifier as e:
            message = "Package %r has an invalid Requires-Python: %s"
            logger.warning(message, self.raw_name, e)
            return SpecifierSet()
        return spec

    def iter_dependencies(self, extras: Collection[str] = ()) -> Iterable[Requirement]:
        """Dependencies of this distribution.

        For modern .dist-info distributions, this is the collection of
        "Requires-Dist:" entries in distribution metadata.
        """
        raise NotImplementedError()

    def iter_provided_extras(self) -> Iterable[str]:
        """Extras provided by this distribution.

        For modern .dist-info distributions, this is the collection of
        "Provides-Extra:" entries in distribution metadata.
        """
        raise NotImplementedError()

    def _iter_files_from_legacy(self) -> Optional[Iterator[pathlib.Path]]:
        try:
            text = self.read_text("installed-files.txt")
        except FileNotFoundError:
            return None
        paths = (pathlib.Path(p) for p in text.splitlines(keepends=False) if p)
        root = self.location
        info = self.info_directory
        if root is None or info is None:
            return paths
        try:
            rel = pathlib.Path(info).relative_to(root)
        except ValueError:  # info is not relative to root.
            return paths
        if not rel.parts:  # info *is* root.
            return paths
        return (_convert_legacy_file(p.parts, rel.parts) for p in paths)

    def _iter_files_from_record(self) -> Optional[Iterator[pathlib.Path]]:
        try:
            text = self.read_text("RECORD")
        except FileNotFoundError:
            return None
        return (pathlib.Path(row[0]) for row in csv.reader(text.splitlines()))

    def iter_files(self) -> Optional[Iterator[pathlib.Path]]:
        """Files in the distribution's record.

        For modern .dist-info distributions, this is the files listed in the
        ``RECORD`` file. All entries are paths relative to this distribution's
        ``location``.

        Note that this can be None for unmanagable distributions, e.g. an
        installation performed by distutils or a foreign package manager.
        """
        return self._iter_files_from_record() or self._iter_files_from_legacy()


class BaseEnvironment:
    """An environment containing distributions to introspect."""

    @classmethod
    def default(cls) -> "BaseEnvironment":
        raise NotImplementedError()

    @classmethod
    def from_paths(cls, paths: Optional[List[str]]) -> "BaseEnvironment":
        raise NotImplementedError()

    def get_distribution(self, name: str) -> Optional["BaseDistribution"]:
        """Given a requirement name, return the installed distributions."""
        raise NotImplementedError()

    def _iter_distributions(self) -> Iterator["BaseDistribution"]:
        """Iterate through installed distributions.

        This function should be implemented by subclass, but never called
        directly. Use the public ``iter_distribution()`` instead, which
        implements additional logic to make sure the distributions are valid.
        """
        raise NotImplementedError()

    def iter_distributions(self) -> Iterator["BaseDistribution"]:
        """Iterate through installed distributions."""
        for dist in self._iter_distributions():
            # Make sure the distribution actually comes from a valid Python
            # packaging distribution. Pip's AdjacentTempDirectory leaves folders
            # e.g. ``~atplotlib.dist-info`` if cleanup was interrupted. The
            # valid project name pattern is taken from PEP 508.
            project_name_valid = re.match(
                r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$",
                dist.canonical_name,
                flags=re.IGNORECASE,
            )
            if not project_name_valid:
                logger.warning(
                    "Ignoring invalid distribution %s (%s)",
                    dist.canonical_name,
                    dist.location,
                )
                continue
            yield dist

    def iter_installed_distributions(
        self,
        local_only: bool = True,
        skip: Container[str] = stdlib_pkgs,
        include_editables: bool = True,
        editables_only: bool = False,
        user_only: bool = False,
    ) -> Iterator[BaseDistribution]:
        """Return a list of installed distributions.

        :param local_only: If True (default), only return installations
        local to the current virtualenv, if in a virtualenv.
        :param skip: An iterable of canonicalized project names to ignore;
            defaults to ``stdlib_pkgs``.
        :param include_editables: If False, don't report editables.
        :param editables_only: If True, only report editables.
        :param user_only: If True, only report installations in the user
        site directory.
        """
        it = self.iter_distributions()
        if local_only:
            it = (d for d in it if d.local)
        if not include_editables:
            it = (d for d in it if not d.editable)
        if editables_only:
            it = (d for d in it if d.editable)
        if user_only:
            it = (d for d in it if d.in_usersite)
        return (d for d in it if d.canonical_name not in skip)


class Wheel(Protocol):
    location: str

    def as_zipfile(self) -> zipfile.ZipFile:
        raise NotImplementedError()


class FilesystemWheel(Wheel):
    def __init__(self, location: str) -> None:
        self.location = location

    def as_zipfile(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(self.location, allowZip64=True)


class MemoryWheel(Wheel):
    def __init__(self, location: str, stream: IO[bytes]) -> None:
        self.location = location
        self.stream = stream

    def as_zipfile(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(self.stream, allowZip64=True)
