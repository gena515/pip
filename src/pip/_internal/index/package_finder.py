"""Routines related to PyPI, indexes"""

import binascii
import bz2
import datetime
import enum
import functools
import itertools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import InvalidVersion, _BaseVersion
from pip._vendor.packaging.version import parse as parse_version

from pip._internal.cache import FetchResolveCache, SerializableEntry
from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
    InvalidWheelFilename,
    UnsupportedWheel,
)
from pip._internal.index.collector import IndexContent, LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link, PersistentLinkCacheArgs
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import Wheel
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging import check_requires_python
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from pip._vendor.typing_extensions import TypeGuard

__all__ = ["FormatControl", "BestCandidateResult", "PackageFinder"]


logger = getLogger(__name__)

BuildTag = Union[Tuple[()], Tuple[int, str]]
CandidateSortingKey = Tuple[int, int, int, _BaseVersion, Optional[int], BuildTag]


def _check_link_requires_python(
    link: Link,
    version_info: Tuple[int, int, int],
    ignore_requires_python: bool = False,
) -> bool:
    """
    Return whether the given Python version is compatible with a link's
    "Requires-Python" value.

    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check.
    :param ignore_requires_python: Whether to ignore the "Requires-Python"
        value if the given Python version isn't compatible.
    """
    try:
        is_compatible = check_requires_python(
            link.requires_python,
            version_info=version_info,
        )
    except specifiers.InvalidSpecifier:
        logger.debug(
            "Ignoring invalid Requires-Python (%r) for link: %s",
            link.requires_python,
            link,
        )
    else:
        if not is_compatible:
            version = ".".join(map(str, version_info))
            if not ignore_requires_python:
                logger.verbose(
                    "Link requires a different Python (%s not in: %r): %s",
                    version,
                    link.requires_python,
                    link,
                )
                return False

            logger.debug(
                "Ignoring failed Requires-Python check (%s not in: %r) for link: %s",
                version,
                link.requires_python,
                link,
            )

    return True


class LinkType(enum.Enum):
    candidate = enum.auto()
    different_project = enum.auto()
    yanked = enum.auto()
    format_unsupported = enum.auto()
    format_invalid = enum.auto()
    platform_mismatch = enum.auto()
    requires_python_mismatch = enum.auto()


class LinkEvaluator(SerializableEntry):
    """
    Responsible for evaluating links for a particular project.
    """

    @classmethod
    def suffix(cls) -> str:
        return ".evaluation"

    _py_version_re = re.compile(r"-py([123]\.?[0-9]?)$")

    def serialize(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "canonical_name": self._canonical_name,
            # Sort these for determinism.
            "formats": sorted(self._formats),
            "target_python": self._target_python.format_given(),
            "allow_yanked": self._allow_yanked,
            "ignore_requires_python": self._ignore_requires_python,
        }

    # Don't include an allow_yanked default value to make sure each call
    # site considers whether yanked releases are allowed. This also causes
    # that decision to be made explicit in the calling code, which helps
    # people when reading the code.
    def __init__(
        self,
        project_name: str,
        canonical_name: str,
        formats: FrozenSet[str],
        target_python: TargetPython,
        allow_yanked: bool,
        ignore_requires_python: Optional[bool] = None,
    ) -> None:
        """
        :param project_name: The user supplied package name.
        :param canonical_name: The canonical package name.
        :param formats: The formats allowed for this package. Should be a set
            with 'binary' or 'source' or both in it.
        :param target_python: The target Python interpreter to use when
            evaluating link compatibility. This is used, for example, to
            check wheel compatibility, as well as when checking the Python
            version, e.g. the Python version embedded in a link filename
            (or egg fragment) and against an HTML link's optional PEP 503
            "data-requires-python" attribute.
        :param allow_yanked: Whether files marked as yanked (in the sense
            of PEP 592) are permitted to be candidates for install.
        :param ignore_requires_python: Whether to ignore incompatible
            PEP 503 "data-requires-python" values in HTML links. Defaults
            to False.
        """
        if ignore_requires_python is None:
            ignore_requires_python = False

        self._allow_yanked = allow_yanked
        self._canonical_name = canonical_name
        self._ignore_requires_python = ignore_requires_python
        self._formats = formats
        self._target_python = target_python

        self.project_name = project_name

    def evaluate_link(self, link: Link) -> Tuple[LinkType, str]:
        """
        Determine whether a link is a candidate for installation.

        :return: A tuple (result, detail), where *result* is an enum
            representing whether the evaluation found a candidate, or the reason
            why one is not found. If a candidate is found, *detail* will be the
            candidate's version string; if one is not found, it contains the
            reason the link fails to qualify.
        """
        version = None
        if link.is_yanked and not self._allow_yanked:
            reason = link.yanked_reason or "<none given>"
            return (LinkType.yanked, f"yanked for reason: {reason}")

        if link.egg_fragment:
            egg_info = link.egg_fragment
            ext = link.ext
        else:
            egg_info, ext = link.splitext()
            if not ext:
                return (LinkType.format_unsupported, "not a file")
            if ext not in SUPPORTED_EXTENSIONS:
                return (
                    LinkType.format_unsupported,
                    f"unsupported archive format: {ext}",
                )
            if "binary" not in self._formats and ext == WHEEL_EXTENSION:
                reason = f"No binaries permitted for {self.project_name}"
                return (LinkType.format_unsupported, reason)
            if "macosx10" in link.path and ext == ".zip":
                return (LinkType.format_unsupported, "macosx10 one")
            if ext == WHEEL_EXTENSION:
                try:
                    wheel = Wheel(link.filename)
                except InvalidWheelFilename:
                    return (
                        LinkType.format_invalid,
                        "invalid wheel filename",
                    )
                if canonicalize_name(wheel.name) != self._canonical_name:
                    reason = f"wrong project name (not {self.project_name})"
                    return (LinkType.different_project, reason)

                supported_tags = self._target_python.get_unsorted_tags()
                if not wheel.supported(supported_tags):
                    # Include the wheel's tags in the reason string to
                    # simplify troubleshooting compatibility issues.
                    file_tags = ", ".join(wheel.get_formatted_file_tags())
                    reason = (
                        f"none of the wheel's tags ({file_tags}) are compatible "
                        f"(run pip debug --verbose to show compatible tags)"
                    )
                    return (LinkType.platform_mismatch, reason)

                version = wheel.version

        # This should be up by the self.ok_binary check, but see issue 2700.
        if "source" not in self._formats and ext != WHEEL_EXTENSION:
            reason = f"No sources permitted for {self.project_name}"
            return (LinkType.format_unsupported, reason)

        if not version:
            version = _extract_version_from_fragment(
                egg_info,
                self._canonical_name,
            )
        if not version:
            reason = f"Missing project version for {self.project_name}"
            return (LinkType.format_invalid, reason)

        match = self._py_version_re.search(version)
        if match:
            version = version[: match.start()]
            py_version = match.group(1)
            if py_version != self._target_python.py_version:
                return (
                    LinkType.platform_mismatch,
                    "Python version is incorrect",
                )

        supports_python = _check_link_requires_python(
            link,
            version_info=self._target_python.py_version_info,
            ignore_requires_python=self._ignore_requires_python,
        )
        if not supports_python:
            reason = f"{version} Requires-Python {link.requires_python}"
            return (LinkType.requires_python_mismatch, reason)

        logger.debug("Found link %s, version: %s", link, version)

        return (LinkType.candidate, version)


def filter_unallowed_hashes(
    candidates: List[InstallationCandidate],
    hashes: Optional[Hashes],
    project_name: str,
) -> List[InstallationCandidate]:
    """
    Filter out candidates whose hashes aren't allowed, and return a new
    list of candidates.

    If at least one candidate has an allowed hash, then all candidates with
    either an allowed hash or no hash specified are returned.  Otherwise,
    the given candidates are returned.

    Including the candidates with no hash specified when there is a match
    allows a warning to be logged if there is a more preferred candidate
    with no hash specified.  Returning all candidates in the case of no
    matches lets pip report the hash of the candidate that would otherwise
    have been installed (e.g. permitting the user to more easily update
    their requirements file with the desired hash).
    """
    if not hashes:
        logger.debug(
            "Given no hashes to check %s links for project %r: "
            "discarding no candidates",
            len(candidates),
            project_name,
        )
        # Make sure we're not returning back the given value.
        return list(candidates)

    matches_or_no_digest = []
    # Collect the non-matches for logging purposes.
    non_matches = []
    match_count = 0
    for candidate in candidates:
        link = candidate.link
        if not link.has_hash:
            pass
        elif link.is_hash_allowed(hashes=hashes):
            match_count += 1
        else:
            non_matches.append(candidate)
            continue

        matches_or_no_digest.append(candidate)

    if match_count:
        filtered = matches_or_no_digest
    else:
        # Make sure we're not returning back the given value.
        filtered = list(candidates)

    if len(filtered) == len(candidates):
        discard_message = "discarding no candidates"
    else:
        discard_message = "discarding {} non-matches:\n  {}".format(
            len(non_matches),
            "\n  ".join(str(candidate.link) for candidate in non_matches),
        )

    logger.debug(
        "Checked %s links for project %r against %s hashes "
        "(%s matches, %s no digest): %s",
        len(candidates),
        project_name,
        hashes.digest_count,
        match_count,
        len(matches_or_no_digest) - match_count,
        discard_message,
    )

    return filtered


@dataclass
class CandidatePreferences:
    """
    Encapsulates some of the preferences for filtering and sorting
    InstallationCandidate objects.
    """

    prefer_binary: bool = False
    allow_all_prereleases: bool = False


class BestCandidateResult:
    """A collection of candidates, returned by `PackageFinder.find_best_candidate`.

    This class is only intended to be instantiated by CandidateEvaluator's
    `compute_best_candidate()` method.
    """

    def __init__(
        self,
        candidates: List[InstallationCandidate],
        applicable_candidates: List[InstallationCandidate],
        best_candidate: Optional[InstallationCandidate],
    ) -> None:
        """
        :param candidates: A sequence of all available candidates found.
        :param applicable_candidates: The applicable candidates.
        :param best_candidate: The most preferred candidate found, or None
            if no applicable candidates were found.
        """
        assert set(applicable_candidates) <= set(candidates)

        if best_candidate is None:
            assert not applicable_candidates
        else:
            assert best_candidate in applicable_candidates

        self._applicable_candidates = applicable_candidates
        self._candidates = candidates

        self.best_candidate = best_candidate

    def iter_all(self) -> Iterable[InstallationCandidate]:
        """Iterate through all candidates."""
        return iter(self._candidates)

    def iter_applicable(self) -> Iterable[InstallationCandidate]:
        """Iterate through the applicable candidates."""
        return iter(self._applicable_candidates)


class CandidateEvaluator:
    """
    Responsible for filtering and sorting candidates for installation based
    on what tags are valid.
    """

    @classmethod
    def create(
        cls,
        project_name: str,
        target_python: Optional[TargetPython] = None,
        prefer_binary: bool = False,
        allow_all_prereleases: bool = False,
        specifier: Optional[specifiers.BaseSpecifier] = None,
        hashes: Optional[Hashes] = None,
    ) -> "CandidateEvaluator":
        """Create a CandidateEvaluator object.

        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.
        :param hashes: An optional collection of allowed hashes.
        """
        if target_python is None:
            target_python = TargetPython()
        if specifier is None:
            specifier = specifiers.SpecifierSet()

        supported_tags = target_python.get_sorted_tags()

        return cls(
            project_name=project_name,
            supported_tags=supported_tags,
            specifier=specifier,
            prefer_binary=prefer_binary,
            allow_all_prereleases=allow_all_prereleases,
            hashes=hashes,
        )

    def __init__(
        self,
        project_name: str,
        supported_tags: List[Tag],
        specifier: specifiers.BaseSpecifier,
        prefer_binary: bool = False,
        allow_all_prereleases: bool = False,
        hashes: Optional[Hashes] = None,
    ) -> None:
        """
        :param supported_tags: The PEP 425 tags supported by the target
            Python in order of preference (most preferred first).
        """
        self._allow_all_prereleases = allow_all_prereleases
        self._hashes = hashes
        self._prefer_binary = prefer_binary
        self._project_name = project_name
        self._specifier = specifier
        self._supported_tags = supported_tags
        # Since the index of the tag in the _supported_tags list is used
        # as a priority, precompute a map from tag to index/priority to be
        # used in wheel.find_most_preferred_tag.
        self._wheel_tag_preferences = {
            tag: idx for idx, tag in enumerate(supported_tags)
        }

    def get_applicable_candidates(
        self,
        candidates: List[InstallationCandidate],
    ) -> List[InstallationCandidate]:
        """
        Return the applicable candidates from a list of candidates.
        """
        # Using None infers from the specifier instead.
        allow_prereleases = self._allow_all_prereleases or None
        specifier = self._specifier

        # We turn the version object into a str here because otherwise
        # when we're debundled but setuptools isn't, Python will see
        # packaging.version.Version and
        # pkg_resources._vendor.packaging.version.Version as different
        # types. This way we'll use a str as a common data interchange
        # format. If we stop using the pkg_resources provided specifier
        # and start using our own, we can drop the cast to str().
        candidates_and_versions = [(c, str(c.version)) for c in candidates]
        versions = set(
            specifier.filter(
                (v for _, v in candidates_and_versions),
                prereleases=allow_prereleases,
            )
        )

        applicable_candidates = [c for c, v in candidates_and_versions if v in versions]
        filtered_applicable_candidates = filter_unallowed_hashes(
            candidates=applicable_candidates,
            hashes=self._hashes,
            project_name=self._project_name,
        )

        return sorted(filtered_applicable_candidates, key=self._sort_key)

    def _sort_key(self, candidate: InstallationCandidate) -> CandidateSortingKey:
        """
        Function to pass as the `key` argument to a call to sorted() to sort
        InstallationCandidates by preference.

        Returns a tuple such that tuples sorting as greater using Python's
        default comparison operator are more preferred.

        The preference is as follows:

        First and foremost, candidates with allowed (matching) hashes are
        always preferred over candidates without matching hashes. This is
        because e.g. if the only candidate with an allowed hash is yanked,
        we still want to use that candidate.

        Second, excepting hash considerations, candidates that have been
        yanked (in the sense of PEP 592) are always less preferred than
        candidates that haven't been yanked. Then:

        If not finding wheels, they are sorted by version only.
        If finding wheels, then the sort order is by version, then:
          1. existing installs
          2. wheels ordered via Wheel.support_index_min(self._supported_tags)
          3. source archives
        If prefer_binary was set, then all wheels are sorted above sources.

        Note: it was considered to embed this logic into the Link
              comparison operators, but then different sdist links
              with the same version, would have to be considered equal
        """
        valid_tags = self._supported_tags
        support_num = len(valid_tags)
        build_tag: BuildTag = ()
        binary_preference = 0
        link = candidate.link
        if link.is_wheel:
            # can raise InvalidWheelFilename
            wheel = Wheel(link.filename)
            try:
                pri = -(
                    wheel.find_most_preferred_tag(
                        valid_tags, self._wheel_tag_preferences
                    )
                )
            except ValueError:
                raise UnsupportedWheel(
                    f"{wheel.filename} is not a supported wheel for this platform. It "
                    "can't be sorted."
                )
            if self._prefer_binary:
                binary_preference = 1
            if wheel.build_tag is not None:
                match = re.match(r"^(\d+)(.*)$", wheel.build_tag)
                assert match is not None, "guaranteed by filename validation"
                build_tag_groups = match.groups()
                build_tag = (int(build_tag_groups[0]), build_tag_groups[1])
        else:  # sdist
            pri = -(support_num)
        has_allowed_hash = int(link.is_hash_allowed(self._hashes))
        yank_value = -1 * int(link.is_yanked)  # -1 for yanked.
        return (
            has_allowed_hash,
            yank_value,
            binary_preference,
            candidate.version,
            pri,
            build_tag,
        )

    def sort_best_candidate(
        self,
        candidates: List[InstallationCandidate],
    ) -> Optional[InstallationCandidate]:
        """
        Return the best candidate per the instance's sort order, or None if
        no candidate is acceptable.
        """
        if not candidates:
            return None
        best_candidate = max(candidates, key=self._sort_key)
        return best_candidate

    def compute_best_candidate(
        self,
        candidates: List[InstallationCandidate],
    ) -> BestCandidateResult:
        """
        Compute and return a `BestCandidateResult` instance.
        """
        applicable_candidates = self.get_applicable_candidates(candidates)

        best_candidate = self.sort_best_candidate(applicable_candidates)

        return BestCandidateResult(
            candidates,
            applicable_candidates=applicable_candidates,
            best_candidate=best_candidate,
        )


_FindCandidates = Callable[["PackageFinder", str], List[InstallationCandidate]]


def _canonicalize_arg(func: _FindCandidates) -> _FindCandidates:
    @functools.wraps(func)
    def wrapper(
        self: "PackageFinder", project_name: str
    ) -> List[InstallationCandidate]:
        return func(self, canonicalize_name(project_name))

    return wrapper


class PackageFinder:
    """This finds packages.

    This is meant to match easy_install's technique for looking for
    packages, by reading pages and looking for appropriate links.
    """

    def __init__(
        self,
        link_collector: LinkCollector,
        target_python: TargetPython,
        allow_yanked: bool,
        format_control: Optional[FormatControl] = None,
        candidate_prefs: Optional[CandidatePreferences] = None,
        ignore_requires_python: Optional[bool] = None,
        fetch_resolve_cache: Optional[FetchResolveCache] = None,
    ) -> None:
        """
        This constructor is primarily meant to be used by the create() class
        method and from tests.

        :param format_control: A FormatControl object, used to control
            the selection of source packages / binary packages when consulting
            the index and links.
        :param candidate_prefs: Options to use when creating a
            CandidateEvaluator object.
        """
        if candidate_prefs is None:
            candidate_prefs = CandidatePreferences()

        format_control = format_control or FormatControl(set(), set())

        self._allow_yanked = allow_yanked
        self._candidate_prefs = candidate_prefs
        self._ignore_requires_python = ignore_requires_python
        self._link_collector = link_collector
        self._target_python = target_python

        self.format_control = format_control

        # These are boring links that have already been logged somehow.
        self._logged_links: Set[Tuple[Link, LinkType, str]] = set()

        self._fetch_resolve_cache = fetch_resolve_cache

    # Don't include an allow_yanked default value to make sure each call
    # site considers whether yanked releases are allowed. This also causes
    # that decision to be made explicit in the calling code, which helps
    # people when reading the code.
    @classmethod
    def create(
        cls,
        link_collector: LinkCollector,
        selection_prefs: SelectionPreferences,
        target_python: Optional[TargetPython] = None,
        fetch_resolve_cache: Optional[FetchResolveCache] = None,
    ) -> "PackageFinder":
        """Create a PackageFinder.

        :param selection_prefs: The candidate selection preferences, as a
            SelectionPreferences object.
        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        """
        if target_python is None:
            target_python = TargetPython()

        candidate_prefs = CandidatePreferences(
            prefer_binary=selection_prefs.prefer_binary,
            allow_all_prereleases=selection_prefs.allow_all_prereleases,
        )

        return cls(
            candidate_prefs=candidate_prefs,
            link_collector=link_collector,
            target_python=target_python,
            allow_yanked=selection_prefs.allow_yanked,
            format_control=selection_prefs.format_control,
            ignore_requires_python=selection_prefs.ignore_requires_python,
            fetch_resolve_cache=fetch_resolve_cache,
        )

    @property
    def target_python(self) -> TargetPython:
        return self._target_python

    @property
    def search_scope(self) -> SearchScope:
        return self._link_collector.search_scope

    @search_scope.setter
    def search_scope(self, search_scope: SearchScope) -> None:
        self._link_collector.search_scope = search_scope

    @property
    def find_links(self) -> List[str]:
        return self._link_collector.find_links

    @property
    def index_urls(self) -> List[str]:
        return self.search_scope.index_urls

    @property
    def trusted_hosts(self) -> Iterable[str]:
        for host_port in self._link_collector.session.pip_trusted_origins:
            yield build_netloc(*host_port)

    @property
    def allow_all_prereleases(self) -> bool:
        return self._candidate_prefs.allow_all_prereleases

    def set_allow_all_prereleases(self) -> None:
        self._candidate_prefs.allow_all_prereleases = True

    @property
    def prefer_binary(self) -> bool:
        return self._candidate_prefs.prefer_binary

    def set_prefer_binary(self) -> None:
        self._candidate_prefs.prefer_binary = True

    def requires_python_skipped_reasons(self) -> List[str]:
        reasons = {
            detail
            for _, result, detail in self._logged_links
            if result == LinkType.requires_python_mismatch
        }
        return sorted(reasons)

    def make_link_evaluator(self, project_name: str) -> LinkEvaluator:
        canonical_name = canonicalize_name(project_name)
        formats = self.format_control.get_allowed_formats(canonical_name)

        return LinkEvaluator(
            project_name=project_name,
            canonical_name=canonical_name,
            formats=formats,
            target_python=self._target_python,
            allow_yanked=self._allow_yanked,
            ignore_requires_python=self._ignore_requires_python,
        )

    def _sort_links(self, links: Iterable[Link]) -> List[Link]:
        """
        Returns elements of links in order, non-egg links first, egg links
        second, while eliminating duplicates
        """
        eggs, no_eggs = [], []
        seen: Set[Link] = set()
        for link in links:
            if link not in seen:
                seen.add(link)
                if link.egg_fragment:
                    eggs.append(link)
                else:
                    no_eggs.append(link)
        return no_eggs + eggs

    def _log_skipped_link(self, link: Link, result: LinkType, detail: str) -> None:
        entry = (link, result, detail)
        if entry not in self._logged_links:
            # Put the link at the end so the reason is more visible and because
            # the link string is usually very long.
            logger.debug("Skipping link: %s: %s", detail, link)
            self._logged_links.add(entry)

    def get_install_candidate(
        self, link_evaluator: LinkEvaluator, link: Link
    ) -> Optional[InstallationCandidate]:
        """
        If the link is a candidate for install, convert it to an
        InstallationCandidate and return it. Otherwise, return None.
        """
        result, detail = link_evaluator.evaluate_link(link)
        if result != LinkType.candidate:
            self._log_skipped_link(link, result, detail)
            return None

        try:
            return InstallationCandidate(
                name=link_evaluator.project_name,
                link=link,
                version=detail,
            )
        except InvalidVersion:
            return None

    def evaluate_links(
        self, link_evaluator: LinkEvaluator, links: Iterable[Link]
    ) -> List[InstallationCandidate]:
        """
        Convert links that are candidates to InstallationCandidate objects.
        """
        candidates = []
        for link in self._sort_links(links):
            candidate = self.get_install_candidate(link_evaluator, link)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    _HTTP_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"

    @classmethod
    def _try_load_http_cache_headers(
        cls,
        etag_path: Path,
        date_path: Path,
        checksum_path: Path,
        project_url: Link,
        headers: Dict[str, str],
    ) -> Tuple[Optional[str], Optional[datetime.datetime], Optional[bytes]]:
        etag: Optional[str] = None
        try:
            etag = etag_path.read_text()
            etag = f'"{etag}"'
            logger.debug(
                "found cached etag for url %s at %s: %s",
                project_url,
                etag_path,
                etag,
            )
            headers["If-None-Match"] = etag
        except OSError as e:
            logger.debug("no etag found for url %s (%s)", project_url, str(e))

        date: Optional[datetime.datetime] = None
        try:
            date_bytes = date_path.read_bytes()
            date_int = int.from_bytes(date_bytes, byteorder="big", signed=False)
            date = datetime.datetime.fromtimestamp(date_int, tz=datetime.timezone.utc)
            logger.debug(
                "found cached date for url %s at %s: '%s'",
                project_url,
                date_path,
                date,
            )
            headers["If-Modified-Since"] = date.strftime(cls._HTTP_DATE_FORMAT)
        except OSError as e:
            logger.debug("no date found for url %s (%s)", project_url, str(e))

        checksum: Optional[bytes] = None
        try:
            checksum = checksum_path.read_bytes()
            logger.debug(
                "found checksum for url %s at %s: '%s'",
                project_url,
                checksum_path,
                binascii.b2a_base64(checksum, newline=False).decode("ascii"),
            )
        except OSError as e:
            logger.debug("no checksum found for url %s (%s)", project_url, str(e))

        return (etag, date, checksum)

    _quoted_value = re.compile(r'^"([^"]*)"$')

    @classmethod
    def _strip_quoted_value(cls, value: str) -> str:
        return cls._quoted_value.sub(r"\1", value)

    _now_local = datetime.datetime.now().astimezone()
    _local_tz = _now_local.tzinfo
    assert _local_tz is not None
    _local_tz_name = _local_tz.tzname(_now_local)

    @classmethod
    def _write_http_cache_info(
        cls,
        etag_path: Path,
        date_path: Path,
        checksum_path: Path,
        project_url: Link,
        index_response: IndexContent,
        prev_etag: Optional[str],
        prev_checksum: Optional[bytes],
    ) -> Tuple[Optional[str], Optional[datetime.datetime], bytes, bool]:
        hasher = sha256()
        hasher.update(index_response.content)
        new_checksum = hasher.digest()
        checksum_path.write_bytes(new_checksum)
        page_unmodified = new_checksum == prev_checksum

        new_etag: Optional[str] = index_response.etag
        if new_etag is None:
            logger.debug("no etag returned from fetch for url %s", project_url.url)
            try:
                etag_path.unlink()
            except OSError:
                pass
        else:
            new_etag = cls._strip_quoted_value(new_etag)
            if new_etag != prev_etag:
                logger.debug(
                    "etag for url %s updated from %s -> %s",
                    project_url.url,
                    prev_etag,
                    new_etag,
                )
                etag_path.write_text(new_etag)
            else:
                logger.debug(
                    "etag was unmodified for url %s (%s)", project_url.url, prev_etag
                )
                assert page_unmodified

        new_date: Optional[datetime.datetime] = None
        date_str: Optional[str] = index_response.date
        if date_str is None:
            logger.debug(
                "no date header was provided in response for url %s", project_url
            )
        else:
            date_str = date_str.strip()
            new_time = time.strptime(date_str, cls._HTTP_DATE_FORMAT)
            new_date = datetime.datetime.strptime(date_str, cls._HTTP_DATE_FORMAT)
            # strptime() doesn't set the timezone according to the parsed %Z arg, which
            # may be any of "UTC", "GMT", or any element of `time.tzname`.
            if new_time.tm_zone in ["UTC", "GMT"]:
                logger.debug(
                    "a UTC timezone was found in response for url %s", project_url
                )
                new_date = new_date.replace(tzinfo=datetime.timezone.utc)
            else:
                assert new_time.tm_zone in time.tzname, new_time
                logger.debug(
                    "a local timezone %s was found in response for url %s",
                    new_time.tm_zone,
                    project_url,
                )
                if new_time.tm_zone == cls._local_tz_name:
                    new_date = new_date.replace(tzinfo=cls._local_tz)
                else:
                    logger.debug(
                        "a local timezone %s had to be discarded in response %s",
                        new_time.tm_zone,
                        project_url,
                    )
                    new_date = None

            if new_date is not None:
                timestamp = new_date.timestamp()
                # The timestamp will only have second resolution according to the parse
                # format string _HTTP_DATE_FORMAT.
                assert not (timestamp % 1), (new_date, timestamp)
                epoch = int(timestamp)
                assert epoch >= 0, (new_date, timestamp, epoch)
                date_bytes = epoch.to_bytes(length=4, byteorder="big", signed=False)
                date_path.write_bytes(date_bytes)

                logger.debug('date "%s" written for url %s', new_date, project_url)
        if new_date is None:
            try:
                date_path.unlink()
            except OSError:
                pass

        return (new_etag, new_date, new_checksum, page_unmodified)

    @staticmethod
    def _try_load_parsed_links_cache(parsed_links_path: Path) -> Optional[List[Link]]:
        page_links: Optional[List[Link]] = None
        try:
            with bz2.open(parsed_links_path, mode="rt", encoding="utf-8") as f:
                logger.debug("reading page links from cache %s", parsed_links_path)
                cached_links = json.load(f)
                page_links = []
                for cache_info in cached_links:
                    link = Link.from_cache_args(
                        PersistentLinkCacheArgs.from_json(cache_info)
                    )
                    assert link is not None
                    page_links.append(link)
        except (OSError, json.decoder.JSONDecodeError, KeyError) as e:
            logger.debug(
                "could not read page links from cache file %s %s(%s)",
                parsed_links_path,
                e.__class__.__name__,
                str(e),
            )
        return page_links

    @staticmethod
    def _write_parsed_links_cache(
        parsed_links_path: Path, links: Iterable[Link]
    ) -> List[Link]:
        cacheable_links: List[Dict[str, Any]] = []
        page_links: List[Link] = []
        for link in links:
            cache_info = link.cache_args()
            assert cache_info is not None
            cacheable_links.append(cache_info.to_json())
            page_links.append(link)

        logger.debug("writing page links to %s", parsed_links_path)
        with bz2.open(parsed_links_path, mode="wt", encoding="utf-8") as f:
            json.dump(cacheable_links, f)

        return page_links

    @staticmethod
    def _try_load_installation_candidate_cache(
        cached_candidates_path: Path,
    ) -> Optional[List[InstallationCandidate]]:
        try:
            with bz2.open(cached_candidates_path, mode="rt", encoding="utf-8") as f:
                serialized_candidates = json.load(f)
            logger.debug("read serialized candidates from %s", cached_candidates_path)
            package_links: List[InstallationCandidate] = []
            for cand in serialized_candidates:
                link_cache_args = PersistentLinkCacheArgs.from_json(cand["link"])
                link = Link.from_cache_args(link_cache_args)
                package_links.append(
                    InstallationCandidate(cand["name"], cand["version"], link)
                )
            return package_links
        except (OSError, json.decoder.JSONDecodeError, KeyError) as e:
            logger.debug(
                "could not read cached candidates at %s %s(%s)",
                cached_candidates_path,
                e.__class__.__name__,
                str(e),
            )
        return None

    @staticmethod
    def _write_installation_candidate_cache(
        cached_candidates_path: Path,
        candidates: Iterable[InstallationCandidate],
    ) -> List[InstallationCandidate]:
        candidates = list(candidates)
        serialized_candidates = [
            {
                "name": candidate.name,
                "version": str(candidate.version),
                "link": candidate.link.cache_args().to_json(),
            }
            for candidate in candidates
        ]
        with bz2.open(cached_candidates_path, mode="wt", encoding="utf-8") as f:
            logger.debug("writing serialized candidates to %s", cached_candidates_path)
            json.dump(serialized_candidates, f)
        return candidates

    def _process_project_url_uncached(
        self, project_url: Link, link_evaluator: LinkEvaluator
    ) -> List[InstallationCandidate]:
        logger.debug(
            "Fetching project page and analyzing links: %s",
            project_url,
        )

        index_response = self._link_collector.fetch_response(project_url)
        if index_response is None:
            return []

        page_links = parse_links(index_response)

        with indent_log():
            package_links = self.evaluate_links(link_evaluator, links=page_links)
        return package_links

    def process_project_url(
        self, project_url: Link, link_evaluator: LinkEvaluator
    ) -> List[InstallationCandidate]:
        if self._fetch_resolve_cache is None:
            return self._process_project_url_uncached(project_url, link_evaluator)

        cached_path = self._fetch_resolve_cache.cache_path(project_url)
        os.makedirs(str(cached_path), exist_ok=True)

        etag_path = cached_path / "etag"
        date_path = cached_path / "modified-since-date"
        checksum_path = cached_path / "checksum"
        parsed_links_path = cached_path / "parsed-links"
        cached_candidates_path = self._fetch_resolve_cache.hashed_entry_path(
            project_url, link_evaluator
        )

        headers: Dict[str, str] = {}
        # NB: mutates headers!
        prev_etag, _prev_date, prev_checksum = self._try_load_http_cache_headers(
            etag_path, date_path, checksum_path, project_url, headers
        )

        logger.debug(
            "Fetching project page and analyzing links: %s",
            project_url,
        )

        # A 304 Not Modified is implicitly converted into a reused cached response from
        # the Cache-Control library, so we won't explicitly check for a 304.
        index_response = self._link_collector.fetch_response(
            project_url,
            headers=headers,
        )
        if index_response is None:
            return []

        (
            _new_etag,
            _new_date,
            _new_checksum,
            page_unmodified,
        ) = self._write_http_cache_info(
            etag_path,
            date_path,
            checksum_path,
            project_url,
            index_response,
            prev_etag=prev_etag,
            prev_checksum=prev_checksum,
        )

        page_links: Optional[List[Link]] = None
        # Only try our persistent link parsing and evaluation caches if we know the page
        # was unmodified via checksum.
        if page_unmodified:
            cached_candidates = self._try_load_installation_candidate_cache(
                cached_candidates_path
            )
            if cached_candidates is not None:
                return cached_candidates

            page_links = self._try_load_parsed_links_cache(parsed_links_path)
        else:
            try:
                parsed_links_path.unlink()
            except OSError:
                pass
            self._fetch_resolve_cache.clear_hashed_entries(project_url, LinkEvaluator)

        if page_links is None:
            logger.debug(
                "extracting new parsed links from index response %s", index_response
            )
            page_links = self._write_parsed_links_cache(
                parsed_links_path,
                parse_links(index_response),
            )

        with indent_log():
            package_links = self._write_installation_candidate_cache(
                cached_candidates_path,
                self.evaluate_links(
                    link_evaluator,
                    links=page_links,
                ),
            )

        return package_links

    @_canonicalize_arg
    @functools.lru_cache(maxsize=None)
    def find_all_candidates(self, project_name: str) -> List[InstallationCandidate]:
        """Find all available InstallationCandidate for project_name

        This checks index_urls and find_links.
        All versions found are returned as an InstallationCandidate list.

        See LinkEvaluator.evaluate_link() for details on which files
        are accepted.
        """
        link_evaluator = self.make_link_evaluator(project_name)

        collected_sources = self._link_collector.collect_sources(
            project_name=project_name,
            candidates_from_page=functools.partial(
                self.process_project_url,
                link_evaluator=link_evaluator,
            ),
        )

        page_candidates_it = itertools.chain.from_iterable(
            source.page_candidates()
            for sources in collected_sources
            for source in sources
            if source is not None
        )
        page_candidates = list(page_candidates_it)

        file_links_it = itertools.chain.from_iterable(
            source.file_links()
            for sources in collected_sources
            for source in sources
            if source is not None
        )
        file_candidates = self.evaluate_links(
            link_evaluator,
            sorted(file_links_it, reverse=True),
        )

        if logger.isEnabledFor(logging.DEBUG) and file_candidates:
            paths = []
            for candidate in file_candidates:
                assert candidate.link.url  # we need to have a URL
                try:
                    paths.append(candidate.link.file_path)
                except Exception:
                    paths.append(candidate.link.url)  # it's not a local file

            logger.debug("Local files found: %s", ", ".join(paths))

        # This is an intentional priority ordering
        return file_candidates + page_candidates

    def make_candidate_evaluator(
        self,
        project_name: str,
        specifier: Optional[specifiers.BaseSpecifier] = None,
        hashes: Optional[Hashes] = None,
    ) -> CandidateEvaluator:
        """Create a CandidateEvaluator object to use."""
        candidate_prefs = self._candidate_prefs
        return CandidateEvaluator.create(
            project_name=project_name,
            target_python=self._target_python,
            prefer_binary=candidate_prefs.prefer_binary,
            allow_all_prereleases=candidate_prefs.allow_all_prereleases,
            specifier=specifier,
            hashes=hashes,
        )

    @functools.lru_cache(maxsize=None)
    def find_best_candidate(
        self,
        project_name: str,
        specifier: Optional[specifiers.BaseSpecifier] = None,
        hashes: Optional[Hashes] = None,
    ) -> BestCandidateResult:
        """Find matches for the given project and specifier.

        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.

        :return: A `BestCandidateResult` instance.
        """
        candidates = self.find_all_candidates(project_name)
        candidate_evaluator = self.make_candidate_evaluator(
            project_name=project_name,
            specifier=specifier,
            hashes=hashes,
        )
        return candidate_evaluator.compute_best_candidate(candidates)

    def find_requirement(
        self, req: InstallRequirement, upgrade: bool
    ) -> Optional[InstallationCandidate]:
        """Try to find a Link matching req

        Expects req, an InstallRequirement and upgrade, a boolean
        Returns a InstallationCandidate if found,
        Raises DistributionNotFound or BestVersionAlreadyInstalled otherwise
        """
        hashes = req.hashes(trust_internet=False)
        best_candidate_result = self.find_best_candidate(
            req.name,
            specifier=req.specifier,
            hashes=hashes,
        )
        best_candidate = best_candidate_result.best_candidate

        installed_version: Optional[_BaseVersion] = None
        if req.satisfied_by is not None:
            installed_version = req.satisfied_by.version

        def _format_versions(cand_iter: Iterable[InstallationCandidate]) -> str:
            # This repeated parse_version and str() conversion is needed to
            # handle different vendoring sources from pip and pkg_resources.
            # If we stop using the pkg_resources provided specifier and start
            # using our own, we can drop the cast to str().
            return (
                ", ".join(
                    sorted(
                        {str(c.version) for c in cand_iter},
                        key=parse_version,
                    )
                )
                or "none"
            )

        if installed_version is None and best_candidate is None:
            logger.critical(
                "Could not find a version that satisfies the requirement %s "
                "(from versions: %s)",
                req,
                _format_versions(best_candidate_result.iter_all()),
            )

            raise DistributionNotFound(f"No matching distribution found for {req}")

        def _should_install_candidate(
            candidate: Optional[InstallationCandidate],
        ) -> "TypeGuard[InstallationCandidate]":
            if installed_version is None:
                return True
            if best_candidate is None:
                return False
            return best_candidate.version > installed_version

        if not upgrade and installed_version is not None:
            if _should_install_candidate(best_candidate):
                logger.debug(
                    "Existing installed version (%s) satisfies requirement "
                    "(most up-to-date version is %s)",
                    installed_version,
                    best_candidate.version,
                )
            else:
                logger.debug(
                    "Existing installed version (%s) is most up-to-date and "
                    "satisfies requirement",
                    installed_version,
                )
            return None

        if _should_install_candidate(best_candidate):
            logger.debug(
                "Using version %s (newest of versions: %s)",
                best_candidate.version,
                _format_versions(best_candidate_result.iter_applicable()),
            )
            return best_candidate

        # We have an existing version, and its the best version
        logger.debug(
            "Installed version (%s) is most up-to-date (past versions: %s)",
            installed_version,
            _format_versions(best_candidate_result.iter_applicable()),
        )
        raise BestVersionAlreadyInstalled


def _find_name_version_sep(fragment: str, canonical_name: str) -> int:
    """Find the separator's index based on the package's canonical name.

    :param fragment: A <package>+<version> filename "fragment" (stem) or
        egg fragment.
    :param canonical_name: The package's canonical name.

    This function is needed since the canonicalized name does not necessarily
    have the same length as the egg info's name part. An example::

    >>> fragment = 'foo__bar-1.0'
    >>> canonical_name = 'foo-bar'
    >>> _find_name_version_sep(fragment, canonical_name)
    8
    """
    # Project name and version must be separated by one single dash. Find all
    # occurrences of dashes; if the string in front of it matches the canonical
    # name, this is the one separating the name and version parts.
    for i, c in enumerate(fragment):
        if c != "-":
            continue
        if canonicalize_name(fragment[:i]) == canonical_name:
            return i
    raise ValueError(f"{fragment} does not match {canonical_name}")


def _extract_version_from_fragment(fragment: str, canonical_name: str) -> Optional[str]:
    """Parse the version string from a <package>+<version> filename
    "fragment" (stem) or egg fragment.

    :param fragment: The string to parse. E.g. foo-2.1
    :param canonical_name: The canonicalized name of the package this
        belongs to.
    """
    try:
        version_start = _find_name_version_sep(fragment, canonical_name) + 1
    except ValueError:
        return None
    version = fragment[version_start:]
    if not version:
        return None
    return version
