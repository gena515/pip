"""Routines related to PyPI, indexes"""

import enum
import functools
import itertools
import logging
import pathlib
import re
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

from pip._vendor.packaging import specifiers
from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import InvalidVersion, _BaseVersion
from pip._vendor.packaging.version import parse as parse_version

from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
    InvalidAlternativeLocationsUrl,
    InvalidWheelFilename,
    InvalidTracksUrl,
    UnsafeMultipleRemoteRepositories,
    UnsupportedWheel,
)
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import FormatControl
from pip._internal.models.link import Link
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


class LinkEvaluator:
    """
    Responsible for evaluating links for a particular project.
    """

    _py_version_re = re.compile(r"-py([123]\.?[0-9]?)$")

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

        check_multiple_remote_repositories(
            candidates=filtered_applicable_candidates,
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

    def process_project_url(
        self, project_url: Link, link_evaluator: LinkEvaluator
    ) -> List[InstallationCandidate]:
        logger.debug(
            "Fetching project page and analyzing links: %s",
            project_url,
        )
        index_response = self._link_collector.fetch_response(project_url)
        if index_response is None:
            return []

        page_links = list(parse_links(index_response))

        with indent_log():
            package_links = self.evaluate_links(
                link_evaluator,
                links=page_links,
            )

        return package_links

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


def check_multiple_remote_repositories(
    candidates: List[InstallationCandidate], project_name: str
) -> None:
    """
    Check whether two or more different namespaces can be flattened into one.

    This check will raise an error when packages from different sources will be merged,
    without clear configuration to indicate the namespace merge is allowed.
    See `PEP 708`_.

    This approach allows safely merging separate namespaces that are actually one
    logical namespace, while enforcing a more secure default.

    Returns None if checks pass, otherwise will raise an error with details of the
    failed checks.
    """

    # NOTE: The checks in this function must occur after:
    # Specification: Filter out any files that do not match known hashes from a
    # lockfile or requirements file.

    # NOTE: The checks in this function must occur before:
    # Specification: Filter out any files that do not match the current platform,
    # Python version, etc. We check for the metadata in this PEP before filtering out
    # based on platform, Python version, etc., because we don’t want errors that only
    # show up on certain platforms, Python versions, etc.

    # NOTE: Implemented in 'filter_unallowed_hashes':
    # Specification: Users who are using lock files or requirements files that include
    # specific hashes of artifacts that are “valid” are assumed to be protected by
    # nature of those hashes, since the rest of these recommendations would apply
    # during hash generation. Thus, we filter out unknown hashes up front.

    # NOTE: Implemented in 'collector.py' 'parse_links':
    # Specification: When using alternate locations, clients MUST implicitly assume
    # that the url the response was fetched from was included in the list.

    # NOTE: Implemented in 'collector.py' 'parse_links':
    # Specification: Order of the elements within the array does not have any
    # particular meaning.

    # NOTE: Implemented by this function, by not checking all repo tracks metadata is
    # the exact same.
    # Specification: Mixed use repositories where some names track a repository and
    # some names do not are explicitly allowed.

    # TODO: This requirement doesn't look like something that pip can do anything about.
    # Specification: All [Tracks metadata] URLs MUST represent the same “project” as
    # the project in the extending repository. This does not mean that they need to
    # serve the same files. It is valid for them to include binaries built on
    # different platforms, copies with local patches being applied, etc. This is
    # purposefully left vague as it’s ultimately up to the expectations that the
    # users have of the repository and its operators what exactly constitutes
    # the “same” project.

    # TODO: This requirement doesn't look like something that pip can do anything about.
    # Specification: It [Tracks metadata] is NOT required that every name in a
    # repository tracks the same repository, or that they all track a repository at all.

    # TODO: This requirement doesn't look like something that pip can do anything about.
    # Specification: It [repository Tracks metadata] MUST be under the control of the
    # repository operators themselves, not any individual publisher using that
    # repository. “Repository Operator” can also include anyone who managed the overall
    # namespace for a particular repository, which may be the case in situations like
    # hosted repository services where one entity operates the software but another
    # owns/manages the entire namespace of that repository.

    # TODO: Consider making requests to repositories revealed via Alternate Locations
    #    or Tracks metadata, to assess the metadata of those additional repositories.
    # Specification: When an installer encounters a project that is using the
    # alternate locations metadata it SHOULD consider that all repositories named are
    # extending the same namespace across multiple repositories.
    # Implementation: It is not possible to enforce this requirement without access
    # to the metadata for all of the remote repositories, as the Tracks metadata
    # might point at a repository that does not exist, or at a repository that is
    # Tracking another repository.

    # TODO: Consider whether pip already has, or should add, ways to indicate exactly
    #   which individual projects to get from exactly which repositories.
    # Specification: If the user has explicitly told the installer that it wants to
    # fetch a project from a certain set of repositories, then there is no reason
    # to question that and we assume that they’ve made sure it is safe to merge
    # those namespaces. If the end user has explicitly told the installer to fetch
    # the project from specific repositories, filter out all other repositories.

    # When no candidates are provided, then no checks are relevant, so just return.
    if candidates is None or len(candidates) == 0:
        logger.debug("No candidates given to multiple remote repository checks")
        return

    # Calculate the canonical name for later comparisons.
    canonical_name = canonicalize_name(project_name)

    # Specification: Look to see if the discovered files span multiple repositories;
    # if they do then determine if either “Tracks” or “Alternate Locations” metadata
    # allows safely merging together ALL the repositories where files were discovered.
    remote_candidates = []
    remote_repositories = set()

    for candidate in candidates:
        candidate_name = candidate.name
        link = candidate.link
        comes_from = link.comes_from

        candidate_canonical_name = canonicalize_name(candidate_name)

        # Specification: Repositories that exist on the local filesystem SHOULD always
        # be implicitly allowed to be merged to any remote repository.
        if link.is_local_only:
            # Ignore any local candidates in later comparisons.
            logger.debug(
                "Ignoring local candidate %s in multiple remote repository checks",
                candidate,
            )
            continue

        try:
            page_url = comes_from.url.lstrip()
        except AttributeError:
            page_url = comes_from.lstrip()

        item = {
            "candidate": candidate,
            "canonical_name": candidate_canonical_name,
        }
        remote_candidates.append(item)

        remote_repositories.add(page_url)
        remote_repositories.update(link.project_track_urls)
        remote_repositories.update(link.repo_alt_urls)

    # If there are no remote candidates, then allow merging repositories.
    if len(remote_candidates) == 0:
        logger.debug("No remote candidates for multiple remote repository checks")
        return

    # Specification: If the project in question only comes from a single repository,
    # then there is no chance of dependency confusion, so there’s no reason to do
    # anything but allow.
    if len(remote_repositories) < 2:
        logger.debug(
            "No chance for dependency confusion when there is only "
            "one remote candidate for multiple remote repository checks"
        )
        return

    # TODO
    if logger.isEnabledFor(logging.INFO):
        logger.info("Remote candidates for multiple remote repository checks:")
        for candidate in remote_candidates:
            logger.info(candidate)

    # TODO: This checks the list of Alternate Locations for the candidates that were
    #   retrieved. It does not request the metadata from any additional repositories
    #   revealed via the lists of Alternate Locations urls or Tracks urls.
    # Specification: In order for this metadata to be trusted, there MUST be agreement
    # between all locations where that project is found as to what the alternate
    # locations are.
    all_repo_alt_urls = list(map_alt_urls.keys())
    if len(all_repo_alt_urls) > 1:
        match_alt_locs = set(all_repo_alt_urls[0])
        invalid_locations = set()

        for item in all_repo_alt_urls[1:]:
            match_alt_locs.intersection_update(item)
            invalid_locations.update(match_alt_locs.symmetric_difference(item))

        logger.debug(match_alt_locs)
        logger.debug(invalid_locations)

        if len(invalid_locations) > 0:
            raise InvalidAlternativeLocationsUrl(
                package=project_name,
                remote_repositories=remote_repositories,
                invalid_locations=invalid_locations,
            )

    for remote_candidate in remote_candidates:
        project_track_urls = remote_candidate.get("candidate").link.project_track_urls
        project_track_urls = remote_candidate.get("candidate").link.project_track_urls
        page_url = remote_candidate.get("url")
        # url_parts = pathlib.Path(urllib.parse.urlsplit(url).path).parts
        for project_track_url in project_track_urls:
            parts = pathlib.Path(urllib.parse.urlsplit(project_track_url).path).parts

            # Specification: It [Tracks metadata] MUST point to the actual URLs
            # for that project, not the base URL for the extended repositories.
            # Specification: It [Tracks metadata] MUST point to a project with
            # the exact same normalized name.
            # Implementation: The normalised project name must be present in one
            # of the Tracks url path parts.
            # TODO: This assumption about the structure of the url may not hold true
            #       for all remote repositories.
            if not parts or not any(
                [canonical_name == canonicalize_name(p) for p in parts]
            ):
                raise InvalidTracksUrl(
                    package=project_name,
                    remote_repositories={page_url},
                    invalid_tracks={project_track_url},
                )

            # Specification: It [Tracks metadata] MUST point to the repositories
            # that “own” the namespaces, not another repository that is also
            # tracking that namespace.
            # Implementation: An 'owner' repository is one that does not Track the
            # same namespace.
            # TODO: Without requesting all repositories revealed by metadata, this
            #       check might pass with incomplete metadata,
            #       when it would fail with complete metadata.
            if project_track_url in map_track_urls:
                raise InvalidTracksUrl(
                    package=project_name,
                    remote_repositories={page_url},
                    invalid_tracks={project_track_url},
                )

    # TODO
    # Specification: Otherwise [if metadata allows] we merge the namespaces,
    # and continue on.

    # Specification: If nothing tells us merging the namespaces is safe, we refuse to
    # implicitly assume it is, and generate an error instead.
    # Specification: If that metadata does NOT allow [merging namespaces], then
    # generate an error.
    error = UnsafeMultipleRemoteRepositories(
        package=project_name, remote_repositories=remote_repositories
    )
    raise error
