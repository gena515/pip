import os
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest
from pip._vendor.resolvelib import BaseReporter, Resolver

from pip._internal.resolution.resolvelib.base import Candidate, Constraint, Requirement
from pip._internal.resolution.resolvelib.factory import Factory
from pip._internal.resolution.resolvelib.provider import PipProvider
from tests.lib import TestData

# NOTE: All tests are prefixed `test_rlr` (for "test resolvelib resolver").
#       This helps select just these tests using pytest's `-k` option, and
#       keeps test names shorter.

# Basic tests:
#   Create a requirement from a project name - "pip"
#   Create a requirement from a name + version constraint - "pip >= 20.0"
#   Create a requirement from a wheel filename
#   Create a requirement from a sdist filename
#   Create a requirement from a local directory (which has no obvious name!)
#   Editables


@pytest.fixture
def test_cases(data: TestData) -> Iterator[List[Tuple[str, str, int]]]:
    def _data_file(name: str) -> Path:
        return data.packages.joinpath(name)

    def data_file(name: str) -> str:
        return os.fspath(_data_file(name))

    def data_url(name: str) -> str:
        return _data_file(name).as_uri()

    test_cases = [
        # requirement, name, matches
        # Version specifiers
        ("simple", "simple", 3),
        ("simple>1.0", "simple", 2),
        # ("simple[extra]==1.0", "simple[extra]", 1),
        # Wheels
        (data_file("simplewheel-1.0-py2.py3-none-any.whl"), "simplewheel", 1),
        (data_url("simplewheel-1.0-py2.py3-none-any.whl"), "simplewheel", 1),
        # Direct URLs
        # TODO: The following test fails
        # ("foo @ " + data_url("simple-1.0.tar.gz"), "foo", 1),
        # SDists
        # TODO: sdists should have a name
        (data_file("simple-1.0.tar.gz"), "simple", 1),
        (data_url("simple-1.0.tar.gz"), "simple", 1),
        # TODO: directory, editables
    ]

    yield test_cases


def test_new_resolver_requirement_has_name(
    test_cases: List[Tuple[str, str, int]], factory: Factory
) -> None:
    """All requirements should have a name"""
    for spec, name, _ in test_cases:
        req = factory.make_requirement_from_spec(spec, comes_from=None)
        assert req is not None
        assert req.name == name


def test_new_resolver_correct_number_of_matches(
    test_cases: List[Tuple[str, str, int]], factory: Factory
) -> None:
    """Requirements should return the correct number of candidates"""
    for spec, _, match_count in test_cases:
        req = factory.make_requirement_from_spec(spec, comes_from=None)
        assert req is not None
        matches = factory.find_candidates(
            req.name,
            {req.name: [req]},
            {},
            Constraint.empty(),
            prefers_installed=False,
            version_selection="max",
        )
        assert sum(1 for _ in matches) == match_count


def test_new_resolver_candidates_match_requirement(
    test_cases: List[Tuple[str, str, int]], factory: Factory
) -> None:
    """Candidates returned from find_candidates should satisfy the requirement"""
    for spec, _, _ in test_cases:
        req = factory.make_requirement_from_spec(spec, comes_from=None)
        assert req is not None
        candidates = factory.find_candidates(
            req.name,
            {req.name: [req]},
            {},
            Constraint.empty(),
            prefers_installed=False,
            version_selection="max",
        )
        previous_candidate = None
        for c in candidates:
            assert isinstance(c, Candidate)
            assert req.is_satisfied_by(c)

            if previous_candidate is not None:
                assert c.version < previous_candidate.version

            previous_candidate = c


def test_new_resolver_candidates_version_selection_min(
    test_cases: List[Tuple[str, str, int]], factory: Factory
) -> None:
    """Candidates returned from find_candidates should satisfy the requirement"""
    for spec, _, _ in test_cases:
        req = factory.make_requirement_from_spec(spec, comes_from=None)
        assert req is not None
        candidates = factory.find_candidates(
            req.name,
            {req.name: [req]},
            {},
            Constraint.empty(),
            prefers_installed=False,
            version_selection="min",
        )
        previous_candidate = None
        for c in candidates:
            assert isinstance(c, Candidate)
            assert req.is_satisfied_by(c)

            if previous_candidate is not None:
                assert c.version > previous_candidate.version

            previous_candidate = c



def test_new_resolver_full_resolve(factory: Factory, provider: PipProvider) -> None:
    """A very basic full resolve"""
    req = factory.make_requirement_from_spec("simplewheel", comes_from=None)
    assert req is not None
    r: Resolver[Requirement, Candidate, str] = Resolver(provider, BaseReporter())
    result = r.resolve([req])
    assert set(result.mapping.keys()) == {"simplewheel"}
