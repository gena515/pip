import functools

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.resolvelib import BaseReporter
from pip._vendor.resolvelib import Resolver as RLResolver

from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver
from pip._internal.resolution.resolvelib.provider import PipProvider
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Dict, List, Optional, Tuple

    from pip._vendor.resolvelib.resolvers import Result

    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.operations.prepare import RequirementPreparer
    from pip._internal.req.req_install import InstallRequirement
    from pip._internal.resolution.base import InstallRequirementProvider


class Resolver(BaseResolver):
    def __init__(
        self,
        preparer,  # type: RequirementPreparer
        finder,  # type: PackageFinder
        make_install_req,  # type: InstallRequirementProvider
        use_user_site,  # type: bool
        ignore_dependencies,  # type: bool
        ignore_installed,  # type: bool
        ignore_requires_python,  # type: bool
        force_reinstall,  # type: bool
        upgrade_strategy,  # type: str
        py_version_info=None,  # type: Optional[Tuple[int, ...]]
    ):
        super(Resolver, self).__init__()
        self.finder = finder
        self.preparer = preparer
        self.ignore_dependencies = ignore_dependencies
        self.make_install_req = make_install_req
        self._result = None  # type: Optional[Result]

    def resolve(self, root_reqs, check_supported_wheels):
        # type: (List[InstallRequirement], bool) -> RequirementSet
        provider = PipProvider(
            finder=self.finder,
            preparer=self.preparer,
            ignore_dependencies=self.ignore_dependencies,
            make_install_req=self.make_install_req,
        )
        reporter = BaseReporter()
        resolver = RLResolver(provider, reporter)

        requirements = [provider.make_requirement(r) for r in root_reqs]
        self._result = resolver.resolve(requirements)

        req_set = RequirementSet(check_supported_wheels=check_supported_wheels)
        for candidate in self._result.mapping.values():
            ireq = provider.get_install_requirement(candidate)
            if ireq is not None:
                req_set.add_named_requirement(ireq)

        return req_set

    def get_installation_order(self, req_set):
        # type: (RequirementSet) -> List[InstallRequirement]
        """Create a list that orders given requirements for installation.

        The returned list should contain all requirements in ``req_set``,
        so the caller can loop through it and have a requirement installed
        before the requiring thing.

        The current implementation walks the resolved dependency graph, and
        make sure every node has a greater "weight" than all its parents.
        """
        assert self._result is not None, "must call resolve() first"
        weights = {}  # type: Dict[Optional[str], int]

        graph = self._result.graph
        key_count = len(self._result.mapping) + 1  # Packages plus sentinal.
        while len(weights) < key_count:
            progressed = False
            for key in graph:
                if key in weights:
                    continue
                parents = list(graph.iter_parents(key))
                if not all(p in weights for p in parents):
                    continue
                if parents:
                    weight = max(weights[p] for p in parents) + 1
                else:
                    weight = 0
                weights[key] = weight
                progressed = True

            # FIXME: This check will fail if there are unbreakable cycles.
            # Implement something to forcifully break them up to continue.
            assert progressed, "Order calculation stuck in dependency loop."

        sorted_items = sorted(
            req_set.requirements.items(),
            key=functools.partial(_req_set_item_sorter, weights=weights),
            reverse=True,
        )
        return [ireq for _, ireq in sorted_items]


def _req_set_item_sorter(
    item,     # type: Tuple[str, InstallRequirement]
    weights,  # type: Dict[Optional[str], int]
):
    # type: (...) -> Tuple[int, str]
    """Key function used to sort install requirements for installation.

    Based on the "weight" mapping calculated in ``get_installation_order()``.
    The canonical package name is returned as the second member as a tie-
    breaker to ensure the result is predictable, which is useful in tests.
    """
    name = canonicalize_name(item[0])
    return weights[name], name
