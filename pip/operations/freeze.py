from __future__ import absolute_import

import logging
import re

import pip
from pip.compat import stdlib_pkgs
from pip.req import InstallRequirement
from pip.utils import get_installed_distributions, get_recursive_dependencies
from pip._vendor import pkg_resources


logger = logging.getLogger(__name__)

# packages to exclude from freeze output
freeze_excludes = stdlib_pkgs + ['setuptools', 'pip', 'distribute']


def freeze(
        only_dists=None,
        recursive=False,
        requirement=None,
        find_links=None, local_only=None, user_only=None, skip_regex=None,
        find_tags=False,
        default_vcs=None,
        isolated=False):
    find_links = find_links or []
    skip_match = None

    if skip_regex:
        skip_match = re.compile(skip_regex)

    dependency_links = []

    for dist in pkg_resources.working_set:
        if dist.has_metadata('dependency_links.txt'):
            dependency_links.extend(
                dist.get_metadata_lines('dependency_links.txt')
            )
    for link in find_links:
        if '#egg=' in link:
            dependency_links.append(link)
    for link in find_links:
        yield '-f %s' % link
    installations = {}

    if only_dists:
        # Freeze only a subset of installed packages.
        if recursive:  # Freeze dependencies, recursively.
            only_dists.extend(get_recursive_dependencies(only_dists))
        only_dists = [pkg_resources.safe_extra(name)
                      for name in only_dists]

    for dist in get_installed_distributions(local_only=local_only,
                                            skip=freeze_excludes,
                                            user_only=user_only):
        safe_name = pkg_resources.safe_extra(dist.project_name)
        if not only_dists or safe_name in only_dists:
            req = pip.FrozenRequirement.from_dist(
                dist,
                dependency_links,
                find_tags=find_tags,
            )
            installations[req.name] = req

    if requirement:
        with open(requirement) as req_file:
            for line in req_file:
                if (not line.strip()
                        or line.strip().startswith('#')
                        or (skip_match and skip_match.search(line))
                        or line.startswith((
                            '-r', '--requirement',
                            '-Z', '--always-unzip',
                            '-f', '--find-links',
                            '-i', '--index-url',
                            '--extra-index-url'))):
                    yield line.rstrip()
                    continue

                if line.startswith('-e') or line.startswith('--editable'):
                    if line.startswith('-e'):
                        line = line[2:].strip()
                    else:
                        line = line[len('--editable'):].strip().lstrip('=')
                    line_req = InstallRequirement.from_editable(
                        line,
                        default_vcs=default_vcs,
                        isolated=isolated,
                    )
                else:
                    line_req = InstallRequirement.from_line(
                        line,
                        isolated=isolated,
                    )

                if not line_req.name:
                    logger.info(
                        "Skipping line because it's not clear what it "
                        "would install: %s",
                        line.strip(),
                    )
                    logger.info(
                        "  (add #egg=PackageName to the URL to avoid"
                        " this warning)"
                    )
                elif line_req.name not in installations:
                    logger.warning(
                        "Requirement file contains %s, but that package is"
                        " not installed",
                        line.strip(),
                    )
                else:
                    yield str(installations[line_req.name]).rstrip()
                    del installations[line_req.name]

        yield(
            '## The following requirements were added by '
            'pip freeze:'
        )
    for installation in sorted(
            installations.values(), key=lambda x: x.name.lower()):
        yield str(installation).rstrip()
