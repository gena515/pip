from __future__ import absolute_import

import logging
from collections import OrderedDict, defaultdict

from pip._internal.exceptions import InstallationError
from pip._internal.utils.logging import indent_log
from pip._internal.wheel import Wheel

logger = logging.getLogger(__name__)


class RequirementSet(object):

    def __init__(self,
                 require_hashes=False, target_dir=None, use_user_site=False,
                 pycompile=True):
        """Create a RequirementSet.

        :param wheel_cache: The pip wheel cache, for passing to
            InstallRequirement.
        """

        self.requirements = OrderedDict()
        self.require_hashes = require_hashes

        # Mapping of alias: real_name
        self.requirement_aliases = {}
        self.unnamed_requirements = []
        self.successfully_downloaded = []
        self.reqs_to_cleanup = []
        self.use_user_site = use_user_site
        self.target_dir = target_dir  # set from --target option
        self.pycompile = pycompile
        # Maps from req -> dependencies_of_req
        self._dependencies = defaultdict(list)

    def __str__(self):
        reqs = [req for req in self.requirements.values()
                if not req.comes_from]
        reqs.sort(key=lambda req: req.name.lower())
        return ' '.join([str(req.req) for req in reqs])

    def __repr__(self):
        reqs = [req for req in self.requirements.values()]
        reqs.sort(key=lambda req: req.name.lower())
        reqs_str = ', '.join([str(req.req) for req in reqs])
        return ('<%s object; %d requirement(s): %s>'
                % (self.__class__.__name__, len(reqs), reqs_str))

    def scan_requirement(self, req, parent_req_name=None,
                         extras_requested=None):
        """Scans a requirement.

        Returns an InstallRequirement that should be explored or None.

        :param parent_req_name: The name of the requirement that needed this
            added. The name is used because when multiple unnamed requirements
            resolve to the same name, we could otherwise end up with dependency
            links that point outside the Requirements set. parent_req must
            already be added. Note that None implies that this is a user
            supplied requirement, vs an inferred one.
        :param extras_requested: an iterable of extras used to evaluate the
            environment markers.
        """
        if not req.match_markers(extras_requested):
            logger.warning(
                "Ignoring %s: markers '%s' don't match your environment",
                req.name, req.markers,
            )
            return None

        # This check has to come after we filter requirements with the
        # environment markers.
        if req.link and req.link.is_wheel:
            wheel = Wheel(req.link.filename)
            if not wheel.supported():
                raise InstallationError(
                    "%s is not a supported wheel on this platform." %
                    wheel.filename
                )

        # FIXME: These cause action at a distance.
        req.use_user_site = self.use_user_site
        req.target_dir = self.target_dir
        req.pycompile = self.pycompile
        req.is_direct = (parent_req_name is None)

        name = req.name
        if not name:
            # url or path requirement w/o an egg fragment
            self.unnamed_requirements.append(req)
            return req

        try:
            existing_req = self.get_requirement(name)
        except KeyError:
            existing_req = None

        already_specified = (
            parent_req_name is None and existing_req and
            not existing_req.constraint and
            existing_req.extras == req.extras and
            not existing_req.req.specifier == req.req.specifier
        )
        if already_specified:
            raise InstallationError(
                'Double requirement given: %s (already in %s, name=%r)' %
                (req, existing_req, name)
            )

        if not existing_req:
            # Add requirement
            self.requirements[name] = req
            # FIXME: what about other normalizations?  E.g., _ vs. -?
            if name.lower() != name:
                self.requirement_aliases[name.lower()] = name
            result = req
        else:
            # Assume there's no need to scan, and that we've already
            # encountered this for scanning.
            result = None
            if not req.constraint and existing_req.constraint:
                unsatisfiable = req.link and not (
                    existing_req.link and
                    req.link.path == existing_req.link.path
                )
                if unsatisfiable:
                    self.reqs_to_cleanup.append(req)
                    raise InstallationError(
                        "Could not satisfy constraints for '%s': "
                        "installation from path or url cannot be "
                        "constrained to a version" % name
                    )

                # If we're now installing a constraint, mark the existing
                # object for real installation.
                existing_req.constraint = False
                existing_req.extras = tuple(sorted(
                    set(existing_req.extras).union(set(req.extras))
                ))
                logger.debug(
                    "Setting %s extras to: %s",
                    existing_req, existing_req.extras
                )
                # And now we need to scan this.
                result = existing_req
            # Canonicalise to the already-added object for the backref
            # check below.
            req = existing_req

        if parent_req_name:
            parent_req = self.get_requirement(parent_req_name)
            self._dependencies[parent_req].append(req)

        return result

    def has_requirement(self, project_name):
        name = project_name.lower()
        if (name in self.requirements and
           not self.requirements[name].constraint or
           name in self.requirement_aliases and
           not self.requirements[self.requirement_aliases[name]].constraint):
            return True
        return False

    @property
    def has_requirements(self):
        return list(req for req in self.requirements.values() if not
                    req.constraint) or self.unnamed_requirements

    def get_requirement(self, project_name):
        for name in project_name, project_name.lower():
            if name in self.requirements:
                return self.requirements[name]
            if name in self.requirement_aliases:
                return self.requirements[self.requirement_aliases[name]]
        raise KeyError("No project with the name %r" % project_name)

    def cleanup_files(self):
        """Clean up files, remove builds."""
        logger.debug('Cleaning up...')
        with indent_log():
            for req in self.reqs_to_cleanup:
                req.remove_temporary_source()

    def _to_install(self):
        """Create the installation order.

        The installation order is topological - requirements are installed
        before the requiring thing. We break cycles at an arbitrary point,
        and make no other guarantees.
        """
        # The current implementation, which we may change at any point
        # installs the user specified things in the order given, except when
        # dependencies must come earlier to achieve topological order.
        order = []
        ordered_reqs = set()

        def schedule(req):
            if req.satisfied_by or req in ordered_reqs:
                return
            if req.constraint:
                return
            ordered_reqs.add(req)
            for dep in self._dependencies[req]:
                schedule(dep)
            order.append(req)

        for req in self.requirements.values():
            schedule(req)
        return order

    def install(self, install_options, global_options=(), *args, **kwargs):
        """
        Install everything in this set (after having downloaded and unpacked
        the packages)
        """
        to_install = self._to_install()

        if to_install:
            logger.info(
                'Installing collected packages: %s',
                ', '.join([req.name for req in to_install]),
            )

        with indent_log():
            for requirement in to_install:
                if requirement.conflicts_with:
                    logger.info(
                        'Found existing installation: %s',
                        requirement.conflicts_with,
                    )
                    with indent_log():
                        requirement.uninstall(auto_confirm=True)
                try:
                    requirement.install(
                        install_options,
                        global_options,
                        *args,
                        **kwargs
                    )
                except:
                    should_rollback = (
                        requirement.conflicts_with and
                        not requirement.install_succeeded
                    )
                    # if install did not succeed, rollback previous uninstall
                    if should_rollback:
                        requirement.uninstalled_pathset.rollback()
                    raise
                else:
                    should_commit = (
                        requirement.conflicts_with and
                        requirement.install_succeeded
                    )
                    if should_commit:
                        requirement.uninstalled_pathset.commit()
                requirement.remove_temporary_source()

        return to_install
