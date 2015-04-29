# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os
import warnings

from pip.basecommand import RequirementCommand
from pip.index import PackageFinder
from pip.exceptions import CommandError, PreviousBuildDirError
from pip.req import RequirementSet
from pip.utils import import_or_raise, normalize_path
from pip.req.req_cache import RequirementCache
from pip.utils.deprecation import RemovedInPip10Warning
from pip.wheel import WheelCache, WheelBuilder
from pip import cmdoptions

DEFAULT_WHEEL_DIR = os.path.join(normalize_path(os.curdir), 'wheelhouse')


logger = logging.getLogger(__name__)


class WheelCommand(RequirementCommand):
    """
    Build Wheel archives for your requirements and dependencies.

    Wheel is a built-package format, and offers the advantage of not
    recompiling your software during every install. For more details, see the
    wheel docs: http://wheel.readthedocs.org/en/latest.

    Requirements: setuptools>=0.8, and wheel.

    'pip wheel' uses the bdist_wheel setuptools extension from the wheel
    package to build individual wheels.

    """

    name = 'wheel'
    usage = """
      %prog [options] <requirement specifier> ...
      %prog [options] -r <requirements file> ...
      %prog [options] [-e] <vcs project url> ...
      %prog [options] [-e] <local project path> ...
      %prog [options] <archive url/path> ..."""

    summary = 'Build wheels from your requirements.'

    def __init__(self, *args, **kw):
        super(WheelCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(
            '-w', '--wheel-dir',
            dest='wheel_dir',
            metavar='dir',
            default=DEFAULT_WHEEL_DIR,
            help=("Build wheels into <dir>, where the default is "
                  "'<cwd>/wheelhouse'."),
        )
        cmd_opts.add_option(cmdoptions.use_wheel())
        cmd_opts.add_option(cmdoptions.no_use_wheel())
        cmd_opts.add_option(cmdoptions.no_binary())
        cmd_opts.add_option(cmdoptions.only_binary())
        cmd_opts.add_option(
            '--build-option',
            dest='build_options',
            metavar='options',
            action='append',
            help="Extra arguments to be supplied to 'setup.py bdist_wheel'.")
        cmd_opts.add_option(cmdoptions.constraints())
        cmd_opts.add_option(cmdoptions.editable())
        cmd_opts.add_option(cmdoptions.requirements())
        cmd_opts.add_option(cmdoptions.src())
        cmd_opts.add_option(cmdoptions.no_deps())
        cmd_opts.add_option(cmdoptions.build_dir())

        cmd_opts.add_option(
            '--global-option',
            dest='global_options',
            action='append',
            metavar='options',
            help="Extra global options to be supplied to the setup.py "
            "call before the 'bdist_wheel' command.")

        cmd_opts.add_option(
            '--pre',
            action='store_true',
            default=False,
            help=("Include pre-release and development versions. By default, "
                  "pip only finds stable versions."),
        )

        cmd_opts.add_option(cmdoptions.no_clean())

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    def check_required_packages(self):
        import_or_raise(
            'wheel.bdist_wheel',
            CommandError,
            "'pip wheel' requires the 'wheel' package. To fix this, run: "
            "pip install wheel"
        )
        pkg_resources = import_or_raise(
            'pkg_resources',
            CommandError,
            "'pip wheel' requires setuptools >= 0.8 for dist-info support."
            " To fix this, run: pip install --upgrade setuptools"
        )
        if not hasattr(pkg_resources, 'DistInfoDistribution'):
            raise CommandError(
                "'pip wheel' requires setuptools >= 0.8 for dist-info "
                "support. To fix this, run: pip install --upgrade "
                "setuptools"
            )

    def run(self, options, args):
        self.check_required_packages()
        cmdoptions.resolve_wheel_no_use_binary(options)
        cmdoptions.check_install_build_global(options)

        if options.allow_external:
            warnings.warn(
                "--allow-external has been deprecated and will be removed in "
                "the future. Due to changes in the repository protocol, it no "
                "longer has any effect.",
                RemovedInPip10Warning,
            )

        if options.allow_all_external:
            warnings.warn(
                "--allow-all-external has been deprecated and will be removed "
                "in the future. Due to changes in the repository protocol, it "
                "no longer has any effect.",
                RemovedInPip10Warning,
            )

        if options.allow_unverified:
            warnings.warn(
                "--allow-unverified has been deprecated and will be removed "
                "in the future. Due to changes in the repository protocol, it "
                "no longer has any effect.",
                RemovedInPip10Warning,
            )

        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index:
            logger.info('Ignoring indexes: %s', ','.join(index_urls))
            index_urls = []

        if options.build_dir:
            options.build_dir = os.path.abspath(options.build_dir)

        with self._build_session(options) as session:

            finder = PackageFinder(
                find_links=options.find_links,
                format_control=options.format_control,
                index_urls=index_urls,
                allow_all_prereleases=options.pre,
                trusted_hosts=options.trusted_hosts,
                process_dependency_links=options.process_dependency_links,
                session=session,
            )

            build_delete = (not (options.no_clean or options.build_dir))
            wheel_cache = WheelCache(options.cache_dir, options.format_control)
            with RequirementCache(
                    options.build_dir, delete=build_delete,
                    src_dir=options.src_dir) as req_cache:
                requirement_set = RequirementSet(
                    download_dir=None,
                    ignore_dependencies=options.ignore_dependencies,
                    ignore_installed=True,
                    isolated=options.isolated_mode,
                    req_cache=req_cache,
                    session=session,
                    wheel_cache=wheel_cache,
                    wheel_download_dir=options.wheel_dir
                )

                self.populate_requirement_set(
                    requirement_set, args, options, finder, session, self.name,
                    wheel_cache
                )

                if not requirement_set.has_requirements:
                    return

                try:
                    # build wheels
                    wb = WheelBuilder(
                        requirement_set,
                        finder,
                        build_options=options.build_options or [],
                        global_options=options.global_options or [],
                    )
                    if not wb.build():
                        raise CommandError(
                            "Failed to build one or more wheels"
                        )
                except PreviousBuildDirError:
                    options.no_clean = True
                    raise
                finally:
                    if not options.no_clean:
                        requirement_set.cleanup_files()
