from __future__ import absolute_import

import errno
import logging
import operator
import os
import shutil

from pip import cmdoptions
from pip.basecommand import RequirementCommand
from pip.cache import WheelCache
from pip.exceptions import (
    CommandError, InstallationError, PreviousBuildDirError
)
from pip.locations import distutils_scheme, virtualenv_no_global
from pip.req import RequirementSet
from pip.resolve import Resolver
from pip.status_codes import ERROR
from pip.utils import ensure_dir, get_installed_version
from pip.utils.filesystem import check_path_owner
from pip.utils.temp_dir import TempDirectory
from pip.wheel import WheelBuilder

try:
    import wheel
except ImportError:
    wheel = None


try:
    import wheel
except ImportError:
    wheel = None


logger = logging.getLogger(__name__)


class InstallCommand(RequirementCommand):
    """
    Install packages from:

    - PyPI (and other indexes) using requirement specifiers.
    - VCS project urls.
    - Local project directories.
    - Local or remote source archives.

    pip also supports installing from "requirements files", which provide
    an easy way to specify a whole environment to be installed.
    """
    name = 'install'

    usage = """
      %prog [options] <requirement specifier> [package-index-options] ...
      %prog [options] -r <requirements file> [package-index-options] ...
      %prog [options] [-e] <vcs project url> ...
      %prog [options] [-e] <local project path> ...
      %prog [options] <archive url/path> ..."""

    summary = 'Install packages.'

    def __init__(self, *args, **kw):
        super(InstallCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(cmdoptions.requirements())
        cmd_opts.add_option(cmdoptions.constraints())
        cmd_opts.add_option(cmdoptions.no_deps())
        cmd_opts.add_option(cmdoptions.pre())

        cmd_opts.add_option(cmdoptions.editable())
        cmd_opts.add_option(
            '-t', '--target',
            dest='target_dir',
            metavar='dir',
            default=None,
            help='Install packages into <dir>. '
                 'By default this will not replace existing files/folders in '
                 '<dir>. Use --upgrade to replace existing packages in <dir> '
                 'with new versions.'
        )
        cmd_opts.add_option(
            '--user',
            dest='use_user_site',
            action='store_true',
            help="Install to the Python user install directory for your "
                 "platform. Typically ~/.local/, or %APPDATA%\\Python on "
                 "Windows. (See the Python documentation for site.USER_BASE "
                 "for full details.)")
        cmd_opts.add_option(
            '--root',
            dest='root_path',
            metavar='dir',
            default=None,
            help="Install everything relative to this alternate root "
                 "directory.")
        cmd_opts.add_option(
            '--prefix',
            dest='prefix_path',
            metavar='dir',
            default=None,
            help="Installation prefix where lib, bin and other top-level "
                 "folders are placed")

        cmd_opts.add_option(cmdoptions.build_dir())

        cmd_opts.add_option(cmdoptions.src())

        cmd_opts.add_option(
            '-U', '--upgrade',
            dest='upgrade',
            action='store_true',
            help='Upgrade all specified packages to the newest available '
                 'version. The handling of dependencies depends on the '
                 'upgrade-strategy used.'
        )

        cmd_opts.add_option(
            '--upgrade-strategy',
            dest='upgrade_strategy',
            default='only-if-needed',
            choices=['only-if-needed', 'eager'],
            help='Determines how dependency upgrading should be handled '
                 '(default: %(default)s). '
                 '"eager" - dependencies are upgraded regardless of '
                 'whether the currently installed version satisfies the '
                 'requirements of the upgraded package(s). '
                 '"only-if-needed" -  are upgraded only when they do not '
                 'satisfy the requirements of the upgraded package(s).'
        )

        cmd_opts.add_option(
            '--force-reinstall',
            dest='force_reinstall',
            action='store_true',
            help='When upgrading, reinstall all packages even if they are '
                 'already up-to-date.')

        cmd_opts.add_option(
            '-I', '--ignore-installed',
            dest='ignore_installed',
            action='store_true',
            help='Ignore the installed packages (reinstalling instead).')

        cmd_opts.add_option(cmdoptions.ignore_requires_python())

        cmd_opts.add_option(cmdoptions.install_options())
        cmd_opts.add_option(cmdoptions.global_options())

        cmd_opts.add_option(
            "--compile",
            action="store_true",
            dest="compile",
            default=True,
            help="Compile Python source files to bytecode",
        )

        cmd_opts.add_option(
            "--no-compile",
            action="store_false",
            dest="compile",
            help="Do not compile Python source files to bytecode",
        )

        cmd_opts.add_option(cmdoptions.no_binary())
        cmd_opts.add_option(cmdoptions.only_binary())
        cmd_opts.add_option(cmdoptions.no_clean())
        cmd_opts.add_option(cmdoptions.require_hashes())
        cmd_opts.add_option(cmdoptions.progress_bar())

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    def run(self, options, args):
        cmdoptions.check_install_build_global(options)

        upgrade_strategy = "to-satisfy-only"
        if options.upgrade:
            upgrade_strategy = options.upgrade_strategy

        if options.build_dir:
            options.build_dir = os.path.abspath(options.build_dir)

        options.src_dir = os.path.abspath(options.src_dir)
        install_options = options.install_options or []
        if options.use_user_site:
            if options.prefix_path:
                raise CommandError(
                    "Can not combine '--user' and '--prefix' as they imply "
                    "different installation locations"
                )
            if virtualenv_no_global():
                raise InstallationError(
                    "Can not perform a '--user' install. User site-packages "
                    "are not visible in this virtualenv."
                )
            install_options.append('--user')
            install_options.append('--prefix=')

        target_temp_dir = TempDirectory(kind="target")
        if options.target_dir:
            options.ignore_installed = True
            options.target_dir = os.path.abspath(options.target_dir)
            if (os.path.exists(options.target_dir) and not
                    os.path.isdir(options.target_dir)):
                raise CommandError(
                    "Target path exists but is not a directory, will not "
                    "continue."
                )

            # Create a target directory for using with the target option
            target_temp_dir.create()
            install_options.append('--home=' + target_temp_dir.path)

        global_options = options.global_options or []

        with self._build_session(options) as session:

            finder = self._build_package_finder(options, session)
            build_delete = (not (options.no_clean or options.build_dir))
            wheel_cache = WheelCache(options.cache_dir, options.format_control)
            if options.cache_dir and not check_path_owner(options.cache_dir):
                logger.warning(
                    "The directory '%s' or its parent directory is not owned "
                    "by the current user and caching wheels has been "
                    "disabled. check the permissions and owner of that "
                    "directory. If executing pip with sudo, you may want "
                    "sudo's -H flag.",
                    options.cache_dir,
                )
                options.cache_dir = None

            with TempDirectory(
                options.build_dir, delete=build_delete, kind="install"
            ) as directory:
                requirement_set = RequirementSet(
                    build_dir=directory.path,
                    src_dir=options.src_dir,
                    target_dir=target_temp_dir.path,
                    pycompile=options.compile,
                    wheel_cache=wheel_cache,
                    require_hashes=options.require_hashes,
                    use_user_site=options.use_user_site,
                    progress_bar=options.progress_bar,
                )

                self.populate_requirement_set(
                    requirement_set, args, options, finder, session, self.name,
                    wheel_cache
                )

                try:
                    resolver = Resolver(
                        finder=finder,
                        session=session,
                        use_user_site=options.use_user_site,
                        upgrade_strategy=upgrade_strategy,
                        force_reinstall=options.force_reinstall,
                        ignore_dependencies=options.ignore_dependencies,
                        ignore_requires_python=options.ignore_requires_python,
                        ignore_installed=options.ignore_installed,
                        isolated=options.isolated_mode,
                    )
                    resolver.resolve(requirement_set)

                    # on -d don't do complex things like building
                    # wheels, and don't try to build wheels when wheel is
                    # not installed.
                    if wheel and options.cache_dir:
                        # build wheels before install.
                        wb = WheelBuilder(
                            requirement_set,
                            finder,
                            build_options=[],
                            global_options=[],
                        )
                        # Ignore the result: a failed wheel will be
                        # installed from the sdist/vcs whatever.
                        wb.build(session=session, autobuilding=True)

                    requirement_set.install(
                        install_options,
                        global_options,
                        root=options.root_path,
                        prefix=options.prefix_path,
                    )

                    possible_lib_locations = get_lib_location_guesses(
                        user=options.use_user_site,
                        home=target_temp_dir.path,
                        root=options.root_path,
                        prefix=options.prefix_path,
                        isolated=options.isolated_mode,
                    )
                    reqs = sorted(
                        requirement_set.successfully_installed,
                        key=operator.attrgetter('name'))
                    items = []
                    for req in reqs:
                        item = req.name
                        try:
                            installed_version = get_installed_version(
                                req.name, possible_lib_locations
                            )
                            if installed_version:
                                item += '-' + installed_version
                        except Exception:
                            pass
                        items.append(item)
                    installed = ' '.join(items)
                    if installed:
                        logger.info('Successfully installed %s', installed)
                except EnvironmentError as e:
                    message_parts = []

                    user_option_part = "Consider using the `--user` option"
                    permissions_part = "Check the permissions"

                    if e.errno == errno.EPERM:
                        if not options.use_user_site:
                            message_parts.extend([
                                user_option_part, " or ",
                                permissions_part.lower(),
                            ])
                        else:
                            message_parts.append(permissions_part)
                        message_parts.append("\n")

                    logger.error(
                        "".join(message_parts), exc_info=(options.verbose > 1)
                    )
                    return ERROR
                except PreviousBuildDirError:
                    options.no_clean = True
                    raise
                finally:
                    # Clean up
                    if not options.no_clean:
                        requirement_set.cleanup_files()

        if options.target_dir:
            self._handle_target_dir(
                options.target_dir, target_temp_dir, options.upgrade
            )
        return requirement_set

    def _handle_target_dir(self, target_dir, target_temp_dir, upgrade):
        ensure_dir(target_dir)

        # Checking both purelib and platlib directories for installed
        # packages to be moved to target directory
        lib_dir_list = []

        with target_temp_dir:
            # Checking both purelib and platlib directories for installed
            # packages to be moved to target directory
            scheme = distutils_scheme('', home=target_temp_dir.path)
            purelib_dir = scheme['purelib']
            platlib_dir = scheme['platlib']
            data_dir = scheme['data']

            if os.path.exists(purelib_dir):
                lib_dir_list.append(purelib_dir)
            if os.path.exists(platlib_dir) and platlib_dir != purelib_dir:
                lib_dir_list.append(platlib_dir)
            if os.path.exists(data_dir):
                lib_dir_list.append(data_dir)

            for lib_dir in lib_dir_list:
                for item in os.listdir(lib_dir):
                    if lib_dir == data_dir:
                        ddir = os.path.join(data_dir, item)
                        if any(s.startswith(ddir) for s in lib_dir_list[:-1]):
                            continue
                    target_item_dir = os.path.join(target_dir, item)
                    if os.path.exists(target_item_dir):
                        if not upgrade:
                            logger.warning(
                                'Target directory %s already exists. Specify '
                                '--upgrade to force replacement.',
                                target_item_dir
                            )
                            continue
                        if os.path.islink(target_item_dir):
                            logger.warning(
                                'Target directory %s already exists and is '
                                'a link. Pip will not automatically replace '
                                'links, please remove if replacement is '
                                'desired.',
                                target_item_dir
                            )
                            continue
                        if os.path.isdir(target_item_dir):
                            shutil.rmtree(target_item_dir)
                        else:
                            os.remove(target_item_dir)

                    shutil.move(
                        os.path.join(lib_dir, item),
                        target_item_dir
                    )


def get_lib_location_guesses(*args, **kwargs):
    scheme = distutils_scheme('', *args, **kwargs)
    return [scheme['purelib'], scheme['platlib']]
