from __future__ import absolute_import

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.basecommand import Command
from pip._internal.exceptions import InstallationError
from pip._internal.req import InstallRequirement, parse_requirements
from pip._internal.utils.misc import confirm_dependencies, sorted_reqs


class UninstallCommand(Command):
    """
    Uninstall packages.

    pip is able to uninstall most installed packages. Known exceptions are:

    - Pure distutils packages installed with ``python setup.py install``, which
      leave behind no metadata to determine what files were installed.
    - Script wrappers installed by ``python setup.py develop``.
    """
    name = 'uninstall'
    usage = """
      %prog [options] <package> ...
      %prog [options] -r <requirements file> ..."""
    summary = 'Uninstall packages.'

    def __init__(self, *args, **kw):
        super(UninstallCommand, self).__init__(*args, **kw)
        self.cmd_opts.add_option(
            '-r', '--requirement',
            dest='requirements',
            action='append',
            default=[],
            metavar='file',
            help='Uninstall all the packages listed in the given requirements '
                 'file.  This option can be used multiple times.',
        )
        self.cmd_opts.add_option(
            '-y', '--yes',
            dest='yes',
            action='store_true',
            help="Don't ask for confirmation of uninstall deletions.")

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        with self._build_session(options) as session:
            reqs_to_uninstall = {}
            for name in args:
                req = InstallRequirement.from_line(
                    name, isolated=options.isolated_mode,
                )
                if req.name:
                    reqs_to_uninstall[canonicalize_name(req.name)] = req
            for filename in options.requirements:
                for req in parse_requirements(
                        filename,
                        options=options,
                        session=session):
                    if req.name:
                        reqs_to_uninstall[canonicalize_name(req.name)] = req
            if not reqs_to_uninstall:
                raise InstallationError(
                    'You must give at least one requirement to %(name)s (see '
                    '"pip help %(name)s")' % dict(name=self.name)
                )
            for req in sorted_reqs(reqs_to_uninstall):
                if not (options.yes or confirm_dependencies(req)):
                    continue
                req.uninstall(
                    auto_confirm=options.yes, verbose=options.verbose != 0
                )
                req.uninstalled_pathset.commit()
