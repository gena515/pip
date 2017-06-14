#!/usr/bin/env python
from __future__ import absolute_import

import locale
import logging
import optparse
import os
import sys
import time
import warnings

# 2016-06-17 barry@debian.org: urllib3 1.14 added optional support for socks,
# but if invoked (i.e. imported), it will issue a warning to stderr if socks
# isn't available.  requests unconditionally imports urllib3's socks contrib
# module, triggering this warning.  The warning breaks DEP-8 tests (because of
# the stderr output) and is just plain annoying in normal usage.  I don't want
# to add socks as yet another dependency for pip, nor do I want to allow-stder
# in the DEP-8 tests, so just suppress the warning.  pdb tells me this has to
# be done before the import of pip.vcs.
from pip._vendor.requests.packages.urllib3.exceptions import (
    DependencyWarning, InsecureRequestWarning
)

# This fixes a peculiarity when importing via __import__ - as we are
# initialising the pip module, "from pip import cmdoptions" is recursive
# and appears not to work properly in that situation.
import pip.cmdoptions
import pip.status_codes
from pip.baseparser import ConfigOptionParser, UpdatingDefaultsHelpFormatter
from pip.commands import commands_dict, get_closest_command, get_summaries
from pip.exceptions import CommandError, ConfigurationError, PipError
from pip.utils import deprecation, get_installed_distributions, get_prog
from pip.vcs import bazaar, git, mercurial, subversion  # noqa

# The version as used in the setup.py and the docs conf.py
__version__ = "10.0.0.dev0"

warnings.filterwarnings("ignore", category=DependencyWarning)  # noqa

# We want to inject the use of SecureTransport as early as possible so that any
# references or sessions or what have you are ensured to have it, however we
# only want to do this in the case that we're running on macOS and the linked
# OpenSSL is too old to handle TLSv1.2
try:
    import ssl
except ImportError:
    pass
else:
    if (sys.platform == "darwin" and
            ssl.OPENSSL_VERSION_NUMBER < 0x1000100f):  # OpenSSL 1.0.1
        try:
            from pip._vendor.requests.packages.urllib3.contrib import (
                securetransport,
            )
        except (ImportError, OSError):
            pass
        else:
            securetransport.inject_into_urllib3()

# assignment for flake8 to be happy
cmdoptions = pip.cmdoptions

logger = logging.getLogger(__name__)

# Hide the InsecureRequestWarning from urllib3
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def autocomplete():
    """Command and option completion for the main option parser (and options)
    and its subcommands (and options).

    Enable by sourcing one of the completion shell scripts (bash, zsh or fish).
    """
    # Don't complete if user hasn't sourced bash_completion file.
    if 'PIP_AUTO_COMPLETE' not in os.environ:
        return
    cwords = os.environ['COMP_WORDS'].split()[1:]
    cword = int(os.environ['COMP_CWORD'])
    try:
        current = cwords[cword - 1]
    except IndexError:
        current = ''

    subcommands = [cmd for cmd, summary in get_summaries()]
    options = []
    # subcommand
    try:
        subcommand_name = [w for w in cwords if w in subcommands][0]
    except IndexError:
        subcommand_name = None

    parser = create_main_parser()
    # subcommand options
    if subcommand_name:
        # special case: 'help' subcommand has no options
        if subcommand_name == 'help':
            sys.exit(1)
        # special case: list locally installed dists for uninstall command
        if subcommand_name == 'uninstall' and not current.startswith('-'):
            installed = []
            lc = current.lower()
            for dist in get_installed_distributions(local_only=True):
                if dist.key.startswith(lc) and dist.key not in cwords[1:]:
                    installed.append(dist.key)
            # if there are no dists installed, fall back to option completion
            if installed:
                for dist in installed:
                    print(dist)
                sys.exit(1)

        subcommand = commands_dict[subcommand_name]()
        options += [(opt.get_opt_string(), opt.nargs)
                    for opt in subcommand.parser.option_list_all
                    if opt.help != optparse.SUPPRESS_HELP]

        # filter out previously specified options from available options
        prev_opts = [x.split('=')[0] for x in cwords[1:cword - 1]]
        options = [(x, v) for (x, v) in options if x not in prev_opts]
        # filter options by current input
        options = [(k, v) for k, v in options if k.startswith(current)]
        for option in options:
            opt_label = option[0]
            # append '=' to options which require args
            if option[1]:
                opt_label += '='
            print(opt_label)
    else:
        # show main parser options only when necessary
        if current.startswith('-') or current.startswith('--'):
            opts = [i.option_list for i in parser.option_groups]
            opts.append(parser.option_list)
            opts = (o for it in opts for o in it)

            subcommands += [i.get_opt_string() for i in opts
                            if i.help != optparse.SUPPRESS_HELP]

        print(' '.join([x for x in subcommands if x.startswith(current)]))
    sys.exit(1)


def create_main_parser():
    parser_kw = {
        'usage': '\n%prog <command> [options]',
        'add_help_option': False,
        'formatter': UpdatingDefaultsHelpFormatter(),
        'name': 'global',
        'prog': get_prog(),
    }

    parser = ConfigOptionParser(**parser_kw)
    parser.disable_interspersed_args()

    pip_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.version = 'pip %s from %s (python %s)' % (
        __version__, pip_pkg_dir, sys.version[:3])

    # add the general options
    gen_opts = cmdoptions.make_option_group(cmdoptions.general_group, parser)
    parser.add_option_group(gen_opts)

    parser.main = True  # so the help formatter knows

    # create command listing for description
    command_summaries = get_summaries()
    description = [''] + ['%-27s %s' % (i, j) for i, j in command_summaries]
    parser.description = '\n'.join(description)

    return parser


def parseopts(args):
    parser = create_main_parser()

    # Note: parser calls disable_interspersed_args(), so the result of this
    # call is to split the initial args into the general options before the
    # subcommand and everything else.
    # For example:
    #  args: ['--timeout=5', 'install', '--user', 'INITools']
    #  general_options: ['--timeout==5']
    #  args_else: ['install', '--user', 'INITools']
    general_options, args_else = parser.parse_args(args)

    # --version
    if general_options.version:
        sys.stdout.write(parser.version)
        sys.stdout.write(os.linesep)
        sys.exit()

    # pip || pip help -> print_help()
    if not args_else or (args_else[0] == 'help' and len(args_else) == 1):
        parser.print_help()
        sys.exit()

    # the subcommand name
    cmd_name = args_else[0]

    # all the args without the subcommand
    cmd_args = args[:]
    cmd_args.remove(cmd_name)

    # Perform diagnosis of what the user typed
    if cmd_name not in commands_dict:
        # these were manually chosen
        suggest_cutoff = 0.6
        autocorrect_cutoff = 0.8

        # Determine if user wants autocorrect.
        try:
            autocorrect_delay = parser.config.get_value("global.autocorrect")
        except ConfigurationError:
            autocorrect_delay = None

        guess, score = get_closest_command(cmd_name)

        # Decide what message has to be shown to user
        msg = 'pip does not have a command "%s"' % cmd_name
        if score > suggest_cutoff:
            msg += ' - did you mean "%s"?' % guess

        allowed_to_autocorrect = (
            autocorrect_delay is not None and score > autocorrect_cutoff
        )
        if not allowed_to_autocorrect:
            raise CommandError(msg)

        # Show a message to the user that pip is assuming.
        msg = (
            'You called a pip command named "%s" which does not exist.\n'
            'Assuming you meant "%s", pip will continue in %.1f seconds...'
        )

        logger.warning(msg, cmd_name, guess, autocorrect_delay)
        try:
            time.sleep(autocorrect_delay)
        except KeyboardInterrupt:
            logger.critical('Operation cancelled by user')
            sys.exit(pip.status_codes.ERROR)

        # Assume and proceed.
        cmd_name = guess

    return cmd_name, cmd_args


def check_isolated(args):
    isolated = False

    if "--isolated" in args:
        isolated = True

    return isolated


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # Configure our deprecation warnings to be sent through loggers
    deprecation.install_warning_logger()

    autocomplete()

    try:
        cmd_name, cmd_args = parseopts(args)
    except PipError as exc:
        sys.stderr.write("ERROR: %s" % exc)
        sys.stderr.write(os.linesep)
        sys.exit(1)

    # Needed for locale.getpreferredencoding(False) to work
    # in pip.utils.encoding.auto_decode
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error as e:
        # setlocale can apparently crash if locale are uninitialized
        logger.debug("Ignoring error %s when setting locale", e)
    command = commands_dict[cmd_name](isolated=check_isolated(cmd_args))
    return command.main(cmd_args)


if __name__ == '__main__':
    sys.exit(main())
