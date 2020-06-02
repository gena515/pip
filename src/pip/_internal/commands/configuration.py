# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

import logging
import os
import subprocess

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
    Configuration,
    get_configuration_files,
    kinds,
)
from pip._internal.exceptions import PipError
from pip._internal.utils.misc import get_prog, write_output
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import List
    from optparse import Values

logger = logging.getLogger(__name__)


class ConfigurationCommand(Command):
    """
    Manage local and global configuration.

    Subcommands:

    - list: List the active configuration (or from the file specified)
    - edit: Edit the configuration file in an editor
    - get: Get the value associated with name
    - set: Set the name=value
    - unset: Unset the value associated with name

    If none of --user, --global and --site are passed, a virtual
    environment configuration file is used if one is active and the file
    exists. Otherwise, all modifications happen on the to the user file by
    default.
    """

    ignore_require_venv = True
    usage = """
        %prog [<file-option>] list
        %prog [<file-option>] [--editor <editor-path>] edit

        %prog [<file-option>] get name
        %prog [<file-option>] set name value
        %prog [<file-option>] unset name
    """

    def __init__(self, name, summary, isolated=False):
        super(ConfigurationCommand, self).__init__(
            name, summary, isolated=isolated
        )

        self.configuration = None

    def add_options(self):
        self.cmd_opts.add_option(
            '--editor',
            dest='editor',
            action='store',
            default=None,
            help=(
                'Editor to use to edit the file. Uses VISUAL or EDITOR '
                'environment variables if not provided.'
            )
        )

        self.cmd_opts.add_option(
            '--global',
            dest='global_file',
            action='store_true',
            default=False,
            help='Use the system-wide configuration file only'
        )

        self.cmd_opts.add_option(
            '--user',
            dest='user_file',
            action='store_true',
            default=False,
            help='Use the user configuration file only'
        )

        self.cmd_opts.add_option(
            '--site',
            dest='site_file',
            action='store_true',
            default=False,
            help='Use the current environment configuration file only'
        )

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        # type: (Values, List[str]) -> int
        handlers = {
            "list": self.list_values,
            "edit": self.open_in_editor,
            "get": self.get_name,
            "set": self.set_name_value,
            "unset": self.unset_name
        }

        # Determine action
        if not args or args[0] not in handlers:
            logger.error("Need an action ({}) to perform.".format(
                ", ".join(sorted(handlers)))
            )
            return ERROR

        action = args[0]

        # Determine which configuration files are to be loaded
        #    Depends on whether the command is modifying.
        try:
            load_only = self._determine_file(
                options, need_value=(action in ["get", "set", "unset", "edit"])
            )
        except PipError as e:
            logger.error(e.args[0])
            return ERROR

        # Load a new configuration
        self.configuration = Configuration(
            isolated=options.isolated_mode, load_only=load_only
        )
        self.configuration.load()

        # Error handling happens here, not in the action-handlers.
        try:
            handlers[action](options, args[1:])
        except PipError as e:
            logger.error(e.args[0])
            return ERROR

        return SUCCESS

    def _determine_file(self, options, need_value):
        file_options = [key for key, value in (
            (kinds.USER, options.user_file),
            (kinds.GLOBAL, options.global_file),
            (kinds.SITE, options.site_file),
        ) if value]

        if not file_options:
            if not need_value:
                return None
            # Default to user, unless there's a site file.
            elif any(
                os.path.exists(site_config_file)
                for site_config_file in get_configuration_files()[kinds.SITE]
            ):
                return kinds.SITE
            else:
                return kinds.USER
        elif len(file_options) == 1:
            return file_options[0]

        raise PipError(
            "Need exactly one file to operate upon "
            "(--user, --site, --global) to perform."
        )

    def list_values(self, options, args):
        self._get_n_args(args, "list", n=0)

        for key, value in sorted(self.configuration.items()):
            write_output("%s=%r", key, value)

    def get_name(self, options, args):
        key = self._get_n_args(args, "get [name]", n=1)
        value = self.configuration.get_value(key)

        write_output("%s", value)

    def set_name_value(self, options, args):
        key, value = self._get_n_args(args, "set [name] [value]", n=2)
        self.configuration.set_value(key, value)

        self._save_configuration()

    def unset_name(self, options, args):
        key = self._get_n_args(args, "unset [name]", n=1)
        self.configuration.unset_value(key)

        self._save_configuration()

    def open_in_editor(self, options, args):
        editor = self._determine_editor(options)

        fname = self.configuration.get_file_to_edit()
        if fname is None:
            raise PipError("Could not determine appropriate file.")

        try:
            subprocess.check_call([editor, fname])
        except subprocess.CalledProcessError as e:
            raise PipError(
                "Editor Subprocess exited with exit code {}"
                .format(e.returncode)
            )

    def _get_n_args(self, args, example, n):
        """Helper to make sure the command got the right number of arguments
        """
        if len(args) != n:
            msg = (
                'Got unexpected number of arguments, expected {}. '
                '(example: "{} config {}")'
            ).format(n, get_prog(), example)
            raise PipError(msg)

        if n == 1:
            return args[0]
        else:
            return args

    def _save_configuration(self):
        # We successfully ran a modifying command. Need to save the
        # configuration.
        try:
            self.configuration.save()
        except Exception:
            logger.error(
                "Unable to save configuration. Please report this as a bug.",
                exc_info=1
            )
            raise PipError("Internal Error.")

    def _determine_editor(self, options):
        if options.editor is not None:
            return options.editor
        elif "VISUAL" in os.environ:
            return os.environ["VISUAL"]
        elif "EDITOR" in os.environ:
            return os.environ["EDITOR"]
        else:
            raise PipError("Could not determine editor to use.")
