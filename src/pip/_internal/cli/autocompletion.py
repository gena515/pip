"""Logic that powers autocompletion installed by ``pip completion``.
"""

import optparse
import os
import sys
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional

from pip._internal.cli.main_parser import create_main_parser
from pip._internal.commands import commands_dict, create_command
from pip._internal.metadata import get_default_environment


DELIMITER = "*"

def autocomplete() -> None:
    """Entry Point for completion of main and subcommand options."""
    # Don't complete if user hasn't sourced bash_completion file.
    if "PIP_AUTO_COMPLETE" not in os.environ:
        return
    cwords = os.environ["COMP_WORDS"].split()[1:]
    cword = int(os.environ["COMP_CWORD"])
    try:
        current = cwords[cword - 1]
    except IndexError:
        current = ""

    parser = create_main_parser()
    subcommands: Dict[str,str] = {
        name: command.summary for name, command in commands_dict.items()
    }
    options = []

    # subcommand
    subcommand_name: Optional[str] = None
    for word in cwords:
        if word in subcommands:
            subcommand_name = word
            break
    # subcommand options
    if subcommand_name is not None:
        # special case: 'help' subcommand has no options
        if subcommand_name == "help":
            sys.exit(1)
        # special case: list locally installed dists for show and uninstall
        should_list_installed = not current.startswith("-") and subcommand_name in [
            "show",
            "uninstall",
        ]
        if should_list_installed:
            env = get_default_environment()
            lc = current.lower()
            installed = [
                (dist.canonical_name, dist.version)
                for dist in env.iter_installed_distributions(local_only=True)
                if dist.canonical_name.startswith(lc)
                and dist.canonical_name not in cwords[1:]
            ]
            # if there are no dists installed, fall back to option completion
            if installed:
                for dist, version in installed:
                    output_completion_with_description(name=dist, description=version)
                sys.exit(1)

        subcommand = create_command(subcommand_name)

        for opt in subcommand.parser.option_list_all:
            if opt.help != optparse.SUPPRESS_HELP:
                for opt_str in opt._long_opts + opt._short_opts:
                    options.append((opt_str, opt.nargs, opt.help))

        # filter out previously specified options from available options
        prev_opts = [x.split("=")[0] for x in cwords[1 : cword - 1]]
        options = [(x, v, h) for (x, v) in options if x not in prev_opts]
        # filter options by current input
        options = [(k, v, h) for k, v, h in options if k.startswith(current)]
        # get completion type given cwords and available subcommand options
        completion_type = get_path_completion_type(
            cwords,
            cword,
            subcommand.parser.option_list_all,
        )
        # get completion files and directories if ``completion_type`` is
        # ``<file>``, ``<dir>`` or ``<path>``
        if completion_type:
            paths = auto_complete_paths(current, completion_type)
            options = [(path, 0, "") for path in paths]
        for label, nargs, description in options:
            # append '=' to options which require args
            if nargs and label.startswith("--"):
                label += "="
            output_completion_with_description(label, description)
    else:
        # show main parser options only when necessary

        opts = [i.option_list for i in parser.option_groups]
        opts.append(parser.option_list)
        flattened_opts = chain.from_iterable(opts)
        if current.startswith("-"):
            for opt in flattened_opts:
                if opt.help != optparse.SUPPRESS_HELP:
                    subcommands.update({name: opt.help for name in opt._long_opts})
                    subcommands.update({name: opt.help for name in opt._short_opts})
        else:
            # get completion type given cwords and all available options
            completion_type = get_path_completion_type(cwords, cword, flattened_opts)
            if completion_type:
                subcommands = {
                    command: ""
                    for command in auto_complete_paths(current, completion_type)
                }

        for name, description in subcommands.items():
            if name.startswith(current):
                output_completion_with_description(name, description)
    sys.exit(1)


def output_completion_with_description(name: str, description: str="") -> None:
    """Prints the string for completion with its description in a consistent way."""
    # Make descriptions oneliners so they're grouped as one.
    description = " ".join(description.splitlines())
    print(name + DELIMITER + description)

def get_path_completion_type(
    cwords: List[str], cword: int, opts: Iterable[Any]
) -> Optional[str]:
    """Get the type of path completion (``file``, ``dir``, ``path`` or None)

    :param cwords: same as the environmental variable ``COMP_WORDS``
    :param cword: same as the environmental variable ``COMP_CWORD``
    :param opts: The available options to check
    :return: path completion type (``file``, ``dir``, ``path`` or None)
    """
    if cword < 2 or not cwords[cword - 2].startswith("-"):
        return None
    for opt in opts:
        if opt.help == optparse.SUPPRESS_HELP:
            continue
        for o in str(opt).split("/"):
            if cwords[cword - 2].split("=")[0] == o:
                if not opt.metavar or any(
                    x in ("path", "file", "dir") for x in opt.metavar.split("/")
                ):
                    return opt.metavar
    return None


def auto_complete_paths(current: str, completion_type: str) -> Iterable[str]:
    """If ``completion_type`` is ``file`` or ``path``, list all regular files
    and directories starting with ``current``; otherwise only list directories
    starting with ``current``.

    :param current: The word to be completed
    :param completion_type: path completion type(`file`, `path` or `dir`)i
    :return: A generator of regular files and/or directories
    """
    directory, filename = os.path.split(current)
    current_path = os.path.abspath(directory)
    # Don't complete paths if they can't be accessed
    if not os.access(current_path, os.R_OK):
        return
    filename = os.path.normcase(filename)
    # list all files that start with ``filename``
    file_list = (
        x for x in os.listdir(current_path) if os.path.normcase(x).startswith(filename)
    )
    for f in file_list:
        opt = os.path.join(current_path, f)
        comp_file = os.path.normcase(os.path.join(directory, f))
        # complete regular files when there is not ``<dir>`` after option
        # complete directories when there is ``<file>``, ``<path>`` or
        # ``<dir>``after option
        if completion_type != "dir" and os.path.isfile(opt):
            yield comp_file
        elif os.path.isdir(opt):
            yield os.path.join(comp_file, "")
