"""
Package containing all pip commands
"""
from __future__ import absolute_import

import logging

from pip.commands.completion import CompletionCommand
from pip.commands.freeze import FreezeCommand
from pip.commands.help import HelpCommand
from pip.commands.list import ListCommand
from pip.commands.search import SearchCommand
from pip.commands.show import ShowCommand
from pip.commands.install import InstallCommand
from pip.commands.uninstall import UninstallCommand
from pip.commands.unzip import UnzipCommand
from pip.commands.zip import ZipCommand
from pip.commands.wheel import WheelCommand
from pip._vendor import pkg_resources


logger = logging.getLogger(__name__)


commands_dict = {
    CompletionCommand.name: CompletionCommand,
    FreezeCommand.name: FreezeCommand,
    HelpCommand.name: HelpCommand,
    SearchCommand.name: SearchCommand,
    ShowCommand.name: ShowCommand,
    InstallCommand.name: InstallCommand,
    UninstallCommand.name: UninstallCommand,
    UnzipCommand.name: UnzipCommand,
    ZipCommand.name: ZipCommand,
    ListCommand.name: ListCommand,
    WheelCommand.name: WheelCommand,
}


commands_order = [
    InstallCommand,
    UninstallCommand,
    FreezeCommand,
    ListCommand,
    ShowCommand,
    SearchCommand,
    WheelCommand,
    ZipCommand,
    UnzipCommand,
    HelpCommand,
]


for ep in pkg_resources.iter_entry_points('pip.command.v1'):
    try:
        command = ep.load()
        logger.debug('from entry point %r loaded command = %r' % (ep, command))
        commands_dict[command.name] = command
    except Exception as e:
        logger.warning('Failed to load command plugin: %r (%r): %s'
                       % (ep, command, e))


def get_summaries(ignore_hidden=True, ordered=True):
    """Yields sorted (command name, command summary) tuples."""

    if ordered:
        cmditems = _sort_commands(commands_dict, commands_order)
    else:
        cmditems = commands_dict.items()

    for name, command_class in cmditems:
        if ignore_hidden and command_class.hidden:
            continue

        yield (name, command_class.summary)


def get_similar_commands(name):
    """Command name auto-correct."""
    from difflib import get_close_matches

    name = name.lower()

    close_commands = get_close_matches(name, commands_dict.keys())

    if close_commands:
        return close_commands[0]
    else:
        return False


def _sort_commands(cmddict, order):
    def keyfn(key):
        try:
            return order.index(key[1])
        except ValueError:
            # unordered items should come last
            return 0xff

    return sorted(cmddict.items(), key=keyfn)
