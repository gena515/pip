from __future__ import absolute_import

from pip.basecommand import SUCCESS, Command
from pip.exceptions import CommandError


class HelpCommand(Command):
    """Show help for commands"""
    name = 'help'
    usage = """
      %prog <command>"""
    summary = 'Show help for commands.'
    ignore_require_venv = True

    def run(self, options, args):
        from pip.commands import commands_dict, get_closest_command

        try:
            # 'pip help' with no args is handled by pip.__init__.parseopt()
            cmd_name = args[0]  # the command we need help for
        except IndexError:
            return SUCCESS

        if cmd_name not in commands_dict:
            guess, score = get_closest_command(cmd_name)
            suggest_cut_off = 0.6

            msg = ['unknown command "%s"' % cmd_name]
            if guess and score > suggest_cut_off:
                msg.append('maybe you meant "%s"' % guess)

            raise CommandError(' - '.join(msg))

        command = commands_dict[cmd_name]()
        command.parser.print_help()

        return SUCCESS
