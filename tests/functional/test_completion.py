import os
import sys

import pytest


def test_completion_for_bash(script):
    """
    Test getting completion for bash shell
    """
    bash_completion = """\
_pip_completion()
{
    COMPREPLY=( $( COMP_WORDS="${COMP_WORDS[*]}" \\
                   COMP_CWORD=$COMP_CWORD \\
                   PIP_AUTO_COMPLETE=1 $1 ) )
}
complete -o default -F _pip_completion pip"""

    result = script.pip('completion', '--bash')
    assert bash_completion in result.stdout, 'bash completion is wrong'


def test_completion_for_zsh(script):
    """
    Test getting completion for zsh shell
    """
    zsh_completion = """\
function _pip_completion {
  local words cword
  read -Ac words
  read -cn cword
  reply=( $( COMP_WORDS="$words[*]" \\
             COMP_CWORD=$(( cword-1 )) \\
             PIP_AUTO_COMPLETE=1 $words[1] ) )
}
compctl -K _pip_completion pip"""

    result = script.pip('completion', '--zsh')
    assert zsh_completion in result.stdout, 'zsh completion is wrong'


def test_completion_for_fish(script):
    """
    Test getting completion for fish shell
    """
    fish_completion = """\
function __fish_complete_pip
    set -lx COMP_WORDS (commandline -o) ""
    set -lx COMP_CWORD ( \\
        math (contains -i -- (commandline -t) $COMP_WORDS)-1 \\
    )
    set -lx PIP_AUTO_COMPLETE 1
    string split \\  -- (eval $COMP_WORDS[1])
end
complete -fa "(__fish_complete_pip)" -c pip"""

    result = script.pip('completion', '--fish')
    assert fish_completion in result.stdout, 'fish completion is wrong'


def test_completion_for_unknown_shell(script):
    """
    Test getting completion for an unknown shell
    """
    error_msg = 'no such option: --myfooshell'
    result = script.pip('completion', '--myfooshell', expect_error=True)
    assert error_msg in result.stderr, 'tests for an unknown shell failed'


def test_completion_alone(script):
    """
    Test getting completion for none shell, just pip completion
    """
    result = script.pip('completion', expect_error=True)
    assert 'ERROR: You must pass --bash or --fish or --zsh' in result.stderr, (
        'completion alone failed -- ' + result.stderr
    )


def setup_completion(script, words, cword):
    script.environ = os.environ.copy()
    script.environ['PIP_AUTO_COMPLETE'] = '1'
    script.environ['COMP_WORDS'] = words
    script.environ['COMP_CWORD'] = cword

    # expect_error is True because autocomplete exists with 1 status code
    result = script.run(
        'python',
        '-c',
        'import pip._internal;pip._internal.autocomplete()',
        expect_error=True,
    )

    return result, script


def test_completion_for_un_snippet(script):
    """
    Test getting completion for ``un`` should return uninstall
    """

    res, env = setup_completion(script, 'pip un', '1')
    assert res.stdout.strip().split() == ['uninstall'], res.stdout


def test_completion_for_default_parameters(script):
    """
    Test getting completion for ``--`` should contain --help
    """

    res, env = setup_completion(script, 'pip --', '1')
    assert '--help' in res.stdout, "autocomplete function could not complete ``--``"


def test_completion_option_for_command(script):
    """
    Test getting completion for ``--`` in command (eg. pip search --)
    """

    res, env = setup_completion(script, 'pip search --', '2')
    assert '--help' in res.stdout, "autocomplete function could not complete ``--``"


def test_completion_short_option(script):
    """
    Test getting completion for short options after ``-`` (eg. pip -)
    """

    res, env = setup_completion(script, 'pip -', '1')

    assert (
        '-h' in res.stdout.split()
    ), "autocomplete function could not complete short options after ``-``"


def test_completion_short_option_for_command(script):
    """
    Test getting completion for short options after ``-`` in command
    (eg. pip search -)
    """

    res, env = setup_completion(script, 'pip search -', '2')

    assert (
        '-h' in res.stdout.split()
    ), "autocomplete function could not complete short options after ``-``"


@pytest.mark.parametrize('flag', ['--bash', '--zsh', '--fish'])
def test_completion_uses_same_executable_name(script, flag):
    expect_stderr = sys.version_info[:2] == (3, 3)
    executable_name = 'pip{}'.format(sys.version_info[0])
    result = script.run(
        executable_name, 'completion', flag, expect_stderr=expect_stderr
    )
    assert executable_name in result.stdout
