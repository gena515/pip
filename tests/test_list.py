import re
import textwrap
from tests.test_pip import pyversion, reset_env, run_pip, write_file


def test_list_command():
    """
    Test default behavior of list command.

    """
    reset_env()
    run_pip('install', 'INITools==0.2', 'simplejson==2.0.0')
    result = run_pip('list')
    assert 'initools (0.2)' in result.stdout
    assert 'simplejson (2.0.0)' in result.stdout


def test_local_flag():
    """
    Test the behavior of --local flag in the list command

    """
    reset_env()
    run_pip('install', 'simplejson==2.0.0')
    result = run_pip('list', '--local')
    assert 'simplejson (2.0.0)' in result.stdout


def test_uptodate_flag():
    """
    Test the behavior of --uptodate flag in the list command

    """
    reset_env()
    run_pip('install', 'simplejson', 'mock==0.8.0')
    result = run_pip('list', '--uptodate')
    output = str(result)
    assert not 'mock' in output
    assert 'simplejson' in output


def test_outdated_flag():
    """
    Test the behavior of --outdated flag in the list command

    """
    env = reset_env()
    total_re = re.compile('LATEST: +([0-9.]+)')
    write_file('initools-req.txt', textwrap.dedent("""\
        INITools==0.2
        # and something else to test out:
        simplejson==2.0.0
        """))
    run_pip('install', '-r', env.scratch_path/'initools-req.txt')
    result = run_pip('search', 'simplejson')
    simplejson_ver = total_re.search(str(result)).group(1)
    result = run_pip('search', 'INITools')
    initools_ver = total_re.search(str(result)).group(1)
    result = run_pip('list', '--outdated', expect_stderr=True)
    assert 'initools (CURRENT: 0.2 LATEST: %s)' % initools_ver in result.stdout
    assert 'simplejson (CURRENT: 2.0.0 LATEST: %s)' % simplejson_ver in result.stdout
