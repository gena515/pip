from __future__ import with_statement

import textwrap
import os
import sys
from os.path import join, abspath, normpath
from tempfile import mkdtemp
from mock import patch
from tests.lib import tests_data, reset_env, assert_all_changes, pyversion
from tests.lib.local_repos import local_repo, local_checkout

from pip.util import rmtree


def test_simple_uninstall():
    """
    Test simple install and uninstall.

    """
    script = reset_env()
    result = script.pip('install', 'INITools==0.2')
    assert join(script.site_packages, 'initools') in result.files_created, sorted(result.files_created.keys())
    #the import forces the generation of __pycache__ if the version of python supports it
    script.run('python', '-c', "import initools")
    result2 = script.pip('uninstall', 'INITools', '-y')
    assert_all_changes(result, result2, [script.venv/'build', 'cache'])


def test_uninstall_with_scripts():
    """
    Uninstall an easy_installed package with scripts.

    """
    script = reset_env()
    result = script.run('easy_install', 'PyLogo', expect_stderr=True)
    easy_install_pth = script.site_packages/ 'easy-install.pth'
    pylogo = sys.platform == 'win32' and 'pylogo' or 'PyLogo'
    assert(pylogo in result.files_updated[easy_install_pth].bytes)
    result2 = script.pip('uninstall', 'pylogo', '-y', expect_error=True)
    assert_all_changes(result, result2, [script.venv/'build', 'cache'])


def test_uninstall_easy_install_after_import():
    """
    Uninstall an easy_installed package after it's been imported

    """
    script = reset_env()
    result = script.run('easy_install', 'INITools==0.2', expect_stderr=True)
    #the import forces the generation of __pycache__ if the version of python supports it
    script.run('python', '-c', "import initools")
    result2 = script.pip('uninstall', 'INITools', '-y')
    assert_all_changes(result, result2, [script.venv/'build', 'cache'])


def test_uninstall_namespace_package():
    """
    Uninstall a distribution with a namespace package without clobbering
    the namespace and everything in it.

    """
    script = reset_env()
    result = script.pip('install', 'pd.requires==0.0.3', expect_error=True)
    assert join(script.site_packages, 'pd') in result.files_created, sorted(result.files_created.keys())
    result2 = script.pip('uninstall', 'pd.find', '-y', expect_error=True)
    assert join(script.site_packages, 'pd') not in result2.files_deleted, sorted(result2.files_deleted.keys())
    assert join(script.site_packages, 'pd', 'find') in result2.files_deleted, sorted(result2.files_deleted.keys())


def test_uninstall_overlapping_package():
    """
    Uninstalling a distribution that adds modules to a pre-existing package
    should only remove those added modules, not the rest of the existing
    package.

    See: GitHub issue #355 (pip uninstall removes things it didn't install)
    """
    parent_pkg = abspath(join(tests_data, 'packages', 'parent-0.1.tar.gz'))
    child_pkg = abspath(join(tests_data, 'packages', 'child-0.1.tar.gz'))
    script = reset_env()
    result1 = script.pip('install', parent_pkg, expect_error=False)
    assert join(script.site_packages, 'parent') in result1.files_created, sorted(result1.files_created.keys())
    result2 = script.pip('install', child_pkg, expect_error=False)
    assert join(script.site_packages, 'child') in result2.files_created, sorted(result2.files_created.keys())
    assert normpath(join(script.site_packages, 'parent/plugins/child_plugin.py')) in result2.files_created, sorted(result2.files_created.keys())
    #the import forces the generation of __pycache__ if the version of python supports it
    script.run('python', '-c', "import parent.plugins.child_plugin, child")
    result3 = script.pip('uninstall', '-y', 'child', expect_error=False)
    assert join(script.site_packages, 'child') in result3.files_deleted, sorted(result3.files_created.keys())
    assert normpath(join(script.site_packages, 'parent/plugins/child_plugin.py')) in result3.files_deleted, sorted(result3.files_deleted.keys())
    assert join(script.site_packages, 'parent') not in result3.files_deleted, sorted(result3.files_deleted.keys())
    # Additional check: uninstalling 'child' should return things to the
    # previous state, without unintended side effects.
    assert_all_changes(result2, result3, [])


def test_uninstall_console_scripts():
    """
    Test uninstalling a package with more files (console_script entry points, extra directories).

    """
    script = reset_env()
    args = ['install']
    args.append('discover')
    result = script.pip(*args, **{"expect_error": True})
    assert script.bin/'discover'+script.exe in result.files_created, sorted(result.files_created.keys())
    result2 = script.pip('uninstall', 'discover', '-y', expect_error=True)
    assert_all_changes(result, result2, [script.venv/'build', 'cache'])


def test_uninstall_easy_installed_console_scripts():
    """
    Test uninstalling package with console_scripts that is easy_installed.

    """
    script = reset_env()
    args = ['easy_install']
    args.append('discover')
    result = script.run(*args, **{"expect_stderr": True})
    assert script.bin/'discover'+script.exe in result.files_created, sorted(result.files_created.keys())
    result2 = script.pip('uninstall', 'discover', '-y')
    assert_all_changes(result, result2, [script.venv/'build', 'cache'])


def test_uninstall_editable_from_svn():
    """
    Test uninstalling an editable installation from svn.

    """
    script = reset_env()
    result = script.pip('install', '-e', '%s#egg=initools-dev' %
                     local_checkout('svn+http://svn.colorstudy.com/INITools/trunk'))
    result.assert_installed('INITools')
    result2 = script.pip('uninstall', '-y', 'initools')
    assert (script.venv/'src'/'initools' in result2.files_after), 'oh noes, pip deleted my sources!'
    assert_all_changes(result, result2, [script.venv/'src', script.venv/'build'])


def test_uninstall_editable_with_source_outside_venv():
    """
    Test uninstalling editable install from existing source outside the venv.

    """
    try:
        temp = mkdtemp()
        tmpdir = join(temp, 'virtualenv')
        _test_uninstall_editable_with_source_outside_venv(tmpdir)
    finally:
        rmtree(temp)


def _test_uninstall_editable_with_source_outside_venv(tmpdir):
    script = reset_env()
    result = script.run('git', 'clone', local_repo('git+git://github.com/pypa/virtualenv'), tmpdir)
    result2 = script.pip('install', '-e', tmpdir)
    assert (join(script.site_packages, 'virtualenv.egg-link') in result2.files_created), list(result2.files_created.keys())
    result3 = script.pip('uninstall', '-y', 'virtualenv', expect_error=True)
    assert_all_changes(result, result3, [script.venv/'build'])


def test_uninstall_from_reqs_file():
    """
    Test uninstall from a requirements file.

    """
    script = reset_env()
    script.scratch_path.join("test-req.txt").write(textwrap.dedent("""\
        -e %s#egg=initools-dev
        # and something else to test out:
        PyLogo<0.4
        """ % local_checkout('svn+http://svn.colorstudy.com/INITools/trunk')))
    result = script.pip('install', '-r', 'test-req.txt')
    script.scratch_path.join("test-req.txt").write(textwrap.dedent("""\
        # -f, -i, and --extra-index-url should all be ignored by uninstall
        -f http://www.example.com
        -i http://www.example.com
        --extra-index-url http://www.example.com

        -e %s#egg=initools-dev
        # and something else to test out:
        PyLogo<0.4
        """ % local_checkout('svn+http://svn.colorstudy.com/INITools/trunk')))
    result2 = script.pip('uninstall', '-r', 'test-req.txt', '-y')
    assert_all_changes(
        result, result2, [script.venv/'build', script.venv/'src', script.scratch/'test-req.txt'])


def test_uninstall_as_egg():
    """
    Test uninstall package installed as egg.

    """
    script = reset_env()
    to_install = abspath(join(tests_data, 'packages', 'FSPkg'))
    result = script.pip('install', to_install, '--egg', expect_error=False)
    fspkg_folder = script.site_packages/'fspkg'
    egg_folder = script.site_packages/'FSPkg-0.1dev-py%s.egg' % pyversion
    assert fspkg_folder not in result.files_created, str(result.stdout)
    assert egg_folder in result.files_created, str(result)

    result2 = script.pip('uninstall', 'FSPkg', '-y', expect_error=True)
    assert_all_changes(result, result2, [script.venv/'build', 'cache'])


@patch('pip.req.logger')
def test_uninstallpathset_no_paths(mock_logger):
    """
    Test UninstallPathSet logs notification when there are no paths to uninstall

    """
    from pip.req import UninstallPathSet
    from pkg_resources import get_distribution
    test_dist = get_distribution('pip')
    # ensure that the distribution is "local"
    with patch("pip.req.dist_is_local") as mock_dist_is_local:
        mock_dist_is_local.return_value = True
        uninstall_set = UninstallPathSet(test_dist)
        uninstall_set.remove() #with no files added to set
    mock_logger.notify.assert_any_call("Can't uninstall 'pip'. No files were found to uninstall.")


@patch('pip.req.logger')
def test_uninstallpathset_non_local(mock_logger):
    """
    Test UninstallPathSet logs notification and returns (with no exception) when dist is non-local

    """
    nonlocal_path = os.path.abspath("/nonlocal")
    from pip.req import UninstallPathSet
    from pkg_resources import get_distribution
    test_dist = get_distribution('pip')
    test_dist.location = nonlocal_path
    # ensure that the distribution is "non-local"
    # setting location isn't enough, due to egg-link file checking for
    # develop-installs
    with patch("pip.req.dist_is_local") as mock_dist_is_local:
        mock_dist_is_local.return_value = False
        uninstall_set = UninstallPathSet(test_dist)
        uninstall_set.remove() #with no files added to set; which is the case when trying to remove non-local dists
    mock_logger.notify.assert_any_call("Not uninstalling pip at %s, outside environment %s" % (nonlocal_path, sys.prefix)), mock_logger.notify.mock_calls
