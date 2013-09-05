"""'pip wheel' tests"""
import os
import sys
import textwrap

from os.path import exists

from pip import wheel
from pip.download import path_to_url as path_to_url_d
from pip.locations import write_delete_marker_file
from pip.status_codes import PREVIOUS_BUILD_DIR_ERROR
from tests.lib import pyversion_nodot, path_to_url


def test_pip_wheel_fails_without_wheel(script, data):
    """
    Test 'pip wheel' fails without wheel
    """
    result = script.pip('wheel', '--no-index', '-f', data.find_links, 'simple==3.0', expect_error=True)
    assert "'pip wheel' requires bdist_wheel" in result.stdout


def test_pip_wheel_success(script, data):
    """
    Test 'pip wheel' success.
    """
    script.pip_install_local('wheel')
    result = script.pip('wheel', '--no-index', '-f', data.find_links, 'simple==3.0')
    wheel_file_name = 'simple-3.0-py%s-none-any.whl' % pyversion_nodot
    wheel_file_path = script.scratch/'wheelhouse'/wheel_file_name
    assert wheel_file_path in result.files_created, result.stdout
    assert "Successfully built simple" in result.stdout, result.stdout


def test_pip_wheel_fail(script, data):
    """
    Test 'pip wheel' failure.
    """
    script.pip_install_local('wheel')
    result = script.pip('wheel', '--no-index', '-f', data.find_links, 'wheelbroken==0.1')
    wheel_file_name = 'wheelbroken-0.1-py%s-none-any.whl' % pyversion_nodot
    wheel_file_path = script.scratch/'wheelhouse'/wheel_file_name
    assert wheel_file_path not in result.files_created, (wheel_file_path, result.files_created)
    assert "FakeError" in result.stdout, result.stdout
    assert "Failed to build wheelbroken" in result.stdout, result.stdout


def test_pip_wheel_ignore_wheels_editables(script, data):
    """
    Test 'pip wheel' ignores editables and *.whl files in requirements
    """
    script.pip_install_local('wheel')

    local_wheel = '%s/simple.dist-0.1-py2.py3-none-any.whl' % data.find_links
    local_editable = data.packages.join("FSPkg")
    script.scratch_path.join("reqs.txt").write(textwrap.dedent("""\
        %s
        -e %s
        simple
        """ % (local_wheel, local_editable)))
    result = script.pip('wheel', '--no-index', '-f', data.find_links, '-r', script.scratch_path / 'reqs.txt')
    wheel_file_name = 'simple-3.0-py%s-none-any.whl' % pyversion_nodot
    wheel_file_path = script.scratch/'wheelhouse'/wheel_file_name
    assert wheel_file_path in result.files_created, (wheel_file_path, result.files_created)
    assert "Successfully built simple" in result.stdout, result.stdout
    assert "Failed to build" not in result.stdout, result.stdout
    assert "ignoring %s" % local_wheel in result.stdout
    ignore_editable = "ignoring %s" % path_to_url(local_editable)
    #TODO: understand this divergence
    if sys.platform == 'win32':
        ignore_editable = "ignoring %s" % path_to_url_d(local_editable)
    assert ignore_editable in result.stdout, result.stdout


def test_no_clean_option_blocks_cleaning_after_wheel(script, data):
    """
    Test --no-clean option blocks cleaning after wheel build
    """
    script.pip_install_local('wheel')
    result = script.pip('wheel', '--no-clean', '--no-index', '--find-links=%s' % data.find_links, 'simple')
    build = script.venv_path/'build'/'simple'
    assert exists(build), "build/simple should still exist %s" % str(result)


def test_pip_wheel_source_deps(script, data):
    """
    Test 'pip wheel --use-wheel' finds and builds source archive dependencies of wheels
    """
    # 'requires_source' is a wheel that depends on the 'source' project
    script.pip_install_local('wheel')
    result = script.pip('wheel', '--use-wheel', '--no-index', '-f', data.find_links, 'requires_source')
    wheel_file_name = 'source-1.0-py%s-none-any.whl' % pyversion_nodot
    wheel_file_path = script.scratch/'wheelhouse'/wheel_file_name
    assert wheel_file_path in result.files_created, result.stdout
    assert "Successfully built source" in result.stdout, result.stdout


def test_pip_wheel_fail_cause_of_previous_build_dir(script, data):
    """Test when 'pip wheel' tries to install a package that has a previous build directory"""

    script.pip_install_local('wheel')

    # Given that I have a previous build dir of the `simple` package
    build = script.venv_path / 'build' / 'simple'
    os.makedirs(build)
    write_delete_marker_file(script.venv_path / 'build')
    build.join('setup.py').write('#')

    # When I call pip trying to install things again
    result = script.pip('wheel', '--no-index', '--find-links=%s' % data.find_links, 'simple==3.0', expect_error=True)

    # Then I see that the error code is the right one
    assert result.returncode == PREVIOUS_BUILD_DIR_ERROR
