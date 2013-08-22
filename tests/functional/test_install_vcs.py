from tests.lib import (reset_env,
                      _create_test_package, _change_test_package_version)
from tests.lib.local_repos import local_checkout


def test_install_editable_from_git_with_https():
    """
    Test cloning from Git with https.
    """
    script = reset_env()
    result = script.pip('install', '-e',
                     '%s#egg=pip-test-package' %
                     local_checkout('git+https://github.com/pypa/pip-test-package.git'),
                     expect_error=True)
    result.assert_installed('pip-test-package', with_files=['.git'])


def test_git_with_sha1_revisions():
    """
    Git backend should be able to install from SHA1 revisions
    """
    script = reset_env()
    version_pkg_path = _create_test_package(script)
    _change_test_package_version(script, version_pkg_path)
    sha1 = script.run('git', 'rev-parse', 'HEAD~1', cwd=version_pkg_path).stdout.strip()
    script.pip('install', '-e', '%s@%s#egg=version_pkg' % ('git+file://' + version_pkg_path.abspath.replace('\\', '/'), sha1))
    version = script.run('version_pkg')
    assert '0.1' in version.stdout, version.stdout


def test_git_with_branch_name_as_revision():
    """
    Git backend should be able to install from branch names
    """
    script = reset_env()
    version_pkg_path = _create_test_package(script)
    script.run('git', 'checkout', '-b', 'test_branch', expect_stderr=True, cwd=version_pkg_path)
    _change_test_package_version(script, version_pkg_path)
    script.pip('install', '-e', '%s@test_branch#egg=version_pkg' % ('git+file://' + version_pkg_path.abspath.replace('\\', '/')))
    version = script.run('version_pkg')
    assert 'some different version' in version.stdout


def test_git_with_tag_name_as_revision():
    """
    Git backend should be able to install from tag names
    """
    script = reset_env()
    version_pkg_path = _create_test_package(script)
    script.run('git', 'tag', 'test_tag', expect_stderr=True, cwd=version_pkg_path)
    _change_test_package_version(script, version_pkg_path)
    script.pip('install', '-e', '%s@test_tag#egg=version_pkg' % ('git+file://' + version_pkg_path.abspath.replace('\\', '/')))
    version = script.run('version_pkg')
    assert '0.1' in version.stdout


def test_git_with_tag_name_and_update():
    """
    Test cloning a git repository and updating to a different version.
    """
    script = reset_env()
    result = script.pip('install', '-e', '%s#egg=pip-test-package' %
                     local_checkout('git+http://github.com/pypa/pip-test-package.git'),
                     expect_error=True)
    result.assert_installed('pip-test-package', with_files=['.git'])
    result = script.pip('install', '--global-option=--version', '-e',
                     '%s@0.1.2#egg=pip-test-package' %
                     local_checkout('git+http://github.com/pypa/pip-test-package.git'),
                     expect_error=True)
    assert '0.1.2' in result.stdout


def test_git_branch_should_not_be_changed():
    """
    Editable installations should not change branch
    related to issue #32 and #161
    """
    script = reset_env()
    script.pip('install', '-e', '%s#egg=pip-test-package' %
                local_checkout('git+http://github.com/pypa/pip-test-package.git'),
                expect_error=True)
    source_dir = script.venv_path/'src'/'pip-test-package'
    result = script.run('git', 'branch', cwd=source_dir)
    assert '* master' in result.stdout, result.stdout


def test_git_with_non_editable_unpacking():
    """
    Test cloning a git repository from a non-editable URL with a given tag.
    """
    script = reset_env()
    result = script.pip('install', '--global-option=--version', local_checkout(
                     'git+http://github.com/pypa/pip-test-package.git@0.1.2#egg=pip-test-package'
                     ), expect_error=True)
    assert '0.1.2' in result.stdout


def test_git_with_editable_where_egg_contains_dev_string():
    """
    Test cloning a git repository from an editable url which contains "dev" string
    """
    script = reset_env()
    result = script.pip('install', '-e', '%s#egg=django-devserver' %
                     local_checkout('git+git://github.com/dcramer/django-devserver.git'))
    result.assert_installed('django-devserver', with_files=['.git'])


def test_git_with_non_editable_where_egg_contains_dev_string():
    """
    Test cloning a git repository from a non-editable url which contains "dev" string
    """
    script = reset_env()
    result = script.pip('install', '%s#egg=django-devserver' %
                     local_checkout('git+git://github.com/dcramer/django-devserver.git'))
    devserver_folder = script.site_packages/'devserver'
    assert devserver_folder in result.files_created, str(result)


def test_git_with_ambiguous_revs():
    """
    Test git with two "names" (tag/branch) pointing to the same commit
    """
    script = reset_env()
    version_pkg_path = _create_test_package(script)
    package_url = 'git+file://%s@0.1#egg=version_pkg' % (version_pkg_path.abspath.replace('\\', '/'))
    script.run('git', 'tag', '0.1', cwd=version_pkg_path)
    result = script.pip('install', '-e', package_url)
    assert 'Could not find a tag or branch' not in result.stdout
    # it is 'version-pkg' instead of 'version_pkg' because
    # egg-link name is version-pkg.egg-link because it is a single .py module
    result.assert_installed('version-pkg', with_files=['.git'])


def test_git_works_with_editable_non_origin_repo():
    # set up, create a git repo and install it as editable from a local directory path
    script = reset_env()
    version_pkg_path = _create_test_package(script)
    script.pip('install', '-e', version_pkg_path.abspath)

    # 'freeze'ing this should not fall over, but should result in stderr output warning
    result = script.pip('freeze', expect_stderr=True)
    assert "Error when trying to get requirement" in result.stderr
    assert "Could not determine repository location" in result.stdout
    assert "version-pkg==0.1" in result.stdout
