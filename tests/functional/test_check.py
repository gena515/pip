from tests.lib import create_test_package_with_setup


def matches_expected_lines(string, expected_lines):
    # Ignore empty lines
    output_lines = set(filter(None, string.splitlines()))
    # Match regardless of order
    return set(output_lines) == set(expected_lines)


def test_check_clean(script):
    """On a clean environment, check should print a helpful message.

    """
    result = script.pip('check')

    expected_lines = (
        "No broken requirements found.",
    )
    assert matches_expected_lines(result.stdout, expected_lines)


def test_check_missing_dependency(script):
    # Setup a small project
    pkga_path = create_test_package_with_setup(
        script,
        name='pkga', version='1.0', install_requires=['missing==0.1'],
    )
    # Let's install pkga without its dependency
    res = script.pip('install', '--no-index', pkga_path, '--no-deps')
    assert "Successfully installed pkga-1.0" in res.stdout, str(res)

    result = script.pip('check', expect_error=True)

    expected_lines = (
        "pkga 1.0 requires missing==0.1, which is not installed.",
    )
    assert matches_expected_lines(result.stdout, expected_lines)
    assert result.returncode == 1


def test_check_broken_dependency(script):
    # Setup pkga depending on pkgb>=1.0
    pkga_path = create_test_package_with_setup(
        script,
        name='pkga', version='1.0', install_requires=['broken>=1.0'],
    )
    # Let's install pkga without its dependency
    res = script.pip('install', '--no-index', pkga_path, '--no-deps')
    assert "Successfully installed pkga-1.0" in res.stdout, str(res)

    # Setup broken==0.1
    broken_path = create_test_package_with_setup(
        script,
        name='broken', version='0.1',
    )
    # Let's install broken==0.1
    res = script.pip(
        'install', '--no-index', broken_path, '--no-warn-conflicts',
    )
    assert "Successfully installed broken-0.1" in res.stdout, str(res)

    result = script.pip('check', expect_error=True)

    expected_lines = (
        "pkga 1.0 has requirement broken>=1.0, but you have broken 0.1.",
    )
    assert matches_expected_lines(result.stdout, expected_lines)
    assert result.returncode == 1


def test_check_broken_dependency_and_missing_dependency(script):
    pkga_path = create_test_package_with_setup(
        script,
        name='pkga', version='1.0', install_requires=['broken>=1.0'],
    )
    # Let's install pkga without its dependency
    res = script.pip('install', '--no-index', pkga_path, '--no-deps')
    assert "Successfully installed pkga-1.0" in res.stdout, str(res)

    # Setup broken==0.1
    broken_path = create_test_package_with_setup(
        script,
        name='broken', version='0.1', install_requires=['missing'],
    )
    # Let's install broken==0.1
    res = script.pip('install', '--no-index', broken_path, '--no-deps')
    assert "Successfully installed broken-0.1" in res.stdout, str(res)

    result = script.pip('check', expect_error=True)

    expected_lines = (
        "broken 0.1 requires missing, which is not installed.",
        "pkga 1.0 has requirement broken>=1.0, but you have broken 0.1."
    )

    assert matches_expected_lines(result.stdout, expected_lines)
    assert result.returncode == 1


def test_check_complicated_name_missing(script):
    package_a_path = create_test_package_with_setup(
        script,
        name='package_A', version='1.0',
        install_requires=['Dependency-B>=1.0'],
    )

    # Without dependency
    result = script.pip('install', '--no-index', package_a_path, '--no-deps')
    assert "Successfully installed package-A-1.0" in result.stdout, str(result)

    result = script.pip('check', expect_error=True)
    expected_lines = (
        "package-a 1.0 requires Dependency-B>=1.0, which is not installed.",
    )
    assert matches_expected_lines(result.stdout, expected_lines)


def test_check_complicated_name_broken(script):
    package_a_path = create_test_package_with_setup(
        script,
        name='package_A', version='1.0',
        install_requires=['Dependency-B>=1.0'],
    )
    dependency_b_path_incompatible = create_test_package_with_setup(
        script,
        name='dependency-b', version='0.1',
    )

    # With broken dependency
    result = script.pip('install', '--no-index', package_a_path, '--no-deps')
    assert "Successfully installed package-A-1.0" in result.stdout, str(result)

    result = script.pip(
        'install', '--no-index', dependency_b_path_incompatible, '--no-deps',
    )
    assert "Successfully installed dependency-b-0.1" in result.stdout

    result = script.pip('check', expect_error=True)
    expected_lines = (
        "package-a 1.0 has requirement Dependency-B>=1.0, but you have "
        "dependency-b 0.1.",
    )
    assert matches_expected_lines(result.stdout, expected_lines)


def test_check_complicated_name_clean(script):
    package_a_path = create_test_package_with_setup(
        script,
        name='package_A', version='1.0',
        install_requires=['Dependency-B>=1.0'],
    )
    dependency_b_path = create_test_package_with_setup(
        script,
        name='dependency-b', version='1.0',
    )

    result = script.pip('install', '--no-index', package_a_path, '--no-deps')
    assert "Successfully installed package-A-1.0" in result.stdout, str(result)

    result = script.pip(
        'install', '--no-index', dependency_b_path, '--no-deps',
    )
    assert "Successfully installed dependency-b-1.0" in result.stdout

    result = script.pip('check', expect_error=True)
    expected_lines = (
        "No broken requirements found.",
    )
    assert matches_expected_lines(result.stdout, expected_lines)

