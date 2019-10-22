import errno

import pytest
from mock import Mock, call, patch

from pip._internal.commands.install import (
    build_wheels,
    create_env_error_message,
)


class TestWheelCache:

    def check_build_wheels(
        self,
        pep517_requirements,
        legacy_requirements,
    ):
        """
        Return: (mock_calls, return_value).
        """
        def build(reqs, **kwargs):
            # Fail the first requirement.
            return [reqs[0]]

        builder = Mock()
        builder.build.side_effect = build

        build_failures = build_wheels(
            builder=builder,
            pep517_requirements=pep517_requirements,
            legacy_requirements=legacy_requirements,
        )

        return (builder.build.mock_calls, build_failures)

    @patch('pip._internal.commands.install.is_wheel_installed')
    def test_build_wheels__wheel_installed(self, is_wheel_installed):
        is_wheel_installed.return_value = True

        mock_calls, build_failures = self.check_build_wheels(
            pep517_requirements=['a', 'b'],
            legacy_requirements=['c', 'd'],
        )

        # Legacy requirements were built.
        assert mock_calls == [
            call(['a', 'b'], should_unpack=True),
            call(['c', 'd'], should_unpack=True),
        ]

        # Legacy build failures are not included in the return value.
        assert build_failures == ['a']

    @patch('pip._internal.commands.install.is_wheel_installed')
    def test_build_wheels__wheel_not_installed(self, is_wheel_installed):
        is_wheel_installed.return_value = False

        mock_calls, build_failures = self.check_build_wheels(
            pep517_requirements=['a', 'b'],
            legacy_requirements=['c', 'd'],
        )

        # Legacy requirements were not built.
        assert mock_calls == [
            call(['a', 'b'], should_unpack=True),
        ]

        assert build_failures == ['a']


def error_creation_helper(with_errno=False):
    env_error = EnvironmentError("No file permission")
    if with_errno:
        env_error.errno = errno.EACCES
    return env_error


@pytest.mark.parametrize('error, show_traceback, using_user_site, expected', [
    # show_traceback = True, using_user_site = True
    (error_creation_helper(), True, True, 'Could not install packages due to'
        ' an EnvironmentError.\n'),
    (error_creation_helper(True), True, True, 'Could not install packages due'
        ' to an EnvironmentError.\nCheck the permissions.\n'),
    # show_traceback = True, using_user_site = False
    (error_creation_helper(), True, False, 'Could not install packages due to'
        ' an EnvironmentError.\n'),
    (error_creation_helper(True), True, False, 'Could not install packages due'
        ' to an EnvironmentError.\nConsider using the `--user` option or check'
        ' the permissions.\n'),
    # show_traceback = False, using_user_site = True
    (error_creation_helper(), False, True, 'Could not install packages due to'
        ' an EnvironmentError: No file permission\n'),
    (error_creation_helper(True), False, True, 'Could not install packages due'
        ' to an EnvironmentError: No file permission\nCheck the'
        ' permissions.\n'),
    # show_traceback = False, using_user_site = False
    (error_creation_helper(), False, False, 'Could not install packages due to'
        ' an EnvironmentError: No file permission\n'),
    (error_creation_helper(True), False, False, 'Could not install packages'
        ' due to an EnvironmentError: No file permission\nConsider using the'
        ' `--user` option or check the permissions.\n'),
])
def test_create_env_error_message(
    error, show_traceback, using_user_site, expected
):
    msg = create_env_error_message(error, show_traceback, using_user_site)
    assert msg == expected
