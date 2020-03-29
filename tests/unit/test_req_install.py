import os
import tempfile
import pytest

from pip._vendor.packaging.requirements import Requirement
from pip._internal.cache import WheelCache
from pip._internal.exceptions import InstallationError
from pip._internal.models.format_control import FormatControl
from pip._internal.req.constructors import (
    install_req_from_line,
    install_req_from_req_string,
)
from pip._internal.req.req_install import InstallRequirement
from pip._internal.models.link import Link
from tests.lib import path_to_url
from tests.lib.wheel import make_wheel


class TestInstallRequirementBuildDirectory(object):
    # no need to test symlinks on Windows
    @pytest.mark.skipif("sys.platform == 'win32'")
    def test_tmp_build_directory(self):
        # when req is None, we can produce a temporary directory
        # Make sure we're handling it correctly with real path.
        requirement = InstallRequirement(None, None)
        tmp_dir = tempfile.mkdtemp('-build', 'pip-')
        tmp_build_dir = requirement.ensure_build_location(
            tmp_dir, autodelete=False
        )
        assert (
            os.path.dirname(tmp_build_dir) ==
            os.path.realpath(os.path.dirname(tmp_dir))
        )
        # are we on a system where /tmp is a symlink
        if os.path.realpath(tmp_dir) != os.path.abspath(tmp_dir):
            assert os.path.dirname(tmp_build_dir) != os.path.dirname(tmp_dir)
        else:
            assert os.path.dirname(tmp_build_dir) == os.path.dirname(tmp_dir)
        os.rmdir(tmp_dir)
        assert not os.path.exists(tmp_dir)

    def test_forward_slash_results_in_a_link(self, tmpdir):
        install_dir = tmpdir / "foo" / "bar"

        # Just create a file for letting the logic work
        setup_py_path = install_dir / "setup.py"
        os.makedirs(str(install_dir))
        with open(setup_py_path, 'w') as f:
            f.write('')

        requirement = install_req_from_line(
            str(install_dir).replace(os.sep, os.altsep or os.sep)
        )

        assert requirement.link is not None


class TestInstallRequirementFrom(object):

    def test_install_req_from_string_invalid_requirement(self):
        """
        Requirement strings that cannot be parsed by
        packaging.requirements.Requirement raise an InstallationError.
        """
        with pytest.raises(InstallationError) as excinfo:
            install_req_from_req_string("http:/this/is/invalid")

        assert str(excinfo.value) == (
            "Invalid requirement: 'http:/this/is/invalid'"
        )

    def test_install_req_from_string_without_comes_from(self):
        """
        Test to make sure that install_req_from_string succeeds
        when called with URL (PEP 508) but without comes_from.
        """
        # Test with a PEP 508 url install string:
        wheel_url = ("https://download.pytorch.org/whl/cu90/"
                     "torch-1.0.0-cp36-cp36m-win_amd64.whl")
        install_str = "torch@ " + wheel_url
        install_req = install_req_from_req_string(install_str)

        assert isinstance(install_req, InstallRequirement)
        assert install_req.link.url == wheel_url
        assert install_req.req.url == wheel_url
        assert install_req.comes_from is None
        assert install_req.is_wheel

    def test_install_req_from_string_with_comes_from_without_link(self):
        """
        Test to make sure that install_req_from_string succeeds
        when called with URL (PEP 508) and comes_from
        does not have a link.
        """
        # Test with a PEP 508 url install string:
        wheel_url = ("https://download.pytorch.org/whl/cu90/"
                     "torch-1.0.0-cp36-cp36m-win_amd64.whl")
        install_str = "torch@ " + wheel_url

        # Dummy numpy "comes_from" requirement without link:
        comes_from = InstallRequirement(
            Requirement("numpy>=1.15.0"), comes_from=None
        )

        # Attempt install from install string comes:
        install_req = install_req_from_req_string(
            install_str, comes_from=comes_from
        )

        assert isinstance(install_req, InstallRequirement)
        assert install_req.comes_from.link is None
        assert install_req.link.url == wheel_url
        assert install_req.req.url == wheel_url
        assert install_req.is_wheel


class TestInstallRequirementWheelCache(object):

    def test_able_to_register_when_get_from_wheel_cache(self, tmpdir):
        """
        This test to make sure that file is able to tell when a link is
        from the wheel cache.
        """

        format_control = FormatControl()
        wheel_path = path_to_url(make_wheel(
            name='pip',
            version='6.9.0'
        ).save_to_dir(tmpdir))

        wheel_cache = WheelCache(tmpdir, format_control)

        install_req = install_req_from_req_string(
            req_string='pip',
            wheel_cache=wheel_cache
        )
        install_req.link = Link(wheel_path)

        assert install_req.from_wheel_cache

    def test_able_to_register_when_not_get_from_wheel_cache(self, tmpdir):
        """
        This test to make sure that file is able to tell when a link is not
        from the wheel cache.
        """

        wheel_path = path_to_url(make_wheel(
            name='pip',
            version='6.9.0'
        ).save_to_dir(tmpdir))

        format_control = FormatControl()

        wheel_cache = WheelCache(tmpdir + '/wheel_cache', format_control)

        install_req = install_req_from_req_string(
            req_string='pip',
            wheel_cache=wheel_cache
        )

        install_req.link = Link(wheel_path)

        assert not install_req.from_wheel_cache
