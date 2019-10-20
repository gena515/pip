import os
import shutil
import sys
import tempfile
from functools import partial

import pytest
from mock import patch
from pip._vendor import pkg_resources
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import Requirement

from pip._internal.commands import create_command
from pip._internal.exceptions import (
    HashErrors,
    InstallationError,
    InvalidWheelFilename,
    PreviousBuildDirError,
)
from pip._internal.legacy_resolve import Resolver
from pip._internal.network.session import PipSession
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req import InstallRequirement, RequirementSet
from pip._internal.req.constructors import (
    _get_url_from_path,
    _looks_like_path,
    install_req_from_editable,
    install_req_from_line,
    install_req_from_req_string,
    parse_editable,
)
from pip._internal.req.req_file import process_line
from pip._internal.req.req_tracker import RequirementTracker
from pip._internal.utils.urls import path_to_url
from tests.lib import (
    DATA_DIR,
    assert_raises_regexp,
    make_test_finder,
    requirements_file,
)


def get_processed_req_from_line(line, fname='file', lineno=1):
    req = list(process_line(line, fname, lineno))[0]
    req.is_direct = True
    return req


class TestRequirementSet(object):
    """RequirementSet tests"""

    def setup(self):
        self.tempdir = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def _basic_resolver(self, finder):
        preparer = RequirementPreparer(
            build_dir=os.path.join(self.tempdir, 'build'),
            src_dir=os.path.join(self.tempdir, 'src'),
            download_dir=None,
            wheel_download_dir=None,
            progress_bar="on",
            build_isolation=True,
            req_tracker=RequirementTracker(),
        )
        make_install_req = partial(
            install_req_from_req_string,
            isolated=False,
            wheel_cache=None,
            use_pep517=None,
        )
        return Resolver(
            preparer=preparer,
            make_install_req=make_install_req,
            session=PipSession(), finder=finder,
            use_user_site=False, upgrade_strategy="to-satisfy-only",
            ignore_dependencies=False, ignore_installed=False,
            ignore_requires_python=False, force_reinstall=False,
        )

    def test_no_reuse_existing_build_dir(self, data):
        """Test prepare_files raise exception with previous build dir"""

        build_dir = os.path.join(self.tempdir, 'build', 'simple')
        os.makedirs(build_dir)
        with open(os.path.join(build_dir, "setup.py"), 'w'):
            pass
        reqset = RequirementSet()
        req = install_req_from_line('simple')
        req.is_direct = True
        reqset.add_requirement(req)
        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)
        assert_raises_regexp(
            PreviousBuildDirError,
            r"pip can't proceed with [\s\S]*%s[\s\S]*%s" %
            (req, build_dir.replace('\\', '\\\\')),
            resolver.resolve,
            reqset,
        )

    # TODO: Update test when Python 2.7 is dropped.
    def test_environment_marker_extras(self, data):
        """
        Test that the environment marker extras are used with
        non-wheel installs.
        """
        reqset = RequirementSet()
        req = install_req_from_editable(
            data.packages.joinpath("LocalEnvironMarker")
        )
        req.is_direct = True
        reqset.add_requirement(req)
        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)
        resolver.resolve(reqset)
        # This is hacky but does test both case in py2 and py3
        if sys.version_info[:2] == (2, 7):
            assert reqset.has_requirement('simple')
        else:
            assert not reqset.has_requirement('simple')

    @pytest.mark.network
    def test_missing_hash_checking(self):
        """Make sure prepare_files() raises an error when a requirement has no
        hash in implicit hash-checking mode.
        """
        reqset = RequirementSet()
        # No flags here. This tests that detection of later flags nonetheless
        # requires earlier packages to have hashes:
        reqset.add_requirement(get_processed_req_from_line(
            'blessings==1.0', lineno=1
        ))
        # This flag activates --require-hashes mode:
        reqset.add_requirement(get_processed_req_from_line(
            'tracefront==0.1 --hash=sha256:somehash', lineno=2,
        ))
        # This hash should be accepted because it came from the reqs file, not
        # from the internet:
        reqset.add_requirement(get_processed_req_from_line(
            'https://files.pythonhosted.org/packages/source/m/more-itertools/'
            'more-itertools-1.0.tar.gz#md5=b21850c3cfa7efbb70fd662ab5413bdd',
            lineno=3,
        ))
        # The error text should list this as a URL and not `peep==3.1.1`:
        reqset.add_requirement(get_processed_req_from_line(
            'https://files.pythonhosted.org/'
            'packages/source/p/peep/peep-3.1.1.tar.gz',
            lineno=4,
        ))
        finder = make_test_finder(index_urls=['https://pypi.org/simple/'])
        resolver = self._basic_resolver(finder)
        assert_raises_regexp(
            HashErrors,
            r'Hashes are required in --require-hashes mode, but they are '
            r'missing .*\n'
            r'    https://files\.pythonhosted\.org/packages/source/p/peep/peep'
            r'-3\.1\.1\.tar\.gz --hash=sha256:[0-9a-f]+\n'
            r'    blessings==1.0 --hash=sha256:[0-9a-f]+\n'
            r'THESE PACKAGES DO NOT MATCH THE HASHES.*\n'
            r'    tracefront==0.1 .*:\n'
            r'        Expected sha256 somehash\n'
            r'             Got        [0-9a-f]+$',
            resolver.resolve,
            reqset
        )

    def test_missing_hash_with_require_hashes(self, data):
        """Setting --require-hashes explicitly should raise errors if hashes
        are missing.
        """
        reqset = RequirementSet(require_hashes=True)
        reqset.add_requirement(get_processed_req_from_line(
            'simple==1.0', lineno=1
        ))

        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)

        assert_raises_regexp(
            HashErrors,
            r'Hashes are required in --require-hashes mode, but they are '
            r'missing .*\n'
            r'    simple==1.0 --hash=sha256:393043e672415891885c9a2a0929b1af95'
            r'fb866d6ca016b42d2e6ce53619b653$',
            resolver.resolve,
            reqset
        )

    def test_missing_hash_with_require_hashes_in_reqs_file(self, data, tmpdir):
        """--require-hashes in a requirements file should make its way to the
        RequirementSet.
        """
        req_set = RequirementSet(require_hashes=False)
        finder = make_test_finder(find_links=[data.find_links])
        session = finder._link_collector.session
        command = create_command('install')
        with requirements_file('--require-hashes', tmpdir) as reqs_file:
            options, args = command.parse_args(['-r', reqs_file])
            command.populate_requirement_set(
                req_set, args, options, finder, session, wheel_cache=None,
            )
        assert req_set.require_hashes

    def test_unsupported_hashes(self, data):
        """VCS and dir links should raise errors when --require-hashes is
        on.

        In addition, complaints about the type of requirement (VCS or dir)
        should trump the presence or absence of a hash.

        """
        reqset = RequirementSet(require_hashes=True)
        reqset.add_requirement(get_processed_req_from_line(
            'git+git://github.com/pypa/pip-test-package --hash=sha256:123',
            lineno=1,
        ))
        dir_path = data.packages.joinpath('FSPkg')
        reqset.add_requirement(get_processed_req_from_line(
            'file://%s' % (dir_path,),
            lineno=2,
        ))
        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)
        sep = os.path.sep
        if sep == '\\':
            sep = '\\\\'  # This needs to be escaped for the regex
        assert_raises_regexp(
            HashErrors,
            r"Can't verify hashes for these requirements because we don't "
            r"have a way to hash version control repositories:\n"
            r"    git\+git://github\.com/pypa/pip-test-package \(from -r file "
            r"\(line 1\)\)\n"
            r"Can't verify hashes for these file:// requirements because they "
            r"point to directories:\n"
            r"    file://.*{sep}data{sep}packages{sep}FSPkg "
            r"\(from -r file \(line 2\)\)".format(sep=sep),
            resolver.resolve,
            reqset)

    def test_unpinned_hash_checking(self, data):
        """Make sure prepare_files() raises an error when a requirement is not
        version-pinned in hash-checking mode.
        """
        reqset = RequirementSet()
        # Test that there must be exactly 1 specifier:
        reqset.add_requirement(get_processed_req_from_line(
            'simple --hash=sha256:a90427ae31f5d1d0d7ec06ee97d9fcf2d0fc9a786985'
            '250c1c83fd68df5911dd', lineno=1,
        ))
        # Test that the operator must be ==:
        reqset.add_requirement(get_processed_req_from_line(
            'simple2>1.0 --hash=sha256:3ad45e1e9aa48b4462af0'
            '123f6a7e44a9115db1ef945d4d92c123dfe21815a06',
            lineno=2,
        ))
        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)
        assert_raises_regexp(
            HashErrors,
            # Make sure all failing requirements are listed:
            r'versions pinned with ==. These do not:\n'
            r'    simple .* \(from -r file \(line 1\)\)\n'
            r'    simple2>1.0 .* \(from -r file \(line 2\)\)',
            resolver.resolve,
            reqset)

    def test_hash_mismatch(self, data):
        """A hash mismatch should raise an error."""
        file_url = path_to_url(
            (data.packages / 'simple-1.0.tar.gz').resolve())
        reqset = RequirementSet(require_hashes=True)
        reqset.add_requirement(get_processed_req_from_line(
            '%s --hash=sha256:badbad' % file_url, lineno=1,
        ))
        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)
        assert_raises_regexp(
            HashErrors,
            r'THESE PACKAGES DO NOT MATCH THE HASHES.*\n'
            r'    file:///.*/data/packages/simple-1\.0\.tar\.gz .*:\n'
            r'        Expected sha256 badbad\n'
            r'             Got        393043e672415891885c9a2a0929b1af95fb866d'
            r'6ca016b42d2e6ce53619b653$',
            resolver.resolve,
            reqset)

    def test_unhashed_deps_on_require_hashes(self, data):
        """Make sure unhashed, unpinned, or otherwise unrepeatable
        dependencies get complained about when --require-hashes is on."""
        reqset = RequirementSet()
        finder = make_test_finder(find_links=[data.find_links])
        resolver = self._basic_resolver(finder)
        reqset.add_requirement(get_processed_req_from_line(
            'TopoRequires2==0.0.1 '  # requires TopoRequires
            '--hash=sha256:eaf9a01242c9f2f42cf2bd82a6a848cd'
            'e3591d14f7896bdbefcf48543720c970',
            lineno=1
        ))
        assert_raises_regexp(
            HashErrors,
            r'In --require-hashes mode, all requirements must have their '
            r'versions pinned.*\n'
            r'    TopoRequires from .*$',
            resolver.resolve,
            reqset)

    def test_hashed_deps_on_require_hashes(self):
        """Make sure hashed dependencies get installed when --require-hashes
        is on.

        (We actually just check that no "not all dependencies are hashed!"
        error gets raised while preparing; there is no reason to expect
        installation to then fail, as the code paths are the same as ever.)

        """
        reqset = RequirementSet()
        reqset.add_requirement(get_processed_req_from_line(
            'TopoRequires2==0.0.1 '  # requires TopoRequires
            '--hash=sha256:eaf9a01242c9f2f42cf2bd82a6a848cd'
            'e3591d14f7896bdbefcf48543720c970',
            lineno=1
        ))
        reqset.add_requirement(get_processed_req_from_line(
            'TopoRequires==0.0.1 '
            '--hash=sha256:d6dd1e22e60df512fdcf3640ced3039b3b02a56ab2cee81ebcb'
            '3d0a6d4e8bfa6',
            lineno=2
        ))


class TestInstallRequirement(object):
    def setup(self):
        self.tempdir = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_url_with_query(self):
        """InstallRequirement should strip the fragment, but not the query."""
        url = 'http://foo.com/?p=bar.git;a=snapshot;h=v0.1;sf=tgz'
        fragment = '#egg=bar'
        req = install_req_from_line(url + fragment)
        assert req.link.url == url + fragment, req.link

    def test_pep440_wheel_link_requirement(self):
        url = 'https://whatever.com/test-0.4-py2.py3-bogus-any.whl'
        line = 'test @ https://whatever.com/test-0.4-py2.py3-bogus-any.whl'
        req = install_req_from_line(line)
        parts = str(req.req).split('@', 1)
        assert len(parts) == 2
        assert parts[0].strip() == 'test'
        assert parts[1].strip() == url

    def test_pep440_url_link_requirement(self):
        url = 'git+http://foo.com@ref#egg=foo'
        line = 'foo @ git+http://foo.com@ref#egg=foo'
        req = install_req_from_line(line)
        parts = str(req.req).split('@', 1)
        assert len(parts) == 2
        assert parts[0].strip() == 'foo'
        assert parts[1].strip() == url

    def test_url_with_authentication_link_requirement(self):
        url = 'https://what@whatever.com/test-0.4-py2.py3-bogus-any.whl'
        line = 'https://what@whatever.com/test-0.4-py2.py3-bogus-any.whl'
        req = install_req_from_line(line)
        assert req.link is not None
        assert req.link.is_wheel
        assert req.link.scheme == "https"
        assert req.link.url == url

    def test_unsupported_wheel_link_requirement_raises(self):
        reqset = RequirementSet()
        req = install_req_from_line(
            'https://whatever.com/peppercorn-0.4-py2.py3-bogus-any.whl',
        )
        assert req.link is not None
        assert req.link.is_wheel
        assert req.link.scheme == "https"

        with pytest.raises(InstallationError):
            reqset.add_requirement(req)

    def test_unsupported_wheel_local_file_requirement_raises(self, data):
        reqset = RequirementSet()
        req = install_req_from_line(
            data.packages.joinpath('simple.dist-0.1-py1-none-invalid.whl'),
        )
        assert req.link is not None
        assert req.link.is_wheel
        assert req.link.scheme == "file"

        with pytest.raises(InstallationError):
            reqset.add_requirement(req)

    def test_installed_version_not_installed(self):
        req = install_req_from_line('simple-0.1-py2.py3-none-any.whl')
        assert req.installed_version is None

    def test_str(self):
        req = install_req_from_line('simple==0.1')
        assert str(req) == 'simple==0.1'

    def test_repr(self):
        req = install_req_from_line('simple==0.1')
        assert repr(req) == (
            '<InstallRequirement object: simple==0.1 editable=False>'
        )

    def test_invalid_wheel_requirement_raises(self):
        with pytest.raises(InvalidWheelFilename):
            install_req_from_line('invalid.whl')

    def test_wheel_requirement_sets_req_attribute(self):
        req = install_req_from_line('simple-0.1-py2.py3-none-any.whl')
        assert isinstance(req.req, Requirement)
        assert str(req.req) == 'simple==0.1'

    def test_url_preserved_line_req(self):
        """Confirm the url is preserved in a non-editable requirement"""
        url = 'git+http://foo.com@ref#egg=foo'
        req = install_req_from_line(url)
        assert req.link.url == url

    def test_url_preserved_editable_req(self):
        """Confirm the url is preserved in a editable requirement"""
        url = 'git+http://foo.com@ref#egg=foo'
        req = install_req_from_editable(url)
        assert req.link.url == url

    @pytest.mark.parametrize('path', (
        '/path/to/foo.egg-info'.replace('/', os.path.sep),
        # Tests issue fixed by https://github.com/pypa/pip/pull/2530
        '/path/to/foo.egg-info/'.replace('/', os.path.sep),
    ))
    def test_get_dist(self, path):
        req = install_req_from_line('foo')
        req.metadata_directory = path
        dist = req.get_dist()
        assert isinstance(dist, pkg_resources.Distribution)
        assert dist.project_name == 'foo'
        assert dist.location == '/path/to'.replace('/', os.path.sep)

    def test_markers(self):
        for line in (
            # recommended syntax
            'mock3; python_version >= "3"',
            # with more spaces
            'mock3 ; python_version >= "3" ',
            # without spaces
            'mock3;python_version >= "3"',
        ):
            req = install_req_from_line(line)
            assert req.req.name == 'mock3'
            assert str(req.req.specifier) == ''
            assert str(req.markers) == 'python_version >= "3"'

    def test_markers_semicolon(self):
        # check that the markers can contain a semicolon
        req = install_req_from_line('semicolon; os_name == "a; b"')
        assert req.req.name == 'semicolon'
        assert str(req.req.specifier) == ''
        assert str(req.markers) == 'os_name == "a; b"'

    def test_markers_url(self):
        # test "URL; markers" syntax
        url = 'http://foo.com/?p=bar.git;a=snapshot;h=v0.1;sf=tgz'
        line = '%s; python_version >= "3"' % url
        req = install_req_from_line(line)
        assert req.link.url == url, req.url
        assert str(req.markers) == 'python_version >= "3"'

        # without space, markers are part of the URL
        url = 'http://foo.com/?p=bar.git;a=snapshot;h=v0.1;sf=tgz'
        line = '%s;python_version >= "3"' % url
        req = install_req_from_line(line)
        assert req.link.url == line, req.url
        assert req.markers is None

    def test_markers_match_from_line(self):
        # match
        for markers in (
            'python_version >= "1.0"',
            'sys_platform == %r' % sys.platform,
        ):
            line = 'name; ' + markers
            req = install_req_from_line(line)
            assert str(req.markers) == str(Marker(markers))
            assert req.match_markers()

        # don't match
        for markers in (
            'python_version >= "5.0"',
            'sys_platform != %r' % sys.platform,
        ):
            line = 'name; ' + markers
            req = install_req_from_line(line)
            assert str(req.markers) == str(Marker(markers))
            assert not req.match_markers()

    def test_markers_match(self):
        # match
        for markers in (
            'python_version >= "1.0"',
            'sys_platform == %r' % sys.platform,
        ):
            line = 'name; ' + markers
            req = install_req_from_line(line, comes_from='')
            assert str(req.markers) == str(Marker(markers))
            assert req.match_markers()

        # don't match
        for markers in (
            'python_version >= "5.0"',
            'sys_platform != %r' % sys.platform,
        ):
            line = 'name; ' + markers
            req = install_req_from_line(line, comes_from='')
            assert str(req.markers) == str(Marker(markers))
            assert not req.match_markers()

    def test_extras_for_line_path_requirement(self):
        line = 'SomeProject[ex1,ex2]'
        filename = 'filename'
        comes_from = '-r %s (line %s)' % (filename, 1)
        req = install_req_from_line(line, comes_from=comes_from)
        assert len(req.extras) == 2
        assert req.extras == {'ex1', 'ex2'}

    def test_extras_for_line_url_requirement(self):
        line = 'git+https://url#egg=SomeProject[ex1,ex2]'
        filename = 'filename'
        comes_from = '-r %s (line %s)' % (filename, 1)
        req = install_req_from_line(line, comes_from=comes_from)
        assert len(req.extras) == 2
        assert req.extras == {'ex1', 'ex2'}

    def test_extras_for_editable_path_requirement(self):
        url = '.[ex1,ex2]'
        filename = 'filename'
        comes_from = '-r %s (line %s)' % (filename, 1)
        req = install_req_from_editable(url, comes_from=comes_from)
        assert len(req.extras) == 2
        assert req.extras == {'ex1', 'ex2'}

    def test_extras_for_editable_url_requirement(self):
        url = 'git+https://url#egg=SomeProject[ex1,ex2]'
        filename = 'filename'
        comes_from = '-r %s (line %s)' % (filename, 1)
        req = install_req_from_editable(url, comes_from=comes_from)
        assert len(req.extras) == 2
        assert req.extras == {'ex1', 'ex2'}

    def test_unexisting_path(self):
        with pytest.raises(InstallationError) as e:
            install_req_from_line(
                os.path.join('this', 'path', 'does', 'not', 'exist'))
        err_msg = e.value.args[0]
        assert "Invalid requirement" in err_msg
        assert "It looks like a path." in err_msg

    def test_single_equal_sign(self):
        with pytest.raises(InstallationError) as e:
            install_req_from_line('toto=42')
        err_msg = e.value.args[0]
        assert "Invalid requirement" in err_msg
        assert "= is not a valid operator. Did you mean == ?" in err_msg

    def test_unidentifiable_name(self):
        test_name = '-'
        with pytest.raises(InstallationError) as e:
            install_req_from_line(test_name)
        err_msg = e.value.args[0]
        assert "Invalid requirement: '{}'".format(test_name) == err_msg

    def test_requirement_file(self):
        req_file_path = os.path.join(self.tempdir, 'test.txt')
        with open(req_file_path, 'w') as req_file:
            req_file.write('pip\nsetuptools')
        with pytest.raises(InstallationError) as e:
            install_req_from_line(req_file_path)
        err_msg = e.value.args[0]
        assert "Invalid requirement" in err_msg
        assert "It looks like a path. It does exist." in err_msg
        assert "appears to be a requirements file." in err_msg
        assert "If that is the case, use the '-r' flag to install" in err_msg


@patch('pip._internal.req.req_install.os.path.abspath')
@patch('pip._internal.req.req_install.os.path.exists')
@patch('pip._internal.req.req_install.os.path.isdir')
def test_parse_editable_local(
        isdir_mock, exists_mock, abspath_mock):
    exists_mock.return_value = isdir_mock.return_value = True
    # mocks needed to support path operations on windows tests
    abspath_mock.return_value = "/some/path"
    assert parse_editable('.') == (None, 'file:///some/path', None)
    abspath_mock.return_value = "/some/path/foo"
    assert parse_editable('foo') == (
        None, 'file:///some/path/foo', None,
    )


def test_parse_editable_explicit_vcs():
    assert parse_editable('svn+https://foo#egg=foo') == (
        'foo',
        'svn+https://foo#egg=foo',
        None,
    )


def test_parse_editable_vcs_extras():
    assert parse_editable('svn+https://foo#egg=foo[extras]') == (
        'foo[extras]',
        'svn+https://foo#egg=foo[extras]',
        None,
    )


@patch('pip._internal.req.req_install.os.path.abspath')
@patch('pip._internal.req.req_install.os.path.exists')
@patch('pip._internal.req.req_install.os.path.isdir')
def test_parse_editable_local_extras(
        isdir_mock, exists_mock, abspath_mock):
    exists_mock.return_value = isdir_mock.return_value = True
    abspath_mock.return_value = "/some/path"
    assert parse_editable('.[extras]') == (
        None, 'file://' + "/some/path", {'extras'},
    )
    abspath_mock.return_value = "/some/path/foo"
    assert parse_editable('foo[bar,baz]') == (
        None, 'file:///some/path/foo', {'bar', 'baz'},
    )


def test_exclusive_environment_markers():
    """Make sure RequirementSet accepts several excluding env markers"""
    eq36 = install_req_from_line(
        "Django>=1.6.10,<1.7 ; python_version == '3.6'")
    eq36.is_direct = True
    ne36 = install_req_from_line(
        "Django>=1.6.10,<1.8 ; python_version != '3.6'")
    ne36.is_direct = True

    req_set = RequirementSet()
    req_set.add_requirement(eq36)
    req_set.add_requirement(ne36)
    assert req_set.has_requirement('Django')


def test_mismatched_versions(caplog, tmpdir):
    original_source = os.path.join(DATA_DIR, 'src', 'simplewheel-1.0')
    source_dir = os.path.join(tmpdir, 'simplewheel')
    shutil.copytree(original_source, source_dir)
    req = InstallRequirement(req=Requirement('simplewheel==2.0'),
                             comes_from=None, source_dir=source_dir)
    req.prepare_metadata()
    req.assert_source_matches_version()
    assert caplog.records[-1].message == (
        'Requested simplewheel==2.0, '
        'but installing version 1.0'
    )


@pytest.mark.parametrize('args, expected', [
    # Test UNIX-like paths
    (('/path/to/installable'), True),
    # Test relative paths
    (('./path/to/installable'), True),
    # Test current path
    (('.'), True),
    # Test url paths
    (('https://whatever.com/test-0.4-py2.py3-bogus-any.whl'), True),
    # Test pep440 paths
    (('test @ https://whatever.com/test-0.4-py2.py3-bogus-any.whl'), True),
    # Test wheel
    (('simple-0.1-py2.py3-none-any.whl'), False),
])
def test_looks_like_path(args, expected):
    assert _looks_like_path(args) == expected


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason='Test only available on Windows'
)
@pytest.mark.parametrize('args, expected', [
    # Test relative paths
    (('.\\path\\to\\installable'), True),
    (('relative\\path'), True),
    # Test absolute paths
    (('C:\\absolute\\path'), True),
])
def test_looks_like_path_win(args, expected):
    assert _looks_like_path(args) == expected


@pytest.mark.parametrize('args, mock_returns, expected', [
    # Test pep440 urls
    (('/path/to/foo @ git+http://foo.com@ref#egg=foo',
     'foo @ git+http://foo.com@ref#egg=foo'), (False, False), None),
    # Test pep440 urls without spaces
    (('/path/to/foo@git+http://foo.com@ref#egg=foo',
     'foo @ git+http://foo.com@ref#egg=foo'), (False, False), None),
    # Test pep440 wheel
    (('/path/to/test @ https://whatever.com/test-0.4-py2.py3-bogus-any.whl',
     'test @ https://whatever.com/test-0.4-py2.py3-bogus-any.whl'),
     (False, False), None),
    # Test name is not a file
    (('/path/to/simple==0.1',
     'simple==0.1'),
     (False, False), None),
])
@patch('pip._internal.req.req_install.os.path.isdir')
@patch('pip._internal.req.req_install.os.path.isfile')
def test_get_url_from_path(
    isdir_mock, isfile_mock, args, mock_returns, expected
):
    isdir_mock.return_value = mock_returns[0]
    isfile_mock.return_value = mock_returns[1]
    assert _get_url_from_path(*args) is expected


@patch('pip._internal.req.req_install.os.path.isdir')
@patch('pip._internal.req.req_install.os.path.isfile')
def test_get_url_from_path__archive_file(isdir_mock, isfile_mock):
    isdir_mock.return_value = False
    isfile_mock.return_value = True
    name = 'simple-0.1-py2.py3-none-any.whl'
    path = os.path.join('/path/to/' + name)
    url = path_to_url(path)
    assert _get_url_from_path(path, name) == url


@patch('pip._internal.req.req_install.os.path.isdir')
@patch('pip._internal.req.req_install.os.path.isfile')
def test_get_url_from_path__installable_dir(isdir_mock, isfile_mock):
    isdir_mock.return_value = True
    isfile_mock.return_value = True
    name = 'some/setuptools/project'
    path = os.path.join('/path/to/' + name)
    url = path_to_url(path)
    assert _get_url_from_path(path, name) == url


@patch('pip._internal.req.req_install.os.path.isdir')
def test_get_url_from_path__installable_error(isdir_mock):
    isdir_mock.return_value = True
    name = 'some/setuptools/project'
    path = os.path.join('/path/to/' + name)
    with pytest.raises(InstallationError) as e:
        _get_url_from_path(path, name)
    err_msg = e.value.args[0]
    assert "Neither 'setup.py' nor 'pyproject.toml' found" in err_msg


class TestRequirementTracker(object):
    @pytest.fixture(autouse=True)
    def blank_env(self):
        with patch.dict(os.environ, clear=True):
            yield

    def test_no_side_effects_on_init(self):
        RequirementTracker()
        assert 'PIP_REQ_TRACKER' not in os.environ

    def test_keeps_original_environ(self, tmpdir):
        os.environ['PIP_REQ_TRACKER'] = str(tmpdir)
        with RequirementTracker():
            assert os.environ['PIP_REQ_TRACKER'] == str(tmpdir)
        assert os.environ['PIP_REQ_TRACKER'] == str(tmpdir)

    def test_setting_environ(self):
        with RequirementTracker():
            assert 'PIP_REQ_TRACKER' in os.environ

        # Ensure that the cleanup is proper
        assert 'PIP_REQ_TRACKER' not in os.environ
