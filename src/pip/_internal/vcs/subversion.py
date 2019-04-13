from __future__ import absolute_import

import logging
import os
import re

from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
    display_path, rmtree, split_auth_from_netloc,
)
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.vcs import VersionControl, vcs

_svn_xml_url_re = re.compile('url="([^"]+)"')
_svn_rev_re = re.compile(r'committed-rev="(\d+)"')
_svn_info_xml_rev_re = re.compile(r'\s*revision="(\d+)"')
_svn_info_xml_url_re = re.compile(r'<url>(.*)</url>')


if MYPY_CHECK_RUNNING:
    from typing import Tuple

logger = logging.getLogger(__name__)


class Subversion(VersionControl):
    name = 'svn'
    dirname = '.svn'
    repo_name = 'checkout'
    schemes = ('svn', 'svn+ssh', 'svn+http', 'svn+https', 'svn+svn')

    @classmethod
    def should_add_vcs_url_prefix(cls, remote_url):
        return True

    @staticmethod
    def get_base_rev_args(rev):
        return ['-r', rev]

    def get_vcs_version(self):
        # type: () -> Tuple[int, ...]
        """Return the version of the currently installed Subversion client.

        :return Tuple containing the parts of the version information.
        :raises BadCommand: If ``svn`` is not installed.
            ValueError: If the version returned from ``svn`` could not be
            parsed.
        """
        version_pfx = 'svn, version '
        version = self.run_command(['--version'], show_stdout=False)
        if version.startswith(version_pfx):
            version = version[len(version_pfx):].split()[0]
        else:
            version = ''

        # Example versions:
        #   svn, version 1.10.3 (r1842928)
        #   svn, version 1.7.14 (r1542130)
        version = version.split('.')
        return tuple(map(int, version))

    def export(self, location):
        """Export the svn repository at the url to the destination location"""
        url, rev_options = self.get_url_rev_options(self.url)

        logger.info('Exporting svn repository %s to %s', url, location)
        with indent_log():
            if os.path.exists(location):
                # Subversion doesn't like to check out over an existing
                # directory --force fixes this, but was only added in svn 1.5
                rmtree(location)
            cmd_args = ['export'] + rev_options.to_args() + [url, location]
            self.run_command(cmd_args, show_stdout=False)

    @classmethod
    def fetch_new(cls, dest, url, rev_options):
        rev_display = rev_options.to_display()
        logger.info(
            'Checking out %s%s to %s',
            url,
            rev_display,
            display_path(dest),
        )
        cmd_args = ['checkout', '-q'] + rev_options.to_args() + [url, dest]
        cls.run_command(cmd_args)

    def switch(self, dest, url, rev_options):
        cmd_args = ['switch'] + rev_options.to_args() + [url, dest]
        self.run_command(cmd_args)

    def update(self, dest, url, rev_options):
        cmd_args = ['update'] + rev_options.to_args() + [dest]
        self.run_command(cmd_args)

    @classmethod
    def get_revision(cls, location):
        """
        Return the maximum revision for all files under a given location
        """
        # Note: taken from setuptools.command.egg_info
        revision = 0

        for base, dirs, files in os.walk(location):
            if cls.dirname not in dirs:
                dirs[:] = []
                continue    # no sense walking uncontrolled subdirs
            dirs.remove(cls.dirname)
            entries_fn = os.path.join(base, cls.dirname, 'entries')
            if not os.path.exists(entries_fn):
                # FIXME: should we warn?
                continue

            dirurl, localrev = cls._get_svn_url_rev(base)

            if base == location:
                base = dirurl + '/'   # save the root url
            elif not dirurl or not dirurl.startswith(base):
                dirs[:] = []
                continue    # not part of the same svn tree, skip it
            revision = max(revision, localrev)
        return revision

    @classmethod
    def get_netloc_and_auth(cls, netloc, scheme):
        """
        This override allows the auth information to be passed to svn via the
        --username and --password options instead of via the URL.
        """
        if scheme == 'ssh':
            # The --username and --password options can't be used for
            # svn+ssh URLs, so keep the auth information in the URL.
            return super(Subversion, cls).get_netloc_and_auth(netloc, scheme)

        return split_auth_from_netloc(netloc)

    @classmethod
    def get_url_rev_and_auth(cls, url):
        # hotfix the URL scheme after removing svn+ from svn+ssh:// readd it
        url, rev, user_pass = super(Subversion, cls).get_url_rev_and_auth(url)
        if url.startswith('ssh://'):
            url = 'svn+' + url
        return url, rev, user_pass

    @staticmethod
    def make_rev_args(username, password):
        extra_args = []
        if username:
            extra_args += ['--username', username]
        if password:
            extra_args += ['--password', password]

        return extra_args

    @classmethod
    def get_remote_url(cls, location):
        # In cases where the source is in a subdirectory, not alongside
        # setup.py we have to look up in the location until we find a real
        # setup.py
        orig_location = location
        while not os.path.exists(os.path.join(location, 'setup.py')):
            last_location = location
            location = os.path.dirname(location)
            if location == last_location:
                # We've traversed up to the root of the filesystem without
                # finding setup.py
                logger.warning(
                    "Could not find setup.py for directory %s (tried all "
                    "parent directories)",
                    orig_location,
                )
                return None

        return cls._get_svn_url_rev(location)[0]

    @classmethod
    def _get_svn_url_rev(cls, location):
        from pip._internal.exceptions import InstallationError

        entries_path = os.path.join(location, cls.dirname, 'entries')
        if os.path.exists(entries_path):
            with open(entries_path) as f:
                data = f.read()
        else:  # subversion >= 1.7 does not have the 'entries' file
            data = ''

        if (data.startswith('8') or
                data.startswith('9') or
                data.startswith('10')):
            data = list(map(str.splitlines, data.split('\n\x0c\n')))
            del data[0][0]  # get rid of the '8'
            url = data[0][3]
            revs = [int(d[9]) for d in data if len(d) > 9 and d[9]] + [0]
        elif data.startswith('<?xml'):
            match = _svn_xml_url_re.search(data)
            if not match:
                raise ValueError('Badly formatted data: %r' % data)
            url = match.group(1)    # get repository URL
            revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)] + [0]
        else:
            try:
                # subversion >= 1.7
                xml = cls.run_command(
                    ['info', '--xml', location],
                    show_stdout=False,
                )
                url = _svn_info_xml_url_re.search(xml).group(1)
                revs = [
                    int(m.group(1)) for m in _svn_info_xml_rev_re.finditer(xml)
                ]
            except InstallationError:
                url, revs = None, []

        if revs:
            rev = max(revs)
        else:
            rev = 0

        return url, rev

    @classmethod
    def is_commit_id_equal(cls, dest, name):
        """Always assume the versions don't match"""
        return False


vcs.register(Subversion)
