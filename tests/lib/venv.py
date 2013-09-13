from __future__ import absolute_import

import os
import subprocess

import virtualenv as _virtualenv

from .path import Path


# On Python < 3.3 we don't have subprocess.DEVNULL
try:
    DEVNULL = subprocess.DEVNULL
except AttributeError:
    DEVNULL = open(os.devnull, "wb")


class VirtualEnvironment(object):
    """
    An abstraction around virtual environments, currently it only uses
    virtualenv but in the future it could use pyvenv.
    """

    def __init__(self, location, *args, **kwargs):
        self.location = Path(location)
        self.pip_source_dir = kwargs.pop("pip_source_dir")
        self._system_site_packages = kwargs.pop("system_site_packages", False)

        super(VirtualEnvironment, self).__init__(*args, **kwargs)

    def __repr__(self):
        return "<VirtualEnvironment {}>".format(self.location)

    @classmethod
    def create(cls, location, clear=False, pip_source_dir=None):
        obj = cls(location, pip_source_dir=pip_source_dir)
        obj._create(clear=clear)
        return obj

    def _create(self, clear=False):
        # Create the actual virtual environment
        _virtualenv.create_environment(
            self.location,
            clear=clear,
            never_download=True,
            no_pip=True,
        )

        # Install our development version of pip install the virtual
        # environment
        cmd = [self.location.join("bin", "python"), "setup.py", "develop"]
        p = subprocess.Popen(
            cmd,
            cwd=self.pip_source_dir,
            stderr=subprocess.STDOUT,
            stdout=DEVNULL,
        )
        p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd[0],
                output=p.stdout,
            )

    def clear(self):
        self._create(clear=True)

    @property
    def system_site_packages(self):
        return self._system_site_packages

    @system_site_packages.setter
    def system_site_packages(self, value):
        _, lib, _, _ = _virtualenv.path_locations(self.location)
        marker = Path(lib).join("no-global-site-packages.txt")
        if value:
            marker.rm()
        else:
            marker.touch()
        self._system_site_packages = value
