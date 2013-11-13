"""Exceptions used throughout package"""
from __future__ import absolute_import


class PipError(Exception):
    """Base pip exception"""


class InstallationError(PipError):
    """General exception during installation"""


class UninstallationError(PipError):
    """General exception during uninstallation"""


class DistributionNotFound(InstallationError):
    """Raised when a distribution cannot be found to satisfy a requirement"""


class BestVersionAlreadyInstalled(PipError):
    """Raised when the most up-to-date version of a package is already
    installed.  """


class BadCommand(PipError):
    """Raised when virtualenv or a command is not found"""


class CommandError(PipError):
    """Raised when there is an error in command-line arguments"""


class PreviousBuildDirError(PipError):
    """Raised when there's a previous conflicting build directory"""


class HashMismatch(InstallationError):
    """Distribution file hash values don't match."""


class InvalidWheelFilename(InstallationError):
    """Invalid wheel filename."""


class UnsupportedWheel(InstallationError):
    """Unsupported wheel."""


class CircularSymlinkException(PipError):
    """When a circular symbolic link is detected."""
    def __init__(self, resources):
        message = (
            'Circular reference detected in "%s" ("%s" > "%s").'
            '' % (resources[0], '" > "'.join(resources), resources[0]),
        )
        PipError.__init__(self, message)
