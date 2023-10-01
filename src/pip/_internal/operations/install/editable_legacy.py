"""Legacy editable installation process, i.e. `setup.py develop`.
"""
import logging
from typing import Optional

from pip._internal.build_env import BuildEnvironment
from pip._internal.utils.logging import indent_log
from pip._internal.utils.setuptools_build import make_setuptools_develop_args
from pip._internal.utils.subprocess import call_subprocess

logger = logging.getLogger(__name__)


def install_editable(
    *,
    prefix: Optional[str],
    home: Optional[str],
    use_user_site: bool,
    name: str,
    setup_py_path: str,
    isolated: bool,
    build_env: BuildEnvironment,
    unpacked_source_directory: str,
) -> None:
    """Install a package in editable mode. Most arguments are pass-through
    to setuptools.
    """
    logger.info("Running setup.py develop for %s", name)

    args = make_setuptools_develop_args(
        setup_py_path,
        no_user_config=isolated,
        prefix=prefix,
        home=home,
        use_user_site=use_user_site,
    )

    with indent_log():
        with build_env:
            call_subprocess(
                args,
                command_desc="python setup.py develop",
                cwd=unpacked_source_directory,
            )
