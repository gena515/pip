import sys

from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import List, Sequence

# Shim to wrap setup.py invocation with setuptools
#
# We set sys.argv[0] to the path to the underlying setup.py file so
# setuptools / distutils don't take the path to the setup.py to be "-c" when
# invoking via the shim.  This avoids e.g. the following manifest_maker
# warning: "warning: manifest_maker: standard file '-c' not found".
_SETUPTOOLS_SHIM = (
    "import sys, setuptools, tokenize; sys.argv[0] = {0!r}; __file__={0!r};"
    "f=getattr(tokenize, 'open', open)(__file__);"
    "code=f.read().replace('\\r\\n', '\\n');"
    "f.close();"
    "exec(compile(code, __file__, 'exec'))"
)


def make_setuptools_shim_args(
    setup_py_path,  # type: str
    global_options=None,  # type: Sequence[str]
    no_user_config=False,  # type: bool
    unbuffered_output=False,  # type: bool
):
    # type: (...) -> List[str]
    """
    Get setuptools command arguments with shim wrapped setup file invocation.

    :param setup_py_path: The path to setup.py to be wrapped.
    :param global_options: Additional global options.
    :param no_user_config: If True, disables personal user configuration.
    :param unbuffered_output: If True, adds the unbuffered switch to the
     argument list.
    """
    args = [sys.executable]
    if unbuffered_output:
        args.append("-u")
    args.extend(["-c", _SETUPTOOLS_SHIM.format(setup_py_path)])
    if global_options:
        args.extend(global_options)
    if no_user_config:
        args.append("--no-user-cfg")
    return args
