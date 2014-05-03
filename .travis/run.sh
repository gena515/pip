#!/bin/bash

set -e
set -x

# Point TMPDIR at our RAMFS
if [[ "$(uname -s)" == "Darwin" ]]; then
    export TMPDIR="/Volumes/Temporary"
else
    export TMPDIR="/mnt/ramdisk"
fi

# This is required in order to get UTF-8 output inside of the subprocesses that
# our tests use.
export LC_CTYPE=en_US.UTF-8

source ~/.venv/bin/activate

case $TOXENV in
    py32)
        tox
        ;;
    *)
        tox -- -n 8
        ;;
esac
