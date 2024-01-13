#! /bin/bash
# Build so files

test -d build && rm -rf build
pip install --prefix=/tmp/tbplas --verbose .
