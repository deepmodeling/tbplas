#! /bin/bash
# Clean local build and so files

test -d build && rm -rf build
find . -name "*.so" | xargs rm
