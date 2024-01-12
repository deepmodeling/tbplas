#! /bin/bash

test -d build && rm -rf build
find . -name "*.so" | xargs rm
