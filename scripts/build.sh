#! /bin/bash

test -d build && rm -rf build
pip install --user --verbose .
