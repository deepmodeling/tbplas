#! /bin/bash
# Build and install into ~/.local

test -d build && rm -rf build
pip install --user --verbose .
