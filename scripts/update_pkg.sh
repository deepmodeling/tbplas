#! /bin/bash

packages=$(pip list --outdated | awk 'NR>2 {print $1}')
for pkg in $packages; do
    pip install --upgrade $pkg
done
