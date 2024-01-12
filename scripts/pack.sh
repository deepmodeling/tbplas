#! /bin/bash

items="CITING.rst CMakeLists.txt config doc examples LICENSE.rst \
pyproject.toml README.rst RELEASE.rst scripts setup.cfg setup.py \
tbplas tests"

# Copy items to destination
test -d tmp/tbplas && rm -rf tmp/tbplas
mkdir -p tmp/tbplas
for item in $items; do
    cp -r $item tmp/tbplas
done

# Remove unnecessary files
cd tmp
for kind in pyc; do
    find . -name "*.$kind" | xargs rm -rf
done

for kind in __pycache__ sim_data; do
    find . -name "$kind" | xargs rm -rf
done

# Compress
tar -cjf tbplas.tar.bz2 tbplas
mv tbplas.tar.bz2 ..

# Clean
cd ..
rm -rf tmp
