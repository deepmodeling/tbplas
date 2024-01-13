#! /bin/bash
# Copy so files to their destinations
# It is unnecessary for the new build system. But generating API reference
# with sphinx requires a local copy of the so files.

cp build/f2py*.so tbplas/fortran 
for i in primitive super sample lindhard atom; do
    cp build/$i*.so tbplas/cython
done
