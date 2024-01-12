#! /bin/bash

cp build/f2py*.so tbplas/fortran 
for i in primitive super sample lindhard atom; do
    cp build/$i*.so tbplas/cython
done
