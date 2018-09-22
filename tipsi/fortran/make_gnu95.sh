#!/bin/bash

# make python wrapper
f2py -h f2py.pyf -m f2py analysis.f90 tbpm.f90 --overwrite-signature

# fortran compilation script using gnu95
f2py --fcompiler=gnu95 \
  --f90flags="-march=native -O3 -fopenmp" -lgomp \
  -c f2py.pyf const.f90 math.f90 csr.f90 fft.f90 \
     random.f90 propagation.f90 funcs.f90 tbpm.f90 analysis.f90

rm -f f2py.pyf
