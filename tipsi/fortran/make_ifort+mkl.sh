#!/bin/bash

# make python wrapper
f2py -h f2py.pyf -m f2py analysis.f90 tbpm.f90 --overwrite-signature

# fortran compilation script using Intel Fortran Compiler
f2py --compiler=intelem --fcompiler=intelem \
     --opt="-qopenmp -O3 -march=native -heap-arrays -mkl=parallel" \
     -liomp5 -lifcoremt \
     -c f2py.pyf const.f90 math_blas.f90 csr_mkl.f90 fftw.f90 \
        random.f90 propagation.f90 funcs.f90 tbpm.f90 analysis.f90

rm -f f2py.pyf
