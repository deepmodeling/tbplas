#!/bin/bash
# fortran compilation script using Intel Fortran Compiler
f2py --compiler=intelem --fcompiler=intelem \
     --opt="-qopenmp -O3 -march=native -heap-arrays -mkl=parallel" \
     -liomp5 -lifcoremt \
     -c tbpm_f2py.pyf tbpm_module_mkl.f90 tbpm_f2py_mkl.f90
