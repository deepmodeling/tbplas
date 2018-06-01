#!/bin/bash
# fortran compilation script using Intel Fortran Compiler
f2py --compiler=intelem --fcompiler=intelem --f90flags="-qopenmp -O3 -march=native -heap-arrays -mkl=parallel" -liomp5 -lifcoremt -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py
