#!/bin/bash

# fortran compilation script for Radboud university Nijmegen FNWI cluster nodes
source /vol/opt/intelcompilers/intel-2014/composerxe/bin/compilervars.sh intel64

# make python wrapper
f2py3 -h f2py.pyf -m f2py analysis.f90 tbpm.f90 --overwrite-signature

# fortran compilation script using Intel Fortran Compiler
f2py3 --compiler=intelem --fcompiler=intelem \
   --opt="-qopenmp -O3 -march=native -heap-arrays" \
   -L/vol/opt/intelcompilers/intel-2014/composerxe/lib/intel64 \
   -liomp5 -lifcoremt \
   -c f2py.pyf const.f90 math.f90 csr.f90 fft.f90 \
      random.f90 propagation.f90 funcs.f90 tbpm.f90 analysis.f90

rm -f f2py.pyf
