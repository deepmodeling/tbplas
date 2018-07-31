#!/bin/bash
# fortran compilation script for Radboud university Nijmegen FNWI cluster nodes
source /vol/opt/intelcompilers/intel-2014/composerxe/bin/compilervars.sh intel64 
source /vol/opt/intelcompilers/intel-2014/mkl/bin/intel64/mklvars_intel64.sh
f2py3 --fcompiler=intelem --f90flags="-openmp -O3 -march=native -heap-arrays -mkl=parallel" \
-L/vol/opt/intelcompilers/intel-2014/composerxe/lib/intel64 -liomp5 -lifcoremt \
-c tbpm_module_mkl.f90 tbpm_f2py.f90 -m tbpm_f2py