#!/bin/bash
# fortran compilation script for Radboud university Nijmegen FNWI cluster nodes
source /vol/opt/intelcompilers/intel-2014/composerxe/bin/compilervars.sh intel64
f2py3 --compiler=intelem --fcompiler=intelem \
      --opt="-openmp -O3 -march=native -heap-arrays" \
      -L/vol/opt/intelcompilers/intel-2014/composerxe/lib/intel64 \
      -liomp5 -lifcoremt -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py
