#!/bin/bash
source /vol/opt/intelcompilers/intel-2014/composerxe/bin/compilervars.sh intel64 
f2py3 --fcompiler=intelem --f90flags="-openmp -O3 -march=native -heap-arrays" -L/vol/opt/intelcompilers/intel-2014/composerxe/lib/intel64 -liomp5 -lifcoremt -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py
