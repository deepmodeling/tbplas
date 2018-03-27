# use gnu95 compiler and f2py3
f2py3 --fcompiler=gnu95 --f90flags='-march=native -O3 -fopenmp' -lgomp -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py