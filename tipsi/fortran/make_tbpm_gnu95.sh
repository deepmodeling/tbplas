# use gnu95 compiler and f2py
f2py --fcompiler=gnu95 --opt='-march=native -O3 -fopenmp' -lgomp \
     -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py
