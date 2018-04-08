Requirements
=================================
Currently tipsi requires python3 with some packages correctly installed. If you don't have these installed, the most easy way of obtaining a working version is to download a version of anaconda at:

    https://www.continuum.io/downloads

Packages reqiured by TiPSi:
    *numpy*
    *scipy*
    *h5py*
    *multiprocessing*
    *matplotlib* (optional, used for plotting)

To be sure you have a working version of a fortran compiler (either GNU fortran compiler or intel fortran compiler).

Installing
=================================

Download the files. First check **setup.cfg** and be sure that all the compiler options are set correctly. For example set::

    f90flags=-qopenmp
    f90flags=-fopenmp

depending on your compiler (the former for intel and the latter for gnufortran). If needed, one can link additional libraries in this file.

Ones this is configured, in the main directory type::

    python setup.py install

This will install tipsi in your default python package location. Options considering
your (fortan) compiler can be put in **setup.cfg**, if you want to not use the default.

To install it in a different location, when one doesn't have sudo rights, use::

    python setup.py install --home=<dir>

This directory needs to be added to your python path.

Alternative
*************************************

Alternatively, one can compile the files in the fortran subdirectory by hand::

    f2py --fcompiler=gfortran --f90flags="-fopenmp -O3 -march=native -heap-arrays" -c tbpm_module.f90 -m tbpm_f2py


Use additional flags when on the radboud university computers::

    -L/vol/opt/intelcompilers/intel-2014/composerxe/lib/intel64 -liomp5 -lifcoremt
