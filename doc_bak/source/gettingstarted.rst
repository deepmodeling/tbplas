===============
Getting started
===============

Requirements
------------
Tipsi requires python3, some dependencies and a fortran compiler. The easiest way to get python3 is to download a version of anaconda at:

    https://conda.io/docs/user-guide/install/index.html

Packages required by TiPSi:
    - numpy
    - scipy
    
Optional:
    - h5py (for binary file input/output)
    - matplotlib (for plotting)

You can get a fortran compiler by running::

    sudo apt-get install gfortran
    
or by manually downloading the gnu fortran compiler from

    https://gcc.gnu.org/fortran/
    
Alternatively, tipsi also works with the intel fortran compiler.

Installing with setup.py
------------------------

With git, you can get the tipsi files by using::

    git clone https://gitlab.science.ru.nl/tcm/tipsi

Alternatively, the tipsi files can be downloaded manually from:

    https://gitlab.science.ru.nl/tcm/tipsi

Now, if you are using the gfortran compiler, you can proceed to running **setup.py**.
If you are using the intel compiler, you need to rename **setup.cfg.intel** to **setup.cfg**.
If needed, you can link additional libraries in the **setup.cfg** file.

Then, go to the main directory and run::

    python3 setup.py install

This will install tipsi in your default python package location.

To install it in a different location, if you don't have sudo rights, use::

    python3 setup.py install --home=<dir>

This directory needs to be added manually to your python path.

Manual installation
-------------------

Alternatively, one can compile the files in the fortran subdirectory by hand.

For gfortran::

    f2py --fcompiler=gnu95 --f90flags='-march=native -O3 -fopenmp' -lgomp -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py

For intel::

    f2py --fcompiler=intelem --f90flags="-qopenmp -O3 -march=native -heap-arrays" -liomp5 -lifcoremt -c tbpm_module.f90 tbpm_f2py.f90 -m tbpm_f2py

For some versions of python3/numpy you need to use the command ``f2py3`` instead of ``f2py``.
    
A selection of compilation scripts can be found in the ``tipsi/fortran/`` folder.
