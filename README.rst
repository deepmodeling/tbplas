Introduction
============

TBPLaS (Tight-Binding Package for Large-scale Simulation) is a package for building and solving
tight-binding models, with emphasis on handling large systems. TBPLaS implements exact
diagonalization-based methods, the tight-binding propagation method (TBPM), kernel polynomial
method (KPM), and Green's function method. Sparse matrices, Cython/FORTRAN extensions and hybrid
OpenMP+MPI parallelization are utilized for optimal performance on modern computers. The main
capabilities of TBPLaS include:

* Modeling
    * Models with arbitrary shape and boundary conditions
    * Defects, impurities and disorders
    * Hetero-structures, quasicrystal, fractals
    * 1D, 2D and 3D structures
    * Interfaces to Wannier90 and LAMMPS
* Exact-diagonalization
    * Band structure, density of states (DOS), wave functions, topological invariants
    * Polarizability, dielectric function, AC conductivity
* Tight-binding propagation method (TBPM)
    * DOS, LDOS and carrier density
    * Optical (AC) conductivity and absorption spectrum
    * Electronic (DC) conductivity and time-dependent diffusion coefficient
    * Carrier velocity, mobility, elastic mean free path, Anderson localization length 
    * Polarization function, response function, dielectric function, energy loss function
    * Plasmon dispersion, plasmon lifetime and damping rate
    * Quasi-eigenstate and realspace charge density
    * Time-dependent wave function
* Kernel polynomial method
    * Electronic (DC) and Hall Conductivity  
* Recursive Green's function method
    * Local density of states (LDOS)
* Fields and strains
    * Homogeneous magnetic field via Peierls substitution
    * User-defined electric field
    * Arbitary deformation with strain and/or stress

Installation
------------

Dependences
^^^^^^^^^^^

1. Operating system
    Currently TBPLaS supports Linux only. Try virtual machines if you have to work on Windows.

2. Compilers and Libraries
    * Fortran compiler with Fortran 2003 support
      Intel Fortran(version >= 2016) with MKL is highly recommended.
      GNU Fortran(version >= 5.0) is an alternative.
      Other compilers are not tested, but may still work.
    * C compiler
      Use the compiler from the same vendor as FORTRAN.
    * MPI (optimal)
      MPICH3 is recommended. OpenMPI is an alternative.

3. Python environment
    * Python 3 and development files
      Python >=3.6 is required. Python 2.x is not supported and will definitely fail.
      The Python intepreter and development files can be installed via system package manager,
      or through Anaconda.
    * Packages
      * numpy
      * scipy
      * matplotlib
      * cython
      * setuptools
      * ase (optimal)
      * mpi4py (optimal)

Procedure
^^^^^^^^^

1. Adjust *setup.cfg* according to your hardware and software environment.
   You can use default Intel compiler and MKLlibrary, or use GNU compiler.
   Examples of *setup.cfg* are placed under *config* directory.
2. Build the extensions with ``python setup.py build``.
3. Install TBPLaS into the default path with ``python setup.py install``,
   or into user-defined path with ``python setup.py install --prefix=/user_defined_path``.
4. Add installation path to the *PYTHONPATH* environment variable (optimal).

Now, you can try TBPLaS in python using ``import tbplas``.

Tutorials
---------

Some examples demonstrating the features of TBPLaS can be found under *examples* directory.
More detailed tutorials can be found in the documentation.

Documentation
-------------

The documentation is available online at `<http://www.tbplas.net>`_. A local copy can be found
under *doc/build/html/index.html*.

Citation
--------

See *CITING.rst* for more details.

License
-------

TBPLaS is released under the BSD license. See *LICENSE.rst* for more details.
