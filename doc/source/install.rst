Install
=======

Requirements
------------

Operating system
^^^^^^^^^^^^^^^^

TBPLaS has been developed and tested on Linux. Ideally, it should work on most Linux distributions.
Due to compiler and library compatibility issues, native compilation of TBPLaS on Windows is difficult.
If you wish to install TBPLaS on Windows, the best practice is to use virtual machines, e.g., VirtualBox
or VMWare. Windows Subsystem for Linux (WSL) is also an option. Since MacOS is actually a unix-like
operating system, installing TBPLaS may be possible, as long as the python environment and packages have
been properly installed. But we have no experience of MacOS. You are encouraged to send us feedbacks if
you have made it.

Compilers and math libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The performance-critical parts of TBPLaS are written in Cython and FORTRAN. Also, sparse matrices are utilized to
reduce memory cost. So, you need both C and FORTRAN compilers to compile TBPLaS, and link it to vendor-provided
math libraries for optimal performance. For Intel CPUs, it is recommended to use Intel compilers and Math Kernel
Library (MKL), which are now bundled as oneAPI. If the Intel toolchain is not available, GNU Compiler Collection
(GCC) is a general choice. In that case, TBPLaS will use the built-in sparse matrix library.

Python environment
^^^^^^^^^^^^^^^^^^

TBPLaS requires ``Python>=3.7``. In addition to the python interpreter, development headers are also required.
The actual package name of development headers may vary among Linux distributions, e.g., ``python3-devel`` on
rpm-based distributions like CentOS Stream, and ``python3-dev`` on deb-based distributions like Debian. Check the
manual of your distribution package manager, e.g., ``dnf`` or ``apt-get``, for guidelines on searching and
installing the package.

If you do not have root privileges, or if your computer cannot access the Internet, try the
`Anaconda <https://www.anaconda.com/products/individual>`_ offline installer, which bundles the development headers
as well as other dependencies.

.. _dependencies:

Dependencies
^^^^^^^^^^^^

TBPLaS requires the following Python packages as mandatory dependencies:

* numpy
* scipy
* matplotlib
* cython
* scikit-build-core

And the following packages as optional:

* ase (for LAMMPS interface)
* mpi4py (for hybrid MPI+OpenMP parallelization)

Note that ``scikit-build-core`` requires ``CMake>=3.17`` to be installed. We recommend to install the latest version of
the packages via the ``pip`` or ``conda`` commands, e.g., ``pip install numpy`` or ``conda install numpy``.
If you do not have root privileges to make a system-wide installation, try either of the following solutions:

* Create a virtual environment in your home direcotry and activate it before installation (recommended for most users)
* Add the ``--user`` option, e.g. ``pip install --user numpy`` (recommended for most users)
* Add the ``--prefix`` option, e.g. ``pip install --prefix=DEST numpy``, and update ``PYTHONPATH`` accordingly
  (only for advanced users)


The installation of ``mpi4py`` is somewhat complex. You need a working MPI implementation. Since OpenMPI
suffers from a limitation on the number of OpenMP threads when there are too few MPI processes,
MPICH3 or newer is preferred. See the following links for installing
`MPICH <https://www.mpich.org/documentation/guides/>`_,
`OpenMPI <https://www.open-mpi.org//faq/?category=building>`_,
and `MPI4PY <https://mpi4py.readthedocs.io/en/stable/install.html>`_.

.. _get_src:

Get the source code
-------------------

The source code is available at `this link <attachments/tbplas.tar.bz2>`_.
If you have downloading problems, send an e-mail to the development team at :ref:`developers`.

Configuration
-------------

The configuration of compilation is stored in the ``tool.scikit-build.cmake.define`` section of
``pyproject.toml`` in the top directory of TBPLaS source code, which should look like:

.. code-block:: toml
    :emphasize-lines: 0

    [tool.scikit-build.cmake.define]
    USE_INDEX64 = "OFF"
    USE_MKL = "OFF"
    # GCC
    CMAKE_C_COMPILER = "gcc"
    CMAKE_Fortran_COMPILER = "gfortran"
    CMAKE_C_FLAGS = "-march=native -mtune=native"
    CMAKE_Fortran_FLAGS = "-cpp -march=native -mtune=native -fopenmp -fno-second-underscore"
    ## Intel
    #CMAKE_C_COMPILER = "icx"
    #CMAKE_Fortran_COMPILER = "ifx"
    #CMAKE_C_FLAGS = "-xHost"
    #CMAKE_Fortran_FLAGS = "-fpp -xHost -qopenmp -ipo -heap-arrays 32"

Here ``USE_INDEX64`` and ``USE_MKL`` are two switches whose value should be either ``ON`` or ``OFF``. The following
lines define the C and Fortran compilers and the compilation flags. Uncomment the lines of the compiler you wish to
use. Note that the executables of Intel compiler have been recently renamed to ``icx`` and ``ifx``. For older versions,
they may be ``icc`` and ``ifort``. For Intel compiler, you can also use the sparse matrix library from MKL by setting
``USE_MKL`` to ``ON``.

64-bit array index
^^^^^^^^^^^^^^^^^^

TBPLaS uses 32-bit array index by default, even if it has been compiled and installed on a 64-bit host. While the RAM
usage is reduced in this approach, segmentation fault may be raised if the model is very large (billions of orbitals).
In that case, the version with 64-bit array index should be used. To compile the 64-bit version, firstly go to
``tbplas/fortran`` directory and pre-process the FORTRAN source files by:

.. code-block:: bash

    cd tbplas/fortran
    ../../scripts/set_int.py 64

Then set the ``USE_INDEX64`` switch to ``ON``. The compilation flags will be updated automatically, so no further
configuration is required. Note that MKL DOES NOT work with 64-bit array index.

Installation
------------

Once ``pyproject.toml`` has been properly configured, you can build and install TBPLaS with ``pip install .``.
The package will be install to the default library path. If you do not have root privileges, try the solutions
aforementioned in :ref:`dependencies`. After installation, go to some other directory and invoke Python, e.g.,
``cd tests && python``. Since TBPLaS uses relative imports for package management, staying in the source code
directory when invoking Python may cause errors like this:

.. code-block:: python

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/yhli/proj/tbplas/tbplas/__init__.py", line 2, in <module>
        from .adapter import *
      File "/home/yhli/proj/tbplas/tbplas/adapter/__init__.py", line 2, in <module>
        from .wannier90 import *
      File "/home/yhli/proj/tbplas/tbplas/adapter/wannier90.py", line 11, in <module>
        from ..builder import PrimitiveCell, PCHopDiagonalError
      File "/home/yhli/proj/tbplas/tbplas/builder/__init__.py", line 2, in <module>
        from .advanced import *
      File "/home/yhli/proj/tbplas/tbplas/builder/advanced.py", line 17, in <module>
        from .primitive import PrimitiveCell, PCInterHopping
      File "/home/yhli/proj/tbplas/tbplas/builder/primitive.py", line 12, in <module>
        from ..cython import primitive as core
    ImportError: cannot import name 'primitive' from 'tbplas.cython' (/home/yhli/proj/tbplas/tbplas/cython/__init__.py)

So it is mandatory to go to another directory. After Python has been invoked, try ``import tbplas``. If no error occurs,
then your installation is successful.

Testing
-------

There are some testing scripts under the ``tests`` directory of source code. You can test your compilation and
installation by invoking these scripts, e.g., ``python test_base.py``. Some output will be printed to the screen and
some figures will be saved to disk. If everything goes well, a notice will be raised saying all the tests have been
passed by the end of each script.
