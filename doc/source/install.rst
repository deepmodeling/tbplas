Install
=======

Requirements
------------

Operating system
^^^^^^^^^^^^^^^^

TBPLaS has been developed and tested on Linux. Ideally, it should work on most Linux distributions.

Due to compiler and library compatibility issues, native compilation of TBPLaS on Windows is particularly
difficult. So, if you wish to install TBPLaS on Windows, the best practice is to use virtual machines,
e.g., VirtualBox or VMWare. Windows Subsystem for Linux (WSL) is also an option. If you insist on native
compilation, and if you are an expert on computer, follow the guidelines
`cython on windows <https://stackoverflow.com/questions/52864588/how-to-install-cython-an-anaconda-64-bits-with-windows-10>`_
and
`f2py on windows <https://stackoverflow.com/questions/48826283/compile-fortran-module-with-f2py-and-python-3-6-on-windows-10>`_

Unfortunately, we have no experience of MacOS. You are encouraged to send us feedbacks on this operating system.

Compilers and math libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The performance-critical parts of TBPLaS are written in Cython and FORTRAN. Also, sparse matrices are utilized to
reduce memory cost. So, you need both C and FORTRAN compilers to compile TBPLaS, and link it to vendor-provided
math libraries for optimal performance. For Intel CPUs, it is recommended to use Intel compilers and Math Kernel
Library (MKL), which are now bundled as oneAPI. If the Intel toolchain is not available, GNU Compiler Collection
(GCC) is a general choice. In that case, built-in sparse matrix library will be enabled automatically.

Python
^^^^^^

TBPLaS requires Python 3.7 or newer. Python 2 is not supported.

In addition to the Python interpreter, development headers are also required. The actual package name of
development headers may vary among Linux distributions, e.g., ``python3-devel`` on rpm-based distributions
such as RedHat/CentOS/Fedora/openSUSE, and ``python3-dev`` on deb-based distributions like Debian/Ubuntu.
Check the manual of your distribution package manager (``yum`` / ``dnf`` / ``zypper`` / ``apt-get``) for
guidelines on searching and installing the package.

If you are a newbie to Linux, or if you do not have root privileges, or if you are working with a computer
without Internet access, try the `Anaconda <https://www.anaconda.com/products/individual>`_ offline installer,
which bundles the development headers as well as other dependencies.

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

We recommend to install the latest version of the packages. The packages can be installed  via the ``pip`` or
``conda`` commands, e.g., ``pip install numpy`` or ``conda install numpy``. If you do not have root privileges
to make a system-wide installation, create a virtual environment under your home direcotry and activate it
before installing the packages.

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

**CAUTION:** the old installation procedure based on ``setup.cfg`` and ``setup.py`` is deprecated.
The files are kept for compatibility concerns only, and will be removed soon.

General rules
^^^^^^^^^^^^^

The configuration of compilation is stored in the ``tool.scikit-build.cmake.define`` section in
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
In that case, the version with 64-bit array index should be used. To compile the 64-bit version, first go to
``tbplas/fortran`` directory and pre-process the FORTRAN source files by:

.. code-block:: bash

    cd tbplas/fortran
    ../../scripts/set_int.py

Then set the ``USE_INDEX64`` switch to ``ON``. Note that MKL DOES NOT work with 64-bit array index.

Installation
------------

Once ``pyproject.toml`` has been properly configured, you can build and install TBPLaS with ``pip install .``.
The package will be install to the default library path. If you do not have root privileges,
try ``pip install --user .``, which will install TBPLaS to ``$HOME/.local/lib/pythonM.N``, with M and N being
the version numbers. Alternatively, you can install TBPLaS to specific directory with the ``--prefix`` option,
e.g., ``pip install --user --prefix=$HOME/test .`` will install TBPLaS into the directory of ``$HOME/test``.
You must add the follow directory to the ``PYTHONPATH`` environment variable depending on your python version

.. code-block:: shell
    :emphasize-lines: 0

    export PYTHONPATH=$HOME/test/lib/python3.12/site-packages:$PYTHONPATH

After installation, go to some other directory and invoke Python, e.g., ``cd tests && python``. Since TBPLaS
uses relative imports for package management, staying in the source code directory when invoking Python will
cause errors like this:

.. code-block:: txt

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
