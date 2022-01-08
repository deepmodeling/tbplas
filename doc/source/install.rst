=======
Install
=======

Requirements
------------

Operating system
^^^^^^^^^^^^^^^^

.. rubric:: Linux


TBPlaS has been developed and tested on Linux. Ideally, it should work on most Linux distributions.
As examples, we have successfully installed and run TBPlaS on CentOS 7.7, openSUSE Leap 15.3 and
Ubuntu 20.04 LTS.

.. rubric:: Windows

Due to compiler and library compatibility issues, native compilation of TBPlaS on Windows is particularly
difficult. So, if you wish to install TBPlaS on Windows, the best practice is to use virtual machines,
e.g., VirtualBox or VMWare. Windows Subsystem for Linux (WSL) is also an option. If you insist on native
compilation, and if you are an expert on computer, follow the guidelines
`cython on windows <https://stackoverflow.com/questions/52864588/how-to-install-cython-an-anaconda-64-bits-with-windows-10>`_
and
`f2py on windows <https://stackoverflow.com/questions/48826283/compile-fortran-module-with-f2py-and-python-3-6-on-windows-10>`_

.. rubric:: MacOS

Unfortunately, we have no experience of MacOS. You are encouraged to send us feedbacks on this operating system.

Compilers and math libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The performance-critical parts of TBPlaS are written in C and FORTRAN. Also, sparse matrices are utilized to
reduce memory cost. So, you need both C and FORTRAN compilers to compile TBPlaS, and link it to vendor-provided
math libraries for optimal performance. For Intel CPUs, it is recommended to use Intel compilers and Math Kernel
Library (MKL), which are now bundled as oneAPI. If the Intel toolchain is not available, GNU Compiler Collection
(GCC) is a general choice. In that case, built-in sparse matrix library will be enabled automatically.

As examples, we have successfully compiled TBPlaS with Intel Compiler 2019.0/2022.0 and GCC 6.3/7.5. Old versions
may still work, although not recommended.


Python
^^^^^^

TBPlaS requires Python 3.6.x or newer. Python 2 is not supported.

In addition to the Python interpreter, development headers are also required. The actual package name of
development headers may vary among Linux distributions, e.g., python3-devel on rpm-based distributions such
as RedHat/CentOS/Fedora/openSUSE, and python3-dev on deb-based distributions like Debian/Ubuntu. Check the
manual of your distribution’s package manager (``yum`` / ``dnf`` / ``zypper`` / ``apt-get``) for guidelines
on searching and installing the package.

If you are a newbie to Linux, or if you do not have root privileges, or if you are working with a computer
without Internet access, try the `Anaconda <https://www.anaconda.com/products/individual>`_ offline installer,
which bundles the development headers as well as other dependencies.

Dependencies
^^^^^^^^^^^^

TBPlaS requires the following Python packages as mandatory dependencies:
    * NumPy >= 1.19.2
    * SciPy >= 1.5.2
    * Matplotlib >= 3.3.2
    * Cython >= 0.29.21
    * Setupools >=50.3.1

And the following packages as optional:
    * ASE >=3.22.1 (for LAMMPS interface)
    * MPI4PY >= 3.0.3 (for hybrid MPI+OpenMP parallelization)

Old versions may still work, but not recommended.

Most of the dependencies can be installed via the pip command, e.g., ``python -m pip install numpy``.
If you do not have root privileges, add ``--user`` option to install into you home directory, e.g.,
``python -m pip install --user numpy``. If your computer has no Internet access, try the
`Anaconda <https://www.anaconda.com/products/individual>`_ offline installer.

The installation of MPI4PY is more complex. You need a working MPI implementation. Since OpenMPI
suffers from a limitation on the number of OpenMP threads when there are too few MPI processes,
MPICH3 or newer is preferred. See the following links for installing
`MPICH <https://www.mpich.org/documentation/guides/>`_,
`OpenMPI <https://www.open-mpi.org//faq/?category=building>`_,
and `MPI4PY <https://mpi4py.readthedocs.io/en/stable/install.html>`_.

Configuration
-------------

General rules
^^^^^^^^^^^^^

The configuration of compilation is stored in ``setup.cfg`` in the top directory of TBPlaS source code.
Some examples are placed under ``config`` subdirectory. A common ``setyp.cfg`` consists of the following
sections:

    * config_cc: C compiler configuration
    * config_fc: FORTRAN compiler configuration
    * build_ext: external library configuration
  
You should adjust these settings according to your computer's hardware and software environment.
Here is an example using Intel compilers and built-in sparse matrix library:

.. code-block:: cfg
    :emphasize-lines: 0

    [config_cc]                                                                                                                                                                             
    compiler = intelem

    [config_fc]
    fcompiler = intelem
    arch = -xHost
    opt = -qopenmp -O3 -ipo -heap-arrays 32
    f90flags = -fpp


And here is the example using Intel compilers and MKL:

.. code-block:: cfg
    :emphasize-lines: 0

    [config_cc]                                                                                                                                                                             
    compiler = intelem

    [config_fc]
    fcompiler = intelem
    arch = -xHost
    opt = -qopenmp -O3 -ipo -heap-arrays 32
    f90flags = -fpp -DMKL

    [build_ext]
    include_dirs = /software/intel/parallelstudio/2019/compilers_and_libraries/linux/mkl/include
    library_dirs = /software/intel/parallelstudio/2019/compilers_and_libraries/linux/mkl/lib/intel64
    libraries = mkl_rt iomp5 pthread m dl

Another example using GCC and built-in sparse matrix library:

.. code-block:: cfg
    :emphasize-lines: 0

    [config_cc]
    compiler = unix

    [config_fc]
    fcompiler = gfortran
    arch = -march=native
    opt = -fopenmp -O3 -flto -mtune=native
    f90flags = -fno-second-underscore -cpp

Workaround for undefined symbol error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may run into errors complaining about ``undefined symbol: GOMP_parallel`` when testing your build and
installation. In that case, find the location of libgomp.so, for instance, ``/usr/lib64``. Add it to
``build_ext`` section of ``setup.cfg`` and re-compile TBPlaS. This issue will be solved.

.. code-block:: cfg
    :emphasize-lines: 0

    [build_ext]                                                                                                                                                                             
    library_dirs = /usr/lib64
    libraries = gomp

Compilation
-----------
Once ``setup.cfg`` has been properly configured, you can build TBPlaS with this command: ``python setup.py build``.
If everything goes well, a new ``build`` directory will be created. The C and FORTRAN extensions can be found under
``lib.linux-x86_64-3.x`` sub-directory, with x being the minor version of Python interpreter. If any error occurs,
check ``setup.cfg`` carefully as described in previous sections.

Installation
------------

TBPlaS can be installed to the default path, user-specified path, or kept in the source code directory. Installin
into the default path is the simplest way, since it does not involve setting up environment variables. However,
it is difficult to keep multiple versions or to update TBPlaS in that approach. Installing into user-specified
path solves this problem, yet it requires appending a **long** path to environment variables. Keeping the source
code simplifies the environment setting process, and offers the access to source code if necessary. So, personally,
we suggest keeping the source code directory.

Installing into default path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installing TBPlaS into the default path is as easy as ``python setup.py install``. After installation you can invoke
Python and try ``import tbplas``. If no error occurs, then your installation is successful. If there are errors on
undefined symbol, check the workaround in previous section.

Installing into user-specified path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installing into user-specified path is achieved by adding ``--prefix`` option to the ``install`` command. For example,
``python setup.py install --prefix=/home/foo/bar`` will install TBPlaS into the directory of ``/home/foo/bar``.
You must add the follow directory to the ``PYTHONPATH`` environment variable:

.. code-block:: shell
    :emphasize-lines: 0

    export PYTHONPATH=/home/foo/bar/lib/python3.6/site-packages:$PYTHONPATH

or

.. code-block:: shell
    :emphasize-lines: 0

    export PYTHONPATH=/home/foo/bar/lib/python3.8/site-packages/tbplas-0.9.8-py3.8-linux-x86_64.egg:$PYTHONPATH

depending on yourpython environment. Anyway, the tbplas sub-directory must reside under the directory you add to
``PYTHONPATH``. You can also add this command into your ``~/.bashrc`` to make it permanently effective, i.e.,
you will not need to type it every time you log in or open a new terminal.

Keeping TBPlaS in the source code directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep TBPlaS in the source code directory you need to manually copy C/FORTRAN extensions from build directory to
proper destinations:

.. code-block:: shell
    :emphasize-lines: 0

    cp build/lib.linux-x86_64-3.8/tbplas/builder/core.cpython-38-x86_64-linux-gnu.so tbplas/builder
    cp build/lib.linux-x86_64-3.8/tbplas/fortran/f2py.cpython-38-x86_64-linux-gnu.so tbplas/fortran

Note the actual locations and names of the extensions depends on the version of your Python interpreter. Then add
source code directory to PYTHONPATH. For instance,

.. code-block:: shell
    :emphasize-lines: 0

    export PYTHONPATH=/home/foo/bar/tbplas_src:$PYTHONPATH

with ``tbplas_src`` being the source code directory, in which ``setup.py`` and other files reside. Also, do not forget
to add this command to your ``~/.bashrc`` to make it permanently effective.

Testing
-------

There are some testing scripts under tests directory of source code. You can test your compilation and installation
by invoking these scripts, e.g., ``python test_core.py``. Some output will be printed to the screen and some figures
will be saved to disk. If everything goes well, a notice will be raised saying all the tests have been passed by the
end of each script.
