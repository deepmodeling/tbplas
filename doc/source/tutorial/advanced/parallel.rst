Parallelization
===============


TBPLaS features a hybrid MPI+OpenMP parallelization for  the evaluation of band structure and DOS
from exact-diagonalization, response properties from Lindhard function, topological invariant
:math:`\mathbb{Z}_2` and TBPM calculations. Both MPI and OpenMP can be switched on/off separately
on demand, while pure OpenMP mode is enabled by default.

The number of OpenMP threads is controlled by the ``OMP_NUM_THREADS`` environment variable. If
TBPLaS has been compiled with MKL support, then the ``MKL_NUM_THREADS`` environment variable will
also take effect. If none of the environment variables has been set, OpenMP will make use of all
the CPU cores on the computing node. To switch off OpenMP, set the environment variables to 1.
On the contrary, MPI-based parallelization is disabled by default, but can be easily enabled with
a single option. The ``calc_bands`` and ``calc_dos`` functions of :class:`.PrimitiveCell` and
:class:`.Sample` classes, the initialization functions of :class:`.Lindhard`, :class:`.Z2`,
:class:`.Solver` and :class:`.Analyzer` classes all accept an argument named ``enable_mpi`` whose
default value is taken to be ``False``. If set to ``True``, MPI-based parallelization is turned on,
provided that the ``MPI4PY`` package has been installed. Hybrid MPI+OpenMP parallelization is
achieved by enabling MPI and OpenMP simultaneously. The number of processes is controlled by the
MPI launcher, which receives arguments from the command line, environment variables or
configuration file. The user is recommended to check the manual of job queuing system on the
computer for properly setting the environment variables and invoking the MPI launcher. For
computers without a queuing system, e.g., laptops, desktops and standalone workstations, the MPI
launcher should be ``mpirun`` or ``mpiexec``, while the number of processes is controlled by the
``-np`` command line option.

The optimal parallelization configuration, i.e., the numbers of MPI processes and OpenMP threads,
depend on the hardware, the model size and the type of calculation. Generally speaking, matrix
diagonalization for a single :math:`\mathbf{k}`-point is poorly parallelized over threads. But the
diagonalization for multiple :math:`\mathbf{k}`-points can be efficiently parallelized over
processes. Therefore, for band structure and DOS calculations, as well as response properties from
Lindhard function and topological invariant from Z2, it is recommended to run in pure MPI-mode by
setting the number of MPI processes to the total number of allocated CPU cores and the number of
OpenMP threads to 1. However, MPI-based parallelization uses more RAM since every process has to
keep a copy of the wave functions and energies. So, if the available RAM imposes a limit, try to
use less processes and more threads. Anyway, the product of the numbers of processes and threads
should be equal to the number of allocated CPU cores. For example, if you have allocated 16 cores,
then you can try 16 processes :math:`\times` 1 thread, 8 processes :math:`\times` 2 threads,
4 processes :math:`\times` 4 threads, etc. For TBPM calculations, the number of random initial wave
functions should be divisible by the number of processes. For example, if you are going to consider
16 initial wave functions, then the number of processes should be 1, 2, 4, 8, or 16. The number of
threads should be set according to the number of processes. Again, if the RAM size is a problem,
try to decrease the number of processes and increase the number of threads. Note that if your
computer has HyperThreading enabled in BIOS or UEFI, then the number of available cores will be
double of the physical cores. DO NOT use the virtual cores from HyperThreading since there will be
significant performance loss. Check the handbook of your CPU for the number of physical cores.

If MPI-based parallelization is enabled, either in pure MPI or hybrid MPI+OpenMP mode, special care
should be taken to output and plotting part of the job script. These operations should be performed
on the master process only, otherwise the output will mess up or files get corrupted, since all the
processes will try to modify the same file or plotting the same data. This situation is avoided by
checking the rank of the process before action. The :class:`.Lindhard`, :class:`.Z2`,
:class:`.Solver`, :class:`.Analyzer` and :class:`.Visualizer` classes all offer an ``is_master``
attribute to detect the master process, whose usage will be demonstrated in the following sections.

Last but not least, we have to mention that all the calculations in previous tutorials can be run
in either interactive or batch mode. You can input the script line-by-line in the terminal, or save
it to a file and pass the file to the Python interpreter. However, MPI-based parallelization
supports only the batch mode, since there is no possibility to input anything in the terminal for
multiple processes in one time. In the following sections, we assume the script file to be
``test_mpi.py``. A common head block of the script is given in :ref:`para_bands` and will not be
explicitly repeated in subsequent sections.

.. _para_bands:

Band structure and DOS
----------------------

We demonstrate the usage of ``calc_bands`` and ``calc_dos`` in parallel mode by calculating the
band structure and DOS of a :math:`12\times12\times1` graphene sample. Procedure shown here is also
valid for the primitive cell. To enable MPI-based parallelization, we need to save the script to a
file, for instance, ``test_mpi.py``. The head block of this file should be:

.. code-block:: python

    #! /usr/bin/env python

    import numpy as np
    import tbplas as tb


    timer = tb.Timer()
    vis = tb.Visualizer(enable_mpi=True)

where the first line is a magic line declaring that the script should be interpreted by the Python
program. In the following lines we import the necessary packages. To record and report the time
usage, we need to create a timer from the :class:`.Timer` class. We also need a visualizer for
plotting the results, where the ``enable_mpi`` argument is set to ``True`` during initialization.
This head block also is essential for other examples in subsequent sections.

For convenience, we will not build the primitive cell from scratch, but import it from the materia
repository with the :func:`.make_graphene_diamond` function:

.. code-block:: python

    cell = tb.make_graphene_diamond()

Then we build the sample by:

.. code-block:: python

    sample = tb.Sample(tb.SuperCell(cell, dim=(12, 12, 1), pbc=(True, True, False)))

The evaluation of band structure in parallel mode is similar to the serial mode, which also
involves generating the :math:`\mathbf{k}`-path and calling ``calc_bands``. The only difference is
that we need to set the ``enable_mpi`` argument to ``True`` when calling ``calc_bands``:

.. code-block:: python

    k_points = np.array([
        [0.0, 0.0, 0.0],
        [2./3, 1./3, 0.0],
        [1./2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    timer.tic("band")
    k_len, bands = sample.calc_bands(k_path, enable_mpi=True)
    timer.toc("band")
    vis.plot_bands(k_len, bands, k_idx, k_label)
    if vis.is_master:
        timer.report_total_time()

The ``tic`` and ``toc`` functions begin and end the recording of time usage, which receive a string
as the argument for tagging the record. The visualizer is aware of the parallel environment, so no
special treatment is needed when plotting the results. Finally, the time usage is reported with the
``report_total_time`` function on the master process only, by checking the ``is_master`` attribute
of the visualizer.

We run ``test_mpi.py`` by:

.. code-block:: bash

    $ export OMP_NUM_THREADS=1
    $ mpirun -np 1 ./test_mpi.py

With the environment variable ``OMP_NUM_THREADS`` set to 1, the script will run in pure MPI-mode.
We invoke 1 MPI process by the ``-np`` option of the MPI launcher ``mpirun``. The output should
look like:

.. code-block:: bash

    band :      11.03s

So, the evaluation of bands takes 11.03 seconds on 1 process. We try with more processes:

.. code-block:: bash

    $ mpirun -np 2 ./test_mpi.py
        band :       5.71s
    $ mpirun -np 4 ./test_mpi.py
        band :       2.93s

Obviously, the time usage scales reversely with the number of processes. Detailed discussion on the
time usage and speedup under different parallelization configurations will be discussed in ref. 4
of :ref:`background`.

Evaluation of DOS can be parallelized in the same approach, by setting the ``enable_mpi`` argument
to ``True``:

.. code-block:: python

    k_mesh = tb.gen_kmesh((20, 20, 1))
    timer.tic("dos")
    energies, dos = sample.calc_dos(k_mesh, enable_mpi=True)
    timer.toc("dos")
    vis.plot_dos(energies, dos)
    if vis.is_master:
        timer.report_total_time()

The script can be run in the same approach as evaluating the band structure.


Response properties from Lindhard function
------------------------------------------

To evaluate response properties in parallel mode, simply set the ``enable_mpi`` argument to
``True`` when creating the Lindhard calculator:

.. code-block:: python

    lind = tb.Lindhard(cell=cell, energy_max=10.0, energy_step=2048,
                       kmesh_size=(600, 600, 1), mu=0.0, temperature=300.0, g_s=2,
                       back_epsilon=1.0, dimension=2, enable_mpi=True)

Subsequent calls to the functions of :class:`.Lindhard` class does not need further special
treatment. For example, the optical conductivity can be evaluated in the same approach as in serial
mode:

.. code-block:: python

    timer.tic("ac_cond")
    omegas, ac_cond = lind.calc_ac_cond(component="xx")
    timer.toc("ac_cond")
    vis.plot_xy(omegas, ac_cond)
    if vis.is_master:
        timer.report_total_time()


Topological invariant from Z2
-----------------------------

The evaluation of phases :math:`\theta_m^D` can be paralleled in the same approach as response
functions:

.. code-block:: python

    z2 = tb.Z2(cell, num_occ=10, enable_mpi=True)
    timer.tic("z2")
    kb_array, phases = z2.calc_phases(ka_array, kb_array, kc)
    timer.toc("z2")
    vis.plot_phases(kb_array, phases / pi)
    if vis.is_master:
        timer.report_total_time()

where we only need to set ``enable_mpi`` argument to ``True`` when creating the :class:`.Z2`
instance.


Properties from TBPM
--------------------

TBPM calculations in parallel mode are similar to the evaluation of response functions. The user
only needs to set the ``enable_mpi`` argument to ``True``. To make the time usage noticeable, we
build a larger sample first:

.. code-block:: python

    sample = tb.Sample(tb.SuperCell(cell, dim=(240, 240, 1), pbc=(True, True, False)))

Then we create the configuration, solver and analyzer, with the argument ``enable_mpi=True``:

.. code-block:: python

    sample.rescale_ham(9.0)
    config = tb.Config()
    config.generic["nr_random_samples"] = 4
    config.generic["nr_time_steps"] = 256
    solver = tb.Solver(sample, config, enable_mpi=True)
    analyzer = tb.Analyzer(sample, config, enable_mpi=True)

Correlation function can be obtained and analyzed in the same way as in serial mode:

.. code-block:: python

    timer.tic("corr_dos")
    corr_dos = solver.calc_corr_dos()
    timer.toc("corr_dos")
    energies, dos = analyzer.calc_dos(corr_dos)
    vis.plot_dos(energies, dos)
    if vis.is_master:
        timer.report_total_time()


Example scripts for SLURM
-------------------------

If you are using a super computer with queuing system like ``SLURM``, ``PBS`` or ``LSF``, then you
need another batch script for submitting the job. Contact the administrator of the super computer
for help on preparing the script.

Here we provide two batch scripts for the ``SLURM`` queing system as examples. ``SLURM`` has the
following options for specifying parallelization details:

* nodes: number of nodes for the job
* ntasks-per-node: number of MPI processes to spawn on each node
* cpus-per-task: number of OpenMP threads for each MPI process

Suppose that we are going to use 4 initial conditions and 1 node. The node has 2 CPUs with 8
cores per CPU. The number of MPI processes should be either 1, 2, 4, and the number of OpenMP
threads is 16, 8, 4, respectively. We will use 2 processes * 8 threads. The batch script is as
following:

.. code-block:: bash

    #! /bin/bash
    #SBATCH --account=alice
    #SBATCH --partition=hpib
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=8
    #SBATCH --job-name=test_mpi
    #SBATCH --time=24:00:00
    #SBATCH --output=slurm-%j.out
    #SBATCH --error=slurm-%j.err

    # Load modules
    module load mpi4py tbplas

    # Set number of threads
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

    # Change to working directory and run the job
    cd $SLURM_SUBMIT_DIR
    srun --mpi=pmi2 python ./test_mpi.py

Here we assume the user name to be ``alice``, and we are submitting to the ``hpib`` partition.
Since we are going to use 1 node, we set ``nodes`` to 1. For each node 2 MPI processes will be
spawned, so ``ntasks-per-node`` is set to 2. There are 16 physical cores on the node, so
``cpus-per-task`` is set to 8.

If you want pure OpenMP parallelization, here is another example:

.. code-block:: bash

    #! /bin/bash
    #SBATCH --account=alice
    #SBATCH --partition=hpib
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=16
    #SBATCH --job-name=test_omp
    #SBATCH --time=24:00:00
    #SBATCH --output=slurm-%j.out
    #SBATCH --error=slurm-%j.err

    # Load modules
    module load tbplas

    # Set number of threads
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

    # Change to working directory and run the job
    cd $SLURM_SUBMIT_DIR
    srun python ./test_omp.py

In this script the number of processes is set to 1, and the number of threads per process is set to
the total number of physical cores. Don't forget to remove ``enable_mpi=True`` when creating the
solver and analyzer, in order to skip unnecessary MPI initialization.
