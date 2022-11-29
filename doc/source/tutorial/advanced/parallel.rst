Parallelization
===============

Notes on parallelization
------------------------

TBPM calculations in TBPLaS are parallelized in a hybrid MPI+OpenMP approach. So we need to know
the parallelism better in order to make the best of our hardware. The most time-demanding part of
a TBPM calculation is evaluation of correlation function, which is obtained from the propagation
of a wave function driven by the Hamiltonian. Averaging over initial wave functions are required
for better accuracy. In TBPLaS, propagation of wave function from different initial conditions are
parallelized over MPI processes, with each process dealing a few initial conditions. The action of
Hamiltonian on the wave function, which is a matrix-vector multiplication mathemtically, is then
parallelized over OpenMP threads. This hybrid parallelism significantly reduces inter-process data
communication and makes better use of CPU cache.

There are two rules for setting the parallelization parameters:

* The number of MPI processes, multiplied by the number of OpenMP threads, should be equal to the
  number of physical cores on the computer, or the number of cores allocated to the job if you are
  using a queuing system like ``SLURM``, ``PBS`` or ``LSF``.

* The number of initial conditions should be a multiple of MPI processes.

For example, if you are going to use 4 initial conditions, and there are 8 cores on your computer.
Then the possible choices of parallelization parameters are:

* 1 MPI process(es) * 8 OpenMP thread(s)
* 2 MPI process(es) * 4 OpenMP thread(s)
* 4 MPI process(es) * 2 OpenMP thread(s)

For better performance, a rough rule is that the number of MPI processes equals to the number of CPU
sockets, while the number of OpenMP threads equals to the number of cores bind to each socket. Most
personal computers have only one socket. Workstations or computational nodes at High Performance
Computer Center may have 2 or 4 sockets. Keep in mind that the optimal setting is highly hardware
dependent, and can only be determined by extensive tests.

Note that if your computer has HyperThreading enabled in BIOS or UEFI, then the number of available
cores will be double of the physical cores. DO NOT use the virtual cores from HyperThreading since
there will be significant performance loss. Check the handbook of your CPU for the number of physical
cores.

MPI+OpenMP parallelization
--------------------------

Finally, we show how to enable hybrid MPI+OpenMP parallelization. The setting up of ``sample`` and
``config`` is the same as pure OpenMP case. The difference is that we need to add the ``enable_mpi``
argument when creating the ``solver`` and ``analyzer``:

.. code-block:: python

    solver = tb.Solver(sample, config, enable_mpi=True)
    analyzer = tb.Analyzer(sample, config, enable_mpi=True)

Evaluation and analysis of correlation function is also the same as pure OpenMP case:

.. code-block:: python

    corr_dos = solver.calc_corr_dos()
    energies_dos, dos = analyzer.calc_dos(corr_dos)

However, we shall plot the results on master process only, in order to avoid conflicts:

.. code-block:: python

    if analyzer.is_master:
        plt.plot(energies_dos, dos)
        plt.xlabel("E (eV)")
        plt.ylabel("DOS")
        plt.savefig("DOS.png")
        plt.close()

We will use 4 MPI processes for the calculation. So ``OMP_NUM_THREADS`` should be set to 2:

.. code-block:: bash

    export OMP_NUM_THREADS=2

Supposing that the python script is saved to ``tbpm.py``, we can run the job as:

.. code-block:: bash

    mpirun -np 4 python ./tbpm.py

The results should be the same as pure OpenMP case. If you are using a super computer with queuing
system like ``SLURM``, ``PBS`` or ``LSF``, then you need another batch script for submitting the
job. Contact the administrator of the super computer for help on preparing the script.

Here we provide two batch scripts for the ``SLURM`` queing system as examples. ``SLURM`` has the
following options for specifying parallelization details:

* nodes: number of nodes for the job
* ntasks-per-node: number of MPI processes to spawn on each node
* cpus-per-task: number of OpenMP threads for each MPI process

Suppose that we are going to use 4 initial conditions and 1 node. The node has 2 sockets with 8 cores
per socket. The number of MPI processes should be either 1, 2, 4, and the number of OpenMP threads is
16, 8, 4, respectively. We will use 2 processes * 8 threads. The batch script is as following:

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
