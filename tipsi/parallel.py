"""parallel.py contains the definition for MPIEnv, a thin wrapper around MPI4PY
for running tipsi parallely via MPI.

Currently supported tasks:
corr_DOS, corr_LDOS, corr_AC, corr_dyn_pol, corr_DC, mu_Hall, quasi_eigenstates

Classes
-------
    MPIEnv
        MPI environment
"""

import os
from mpi4py import MPI
import numpy as np
import tipsi


class MPIEnv(object):
    """
    Wrapper around MPI4PY APIs.

    Attributes
    ----------
    comm: mpi communicator
    rank: integer
        id of this process in mpi communicator
    size: integer
        number of processes in mpi communicator
    """
    def __init__(self) -> None:
        super().__init__()

        # Initialize MPI environment
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Print simulation details
        self.print("\nParallelization details:")
        self.print("%17s:%4d" % ("MPI processes", self.size))
        for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if env_name in os.environ.keys():
                self.print("%17s:%4s" % (env_name, os.environ[env_name]))
            else:
                self.print("%17s: n/a" % env_name)
        self.print()

    def print(self, text=""):
        """Print text on master process"""
        # NOTE: flush=True is essential for some MPI implementations, e.g.
        # MPICH3.
        if self.rank == 0:
            print(text, flush=True)

    def is_master(self):
        """
        Returns whether this process is the master process or not.

        Post-process (analysis) should be performed on the master process.
        Otherwise, output files will be corrupted. So we need to call this
        function to detect the master process. Although checking if 
        self.rank == 0 outside the class does the same work, we do not think
        it a good idea. So we made this wrapper function.
        """
        return self.rank == 0

    def __all_average(self, data_local):
        """
        Average results over initial random samples and then broadcast to all
        processes.

        Parameters
        ----------
        data_local: numpy array, shape and dtype dependent on calculation
            local results on each process

        Returns
        -------
        data: numpy array, with same shape and dtype as data_local
            averaged results
        """
        # NOTE: Order of memory layout is particularly important when using
        # MPI. As data_local is returned by FORTRAN subroutines, it should
        # in column-major order. Otherwise no errors will be casted, but the
        # results will be weired. As numpy uses row-major order, so DO NOT
        # remove the order=F argument below.
        data = np.zeros(data_local.shape, dtype=data_local.dtype, order="F")
        self.comm.Allreduce(data_local, data, op=MPI.SUM)
        data /= self.size
        return data

    def run_tipsi(self, sample, config, func_name):
        """
        A unified interface for running Tipsi via MPI parallelism.

        Supported functions:
        corr_DOS, corr_LDOS, corr_AC, corr_dyn_pol, corr_DC, mu_Hall,
        quasi_eigenstates

        Parameters
        ----------
        sample: instance of tipsi Sample class
        config: instance of tipsi Config class
        func_name: string, name of functions defined in tipsi.correlation
    
        Returns
        -------
        numpy arrays, dependent on func_name
        """
        # Determine the function and number of returned values
        func_dict = dict()
        func_dict["corr_DOS"] = (tipsi.corr_DOS, 1)
        func_dict["corr_LDOS"] = (tipsi.corr_LDOS, 1)
        func_dict["corr_AC"] = (tipsi.corr_AC, 1)
        func_dict["corr_dyn_pol"] = (tipsi.corr_dyn_pol, 1)
        func_dict["corr_DC"] = (tipsi.corr_DC, 2)
        func_dict["mu_Hall"] = (tipsi.mu_Hall, 1)
        func_dict["quasi_eigenstates"] = (tipsi.quasi_eigenstates, 1)
        try:
            func, num_return = func_dict[func_name]
        except KeyError:
            raise ValueError("Illegal func_name '%s'" % func_name)

        # Adjust attributes of config for parallelrun
        num_sample_bak = num_sample = config.generic["nr_random_samples"]
        while num_sample % self.size != 0:
            num_sample += 1
        if num_sample != config.generic["nr_random_samples"]:
            self.print("\nWARNING:\n  nr_random_samples adjusted to %d for"
                       " optimal balance" % num_sample)
        config.generic['nr_random_samples'] = num_sample / self.size

        # Run jobs assigned to this process
        # NOTE: the master process has additional I/O tasks such as creating
        # directories and saving config to disk. On some file systems, e.g.
        # btrfs, creating directories may be slow. Other processes may try to
        # access the directory before the creation finishes and run into I/O
        # error. So we need to put a barrier here.
        self.comm.Barrier()
        if num_return == 1:
            data1_local = func(sample, config)
        else:
            data1_local, data2_local = func(sample, config)

        # Collect data and average
        data1 = self.__all_average(data1_local)
        if num_return == 2:
            data2 = self.__all_average(data2_local)

        # Restore attributes of config and return
        config.generic['nr_random_samples'] = num_sample_bak
        if num_return == 1:
            return data1
        else:
            return data1, data2
