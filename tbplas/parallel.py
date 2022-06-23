"""
Functions and classes for running TBPlaS in parallel mode via MPI.

Functions
---------
    None

Classes
-------
    MPIEnv: developer class
        wrapper over MPI4PY APIs
"""

import os

import numpy as np

from .utils import split_list, split_range


class MPIEnv:
    """
    Wrapper over MPI4PY APIs.

    Attributes
    ----------
    mpi: module
        MPI module of mpi4py package
    comm: instance of 'mpi4py.MPI.Intracomm' class
        default global mpi communicator
    rank: integer
        id of this process in mpi communicator
    size: integer
        number of processes in mpi communicator
    """
    def __init__(self, enable_mpi=True, echo_details=True) -> None:
        """
        :param enable_mpi: boolean
            whether to enable parallelization using MPI
        :param echo_details: boolean
            whether to report parallelization details
        """
        # Initialize MPI variables
        if enable_mpi:
            from mpi4py import MPI
            self.mpi = MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.mpi = None
            self.comm = None
            self.rank = 0
            self.size = 1

        # Print simulation details
        if echo_details:
            self.print("\nParallelization details:")
            if self.mpi_enabled:
                self.print("%17s:%4d" % ("MPI processes", self.size))
            else:
                self.print("%17s" % "MPI disabled")
            for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                if env_name in os.environ.keys():
                    self.print("%17s:%4s" % (env_name, os.environ[env_name]))
                else:
                    self.print("%17s: n/a" % env_name)
            self.print()

    @property
    def mpi_enabled(self):
        """Determine whether MPI is enabled."""
        return self.comm is not None

    @property
    def is_master(self):
        """Determine whether this is the master process."""
        return self.rank == 0

    @staticmethod
    def __get_array_order(array: np.ndarray):
        """
        Get data order of array.

        NOTE: Order of memory layout is particularly important when using
        MPI. If data_local is returned by FORTRAN subroutines, it should
        in column-major order. Otherwise, it will be in row-major order.
        If mistaken, no errors will be raised, but the results will be weired.

        :param array: numpy array
        :return: string, should be either "C" or "F"
        :raise ValueError: if array is neither C nor FORTRAN contiguous
        """
        if array.flags.c_contiguous:
            order = "C"
        elif array.flags.f_contiguous:
            order = "F"
        else:
            raise ValueError("Array is neither C nor FORTRAN contiguous")
        return order

    def dist_list(self, raw_list, algorithm="range"):
        """
        Distribute a list over processes.

        :param raw_list: list
            raw list to distribute
        :param algorithm: string
            distribution algorithm, should be either "remainder" or "range"
        :return: list: sublist assigned to this process
        """
        return split_list(raw_list, self.size, algorithm)[self.rank]

    def dist_range(self, n_max):
        """
        Distribute range(n_max) over processes.

        :param n_max: int
            upper bound of range
        :return: range: range assigned to this process
        """
        return split_range(n_max, num_group=self.size)[self.rank]

    def reduce(self, data_local):
        """
        Reduce local data to master process.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            summed data from data_local
        """
        if self.mpi_enabled:
            if self.is_master:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
            else:
                data = None
            self.comm.Reduce(data_local, data, op=self.mpi.SUM, root=0)
        else:
            data = data_local
        return data

    def all_reduce(self, data_local):
        """
        Reduce local data and broadcast to all processes.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            summed data from data_local
        """
        if self.mpi_enabled:
            data = np.zeros(data_local.shape, dtype=data_local.dtype,
                            order=self.__get_array_order(data_local))
            self.comm.Allreduce(data_local, data, op=self.mpi.SUM)
        else:
            data = data_local
        return data

    def average(self, data_local):
        """
        Average results over random samples and store results to master process.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            averaged data from data_local
        """
        if self.mpi_enabled:
            if self.is_master:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
            else:
                data = None
            self.comm.Reduce(data_local, data, op=self.mpi.SUM, root=0)
            if self.is_master:
                data /= self.size
        else:
            data = data_local
        return data

    def all_average(self, data_local):
        """
        Average results over random samples broadcast to all process.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            averaged data from data_local
        """
        if self.mpi_enabled:
            data = np.zeros(data_local.shape, dtype=data_local.dtype,
                            order=self.__get_array_order(data_local))
            self.comm.Allreduce(data_local, data, op=self.mpi.SUM)
            data /= self.size
        else:
            data = data_local
        return data

    def barrier(self):
        """Wrapper for self.comm.Barrier."""
        if self.mpi_enabled:
            self.comm.Barrier()

    def print(self, text=""):
        """
        Print text on master process.

        NOTE: flush=True is essential for some MPI implementations,
        e.g. MPICH3.
        """
        if self.is_master:
            print(text, flush=True)
