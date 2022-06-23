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

from mpi4py import MPI
import numpy as np

from .utils import split_list, split_range


class MPIEnv:
    """
    Wrapper over MPI4PY APIs.

    Attributes
    ----------
    comm: instance of 'mpi4py.MPI.Intracomm' class
        default global mpi communicator
    rank: integer
        id of this process in mpi communicator
    size: integer
        number of processes in mpi communicator
    """
    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

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
        if self.rank == 0:
            data = np.zeros(data_local.shape, dtype=data_local.dtype,
                            order=self.__get_array_order(data_local))
        else:
            data = None
        self.comm.Reduce(data_local, data, op=MPI.SUM, root=0)
        return data

    def all_reduce(self, data_local):
        """
        Reduce local data and broadcast to all processes.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            summed data from data_local
        """
        data = np.zeros(data_local.shape, dtype=data_local.dtype,
                        order=self.__get_array_order(data_local))
        self.comm.Allreduce(data_local, data, op=MPI.SUM)
        return data

    def average(self, data_local):
        """
        Average results over random samples and store results to master process.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            averaged data from data_local
        """
        if self.rank == 0:
            data = np.zeros(data_local.shape, dtype=data_local.dtype,
                            order=self.__get_array_order(data_local))
        else:
            data = None
        self.comm.Reduce(data_local, data, op=MPI.SUM, root=0)
        if self.rank == 0:
            data /= self.size
        return data

    def all_average(self, data_local):
        """
        Average results over random samples broadcast to all process.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            averaged data from data_local
        """
        data = np.zeros(data_local.shape, dtype=data_local.dtype,
                        order=self.__get_array_order(data_local))
        self.comm.Allreduce(data_local, data, op=MPI.SUM)
        data /= self.size
        return data

    def barrier(self):
        """Wrapper for self.comm.Barrier."""
        self.comm.Barrier()
