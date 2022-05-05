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

from .utils import split_list


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

    def average(self, data_local):
        """
        Average results over random samples and store results to master process.

        :param data_local: numpy array
            local results on each process
        :return: data: numpy array
            averaged data from data_local

        NOTE: Order of memory layout is particularly important when using
        MPI. As data_local is returned by FORTRAN subroutines, it should
        in column-major order. Otherwise no errors will be casted, but the
        results will be weired. As numpy uses row-major order by default, DO
        NOT remove the order=F argument below.
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

    def barrier(self):
        """Wrapper for self.comm.Barrier."""
        self.comm.Barrier()
