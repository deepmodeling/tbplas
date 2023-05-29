"""Functions and classes for the parallel environment."""

import os
from typing import Tuple, List, Any

import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from .utils import split_list, split_range, get_datetime


__all__ = ["MPIEnv"]


class MPIEnv:
    """
    Wrapper over MPI4PY APIs.

    Attributes
    ----------
    __comm: 'mpi4py.MPI.Intracomm' class
        default global mpi communicator
    __rank: integer
        id of this process in mpi communicator
    __size: integer
        number of processes in mpi communicator
    """
    def __init__(self, enable_mpi: bool = True,
                 echo_details: bool = False) -> None:
        """
        This class has two usages: as a developer class for parallel jobs
        or as a user class for detecting the master process. For the latter
        purpose, the default value of enable_mpi should be True while
        echo_details should be False. DO NOT change them.

        :param enable_mpi: whether to enable parallelization using MPI
        :param echo_details: whether to report parallelization details
        :return: None
        """
        # Initialize MPI variables
        if enable_mpi:
            if MPI is not None:
                self.__comm = MPI.COMM_WORLD
                self.__rank = self.__comm.Get_rank()
                self.__size = self.__comm.Get_size()
            else:
                raise ImportError("MPI4PY cannot be imported")
        else:
            self.__comm = None
            self.__rank = 0
            self.__size = 1

        # Print simulation details
        if echo_details:
            spaces = " " * 2
            self.print("\nParallelization details:")
            if self.mpi_enabled:
                self.print(f"{spaces}{'MPI processes':16s} : {self.__size:<6d}")
            else:
                self.print(f"{spaces}{'MPI disabled':16s}")
            for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                try:
                    env_value = os.environ[env_name]
                except KeyError:
                    env_value = "n/a"
                self.print(f"{spaces}{env_name:16s} : {env_value:<6s}")
            self.print()

    @property
    def mpi_enabled(self) -> bool:
        """Determine whether MPI is enabled."""
        return self.__comm is not None

    @property
    def is_master(self) -> bool:
        """Determine whether this is the master process."""
        return self.__rank == 0

    @staticmethod
    def __get_array_order(array: np.ndarray) -> str:
        """
        Get data order of array.

        NOTE: Order of memory layout is particularly important when using
        MPI. If data_local is returned by FORTRAN subroutines, it should
        in column-major order. Otherwise, it will be in row-major order.
        If mistaken, no errors will be raised, but the results will be weired.

        :param array: incoming numpy array
        :return: whether the array is in C or Fortran order
        :raise ValueError: if array is neither C nor FORTRAN contiguous
        """
        if array.flags.c_contiguous:
            order = "C"
        elif array.flags.f_contiguous:
            order = "F"
        else:
            raise ValueError("Array is neither C nor FORTRAN contiguous")
        return order

    def dist_list(self, raw_list: List[Any],
                  algorithm: str = "range") -> List[Any]:
        """
        Distribute a list over processes.

        :param raw_list: raw list to distribute
        :param algorithm: distribution algorithm, should be either "remainder"
            or "range"
        :return: sublist assigned to this process
        """
        return split_list(raw_list, self.__size, algorithm)[self.__rank]

    def dist_range(self, n_max: int) -> range:
        """
        Distribute range(n_max) over processes.

        :param n_max: upper bound of the range
        :return: subrange assigned to this process
        """
        return split_range(n_max, num_group=self.__size)[self.__rank]

    def dist_bound(self, n_max: int) -> Tuple[int, int]:
        """
        Same as dist_range, but returns the lower and upper bounds.
        Both of the bounds are close, i.e. [i_min, i_max].

        :param n_max: upper bound of range
        :return: lower and upper bounds of subrange assigned to this process
        """
        i_index = self.dist_range(n_max)
        i_min, i_max = min(i_index), max(i_index)
        return i_min, i_max

    def reduce(self, data_local: np.ndarray) -> np.ndarray:
        """
        Reduce local data to master process.

        :param data_local: local results on each process
        :return: summed data from data_local
        """
        if self.mpi_enabled:
            if self.is_master:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
            else:
                data = None
            self.__comm.Reduce(data_local, data, op=MPI.SUM, root=0)
        else:
            data = data_local
        return data

    def all_reduce(self, data_local: np.ndarray) -> np.ndarray:
        """
        Reduce local data and broadcast to all processes.

        :param data_local: local results on each process
        :return: summed data from data_local
        """
        if self.mpi_enabled:
            data = np.zeros(data_local.shape, dtype=data_local.dtype,
                            order=self.__get_array_order(data_local))
            self.__comm.Allreduce(data_local, data, op=MPI.SUM)
        else:
            data = data_local
        return data

    def average(self, data_local: np.ndarray) -> np.ndarray:
        """
        Average results over random samples and store results to master process.

        :param data_local: local results on each process
        :return: averaged data from data_local
        """
        if self.mpi_enabled:
            if self.is_master:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
            else:
                data = None
            self.__comm.Reduce(data_local, data, op=MPI.SUM, root=0)
            if self.is_master:
                data /= self.__size
        else:
            data = data_local
        return data

    def all_average(self, data_local: np.ndarray) -> np.ndarray:
        """
        Average results over random samples broadcast to all process.

        :param data_local: local results on each process
        :return: averaged data from data_local
        """
        if self.mpi_enabled:
            data = np.zeros(data_local.shape, dtype=data_local.dtype,
                            order=self.__get_array_order(data_local))
            self.__comm.Allreduce(data_local, data, op=MPI.SUM)
            data /= self.__size
        else:
            data = data_local
        return data

    def bcast(self, data_local: np.ndarray) -> None:
        """
        Broadcast data from master to other processes.

        :param data_local: local results on each process
        :return: None
        """
        if self.mpi_enabled:
            self.__comm.Bcast(data_local, root=0)

    def barrier(self) -> None:
        """Wrapper for self.comm.Barrier."""
        if self.mpi_enabled:
            self.__comm.Barrier()

    def print(self, text: str = "") -> None:
        """
        Print text on master process.

        NOTE: flush=True is essential for some MPI implementations,
        e.g. MPICH3.

        :param text: text to print
        :return: None
        """
        if self.is_master:
            print(text, flush=True)

    def log(self, event: str = "", fmt: str = "%x %X") -> None:
        """
        Log the date and time of event.

        :param event: notice of the event
        :param fmt: date and time format
        :return: None.
        """
        if self.is_master:
            date_time = get_datetime(fmt=fmt)
            print(f"{event} at {date_time}", flush=True)

    @property
    def rank(self) -> int:
        """
        Interface for the '__rank' attribute.

        :return: rank of this MPI process
        """
        return self.__rank

    @property
    def size(self) -> int:
        """
        Interface for the '__size' attribute.

        :return: number of MPI processes
        """
        return self.__size
