"""Helper classes and functions for visualizing models."""

from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc

from .base import rn3_type, pos3_type


class ModelViewer:
    """
    Class for plotting the model.

    Attributes
    ----------
    _axes: matplotlib 'axes' instance
        axes on which to plot the model
    _lat_vec: (3, 3) float64 array
        Cartesian coordinates of model lattice vectors
    _origin: (3,) float64 array
        Cartesian coordinate of model lattice origin
    _view: str
       kind of view, should be in 'ab', 'bc', 'ca', 'ba', 'cb' and 'ac'
    _dim_x, _dim_y, _dim_z: int
        column indices of x, y, z components of current view for projection
    _hop_lc: List[Tuple[np.ndarray, np.ndarray]]
        line collection for hopping terms
    _cell_lc: List[Tuple[np.ndarray, np.ndarray]]
        line collections for cells
    """
    def __init__(self, axes: plt.axes,
                 lat_vec: np.ndarray,
                 origin: np.ndarray,
                 view: str = "ab") -> None:
        """
        :param axes: axis on which to plot the model
        :param lat_vec: (3, 3) float64 array
            Cartesian coordinates of model lattice vectors
        :param origin: (3,) float64 array
            Cartesian coordinate of model lattice origin
        :param view: kind of view, should be in 'ab', 'bc', 'ca', 'ba', 'cb'
            and 'ac'
        :raises ValueError: if view is illegal
        """
        self._axes = axes
        self._lat_vec = lat_vec
        self._origin = origin
        if view not in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac'):
            raise ValueError(f"Illegal view {view}")
        self._view = view
        dim_dict = {'a': 0, 'b': 1, 'c': 2}
        self._dim_x, self._dim_y = dim_dict[view[0]], dim_dict[view[1]]
        self._dim_z = list({0, 1, 2}.difference({self._dim_x, self._dim_y}))[0]
        self._hop_lc = []
        self._cell_lc = []

    def _proj_coord(self, coord: np.ndarray) -> np.ndarray:
        """
        Project 3-dimensional coordinates to 2d.

        :param coord: (num_row, 3) float64 array
            coordinates for projection
        :return: (num_row, 2) or (2,) float64 array
            projected coordinates
        """
        if len(coord.shape) == 1:
            projected_coord = coord[[self._dim_x, self._dim_y]]
        else:
            projected_coord = coord[:, [self._dim_x, self._dim_y]]
        return projected_coord

    def _restore_coord(self, a: Union[int, float],
                       b: Union[int, float]) -> Union[rn3_type, pos3_type]:
        """
        Restore 2-dimensional coordinates to 3d.

        :param a: 0th component of the coordinate
        :param b: 1st component of the coordinate
        :return: restored coordinate
        """
        if self._dim_z == 0:
            return 0, a, b
        elif self._dim_z == 1:
            return a, 0, b
        else:
            return a, b, 0

    def scatter(self, coord: np.ndarray, **kwargs) -> None:
        """
        Scatter plot for showing orbitals.

        :param coord: (num_orb, 3) float64 array
            Cartesian coordinates of orbitals
        :param kwargs: keyword arguments for axes.scatter()
        :return: None
        """
        coord = self._proj_coord(coord)
        self._axes.scatter(coord[:, 0], coord[:, 1], **kwargs)

    def plot_arrow(self, coord_i: np.ndarray,
                   coord_j: np.ndarray,
                   **kwargs) -> None:
        """
        Plot an arrow between given points.

        :param coord_i: (3,) float64 array
            Cartesian coordinates of starting point
        :param coord_j: (3,) float64 array
            Cartesian coordinates of ending point
        :param kwargs: keyword arguments for axes.arrow()
        :return: None
        """
        coord_i = self._proj_coord(coord_i)
        coord_j = self._proj_coord(coord_j)
        diff = coord_j - coord_i
        self._axes.arrow(coord_i[0], coord_i[1], diff[0], diff[1], **kwargs)

    def add_line(self, coord_i: np.ndarray, coord_j: np.ndarray) -> None:
        """
        Add a line connecting give points.

        :param coord_i: (3,) float64 array
            Cartesian coordinates of starting point
        :param coord_j: (3,) float64 array
            Cartesian coordinates of ending point
        :return: None
        """
        coord_i = self._proj_coord(coord_i)
        coord_j = self._proj_coord(coord_j)
        self._hop_lc.append((coord_i, coord_j))

    def plot_line(self, **kwargs) -> None:
        """
        Plot lines in self.hop_lc.

        :param kwargs: keyword arguments for mc.LineCollection()
        :return: None
        """
        self._axes.add_collection(mc.LineCollection(self._hop_lc, **kwargs))

    def add_grid(self, a_min: int, a_max: int, b_min: int, b_max: int) -> None:
        """
        Add grid on the range of [a_min, a_max] * [b_min, b_max].

        :param a_min: lower bound of range on a-axis
        :param a_max: upper bound of range on a-axis
        :param b_min: lower bound of range on b-axis
        :param b_max: upper bound of range on b-axis
        :return: None
        """
        for ia in range(a_min, a_max + 1):
            x0 = self._restore_coord(ia, b_min)
            x1 = self._restore_coord(ia, b_max)
            x0 = self._proj_coord(np.matmul(x0, self._lat_vec) + self._origin)
            x1 = self._proj_coord(np.matmul(x1, self._lat_vec) + self._origin)
            self._cell_lc.append((x0, x1))
        for ib in range(b_min, b_max + 1):
            x0 = self._restore_coord(a_min, ib)
            x1 = self._restore_coord(a_max, ib)
            x0 = self._proj_coord(np.matmul(x0, self._lat_vec) + self._origin)
            x1 = self._proj_coord(np.matmul(x1, self._lat_vec) + self._origin)
            self._cell_lc.append((x0, x1))

    def plot_grid(self, **kwargs) -> None:
        """
        Plot grid in self.cell_lc.

        :param kwargs: keyword arguments for mc.LineCollection()
        :return:
        """
        self._axes.add_collection(mc.LineCollection(self._cell_lc, **kwargs))

    def plot_lat_vec(self, **kwargs) -> None:
        """
        Plot lattice vectors of (0, 0, 0)-th primitive cell as arrows.

        :param kwargs: keyword arguments for axes.arrow()
        :return: None
        """
        x0 = self._proj_coord(self._origin)
        for x1 in ((0, 1), (1, 0)):
            x1 = self._restore_coord(x1[0], x1[1])
            x1 = self._proj_coord(np.matmul(x1, self._lat_vec) + self._origin)
            diff = x1 - x0
            self._axes.arrow(x0[0], x0[1], diff[0], diff[1], **kwargs)
