"""
Helper classes and functions used among builder package.

Classes
-------
    ModelViewer: developer api
        class for viewing primitive cells and samples

Functions
---------
    None.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc


class ModelViewer:
    """
    Class for showing primitive cells and samples from given view.

    Attributes
    ----------
    axes: instance of matplotlib 'axes' class
        axes on which the cell or sample will be plotted
    lat_vec: (3, 3) float64 array
        lattice vectors of primitive cell
    view: string
       kind of view, should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
    dim_x, dim_y, dim_z: integer
        column indices of x, y, z components of current view for projection
    hop_lc: list
        line collection for hopping terms
    cell_lc: list
        line collections for cells
    """
    def __init__(self, axes: plt.axes, lat_vec: np.ndarray, view="ab"):
        """
        :param axes: instance of matplotlib 'axes' class
            axis on which the cell or sample will be plotted
        :param lat_vec: (3, 3) float64 array
            lattice vectors of primitive cell
        :param view: string
            kind of view, should be in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac')
        :raises ValueError: if view is illegal
        """
        self.axes = axes
        self.lat_vec = lat_vec
        if view not in ('ab', 'bc', 'ca', 'ba', 'cb', 'ac'):
            raise ValueError(f"Illegal view {view}")
        self.view = view
        dim_dict = {'a': 0, 'b': 1, 'c': 2}
        self.dim_x, self.dim_y = dim_dict[view[0]], dim_dict[view[1]]
        self.dim_z = list({0, 1, 2}.difference({self.dim_x, self.dim_y}))[0]
        self.hop_lc = []
        self.cell_lc = []

    def __proj_coord(self, coord: np.ndarray):
        """
        Project 3-dimensional coordinates to 2d.

        :param coord: (num_row, 3) float64 array
            coordinates for projection
        :return: projected_coord (num_row, 2) or (2,) float64 array
            projected coordinates
        """
        if len(coord.shape) == 1:
            projected_coord = coord[[self.dim_x, self.dim_y]]
        else:
            projected_coord = coord[:, [self.dim_x, self.dim_y]]
        return projected_coord

    def __restore_coord(self, a, b):
        """
        Restore 2-dimensional coordinates to 3d.

        :param a: integer or float
            0th component of the coordinate
        :param b: integer or float
            1st component of the coordinate
        :return: coord: tuple of 3 integers of floats
            restored coordinate
        """
        if self.dim_z == 0:
            return 0, a, b
        elif self.dim_z == 1:
            return a, 0, b
        else:
            return a, b, 0

    def scatter(self, coord: np.ndarray, **kwargs):
        """
        Scatter plot for showing orbitals.

        :param coord: (num_orb, 3) float64 array
            Cartesian coordinates of orbitals
        :param kwargs: dictionary
            keyword arguments to be forwarded to axes.scatter()
        :return: None
        """
        coord = self.__proj_coord(coord)
        self.axes.scatter(coord[:, 0], coord[:, 1], **kwargs)

    def plot_arrow(self, coord_i: np.ndarray, coord_j: np.ndarray, **kwargs):
        """
        Plot an arrow between given points.

        :param coord_i: (3,) float64 array
            Cartesian coordinates of starting point
        :param coord_j: (3,) float64 array
            Cartesian coordinates of ending point
        :param kwargs: dictionary
            keyword arguments to be forwarded to axes.arrow()
        :return: None
        """
        coord_i = self.__proj_coord(coord_i)
        coord_j = self.__proj_coord(coord_j)
        diff = coord_j - coord_i
        self.axes.arrow(coord_i[0], coord_i[1], diff[0], diff[1], **kwargs)

    def add_line(self, coord_i: np.ndarray, coord_j: np.ndarray):
        """
        Add a line connecting give points.

        :param coord_i: (3,) float64 array
            Cartesian coordinates of starting point
        :param coord_j: (3,) float64 array
            Cartesian coordinates of ending point
        :return: None
        """
        coord_i = self.__proj_coord(coord_i)
        coord_j = self.__proj_coord(coord_j)
        self.hop_lc.append((coord_i, coord_j))

    def plot_line(self, **kwargs):
        """
        Plot lines in self.hop_lc.

        :param kwargs: dictionary
           keyword arguments to be forwarded to mc.LineCollection()
        :return: None
        """
        self.axes.add_collection(mc.LineCollection(self.hop_lc, **kwargs))

    def add_grid(self, a_min, a_max, b_min, b_max):
        """
        Add grid on the range of [a_min, a_max] * [b_min, b_max].

        :param a_min: integer
            lower bound of a
        :param a_max: integer
            upper bound of a
        :param b_min: integer
            lower bound of b
        :param b_max: integer
            upper bound of b
        :return: None
        """
        for ia in range(a_min, a_max + 1):
            x0 = self.__restore_coord(ia, b_min)
            x1 = self.__restore_coord(ia, b_max)
            x0 = self.__proj_coord(np.matmul(x0, self.lat_vec))
            x1 = self.__proj_coord(np.matmul(x1, self.lat_vec))
            self.cell_lc.append((x0, x1))
        for ib in range(b_min, b_max + 1):
            x0 = self.__restore_coord(a_min, ib)
            x1 = self.__restore_coord(a_max, ib)
            x0 = self.__proj_coord(np.matmul(x0, self.lat_vec))
            x1 = self.__proj_coord(np.matmul(x1, self.lat_vec))
            self.cell_lc.append((x0, x1))

    def plot_grid(self, **kwargs):
        """
        Plot grid in self.cell_lc.

        :param kwargs: dictionary
            keyword arguments to be forwarded to mc.LineCollection()
        :return:
        """
        self.axes.add_collection(mc.LineCollection(self.cell_lc, **kwargs))

    def plot_lat_vec(self, **kwargs):
        """
        Plot lattice vectors of (0, 0, 0)-th primitive cell as arrows.

        :param kwargs: dictionary
            keyword arguments to be forwarded to axes.arrow()
        :return: None
        """
        x0 = self.__restore_coord(0, 1)
        x1 = self.__restore_coord(1, 0)
        x0 = self.__proj_coord(np.matmul(x0, self.lat_vec))
        x1 = self.__proj_coord(np.matmul(x1, self.lat_vec))
        self.axes.arrow(0, 0, x0[0], x0[1], **kwargs)
        self.axes.arrow(0, 0, x1[0], x1[1], **kwargs)
