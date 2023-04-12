"""
Utilities for visualizing results from exact diagonalizing or TBPM.

Functions
---------
    None

Classes
-------
    Visualizer: user class
        class for visualizing data
"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .builder import Sample
from .parallel import MPIEnv


class Visualizer(MPIEnv):
    """Class for visualizing data."""
    def __init__(self, enable_mpi=False):
        """
        :param enable_mpi: boolean
            whether to enable mpi parallelism
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=False)

    def __output(self, fig_name=None, fig_dpi=300):
        """
        Show or save the figure and close.

        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        if self.is_master:
            if fig_name is not None:
                plt.savefig(fig_name, dpi=fig_dpi)
            else:
                plt.show()
            plt.close()

    def plot_xy(self, x: np.ndarray, y: np.ndarray,
                x_label=None, y_label=None, x_lim=None, y_lim=None,
                color="r", linewidth=1.2, fig_name=None, fig_dpi=300):
        """
        Plot y as function of x.

        :param x: (num_data,) float64 array
            data x for plot
        :param y: (num_data,) float64 array
            data y for plot
        :param x_label: string
            label for x-axis
        :param y_label: string
            label for y-axis
        :param x_lim: (x_min, x_max)
            range of x
        :param y_lim: (y_min, y_max)
            range of y
        :param color: string
            line color
        :param linewidth: float
            line width
        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        if self.is_master:
            plt.plot(x, y, color=color, linewidth=linewidth)
            if x_label is not None:
                plt.xlabel(x_label)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)
            plt.tight_layout()
            self.__output(fig_name, fig_dpi)

    def plot_bands(self, k_len: np.ndarray, bands: np.ndarray,
                   k_idx: np.ndarray, k_label: List[str],
                   x_label: str = "k (1/nm)", y_label: str = "Energy (eV)",
                   color="r", linewidth=1.2,  fig_name=None, fig_dpi=300):
        """
        Plot band structure.

        :param k_len: (num_kpt,) float64 array
            distance of k-path in reciprocal space
        :param bands: (num_kpt, num_band) float64 array
            energies corresponding to k_len
        :param k_idx: (num_hsk,) int32 array
            indices of highly-symmetric k-points in k_len
        :param k_label: (num_hsk,) string
            labels of highly-symmetric k-points
        :param x_label: string
            label of x-axis
        :param y_label: string
            label of y-axis
        :param color: string
            line color
        :param linewidth: float
            line width
        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        if self.is_master:
            # Plot band structure
            num_bands = bands.shape[1]
            for i in range(num_bands):
                plt.plot(k_len, bands[:, i], color=color, linewidth=linewidth)

            # Label highly-symmetric k-points
            for idx in k_idx:
                plt.axvline(k_len[idx], color="k",
                            linewidth=plt.rcParams['axes.linewidth'])

            # Adjustment
            plt.xlim((0, np.amax(k_len)))
            plt.xticks(k_len[k_idx], k_label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()

            # Output figure
            self.__output(fig_name, fig_dpi)

    def plot_dos(self, energies: np.ndarray, dos: np.ndarray,
                 x_label="Energy (eV)", y_label="DOS (1/eV)",
                 **kwargs):
        """
        Plot density of states.

        :param energies: (num_eng,) float64 array
            energy grids
        :param dos: (num_eng,) float64 array
            density of states
        :param x_label: string
            label for x-axis
        :param y_label: string
            label for y-axis
        :param kwargs: dict
            keyword arguments for plot_xy
        :return: None
        """
        self.plot_xy(energies, dos, x_label, y_label, **kwargs)

    def plot_phases(self, kb_array: np.ndarray, phases: np.ndarray,
                    scatter: bool = True, polar: bool = False,
                    x_label: str = "$k_b (G_b)$", y_label: str = r"$\theta$",
                    color: str = "r", linewidth: float = 1.2,
                    fig_name: str = None, fig_dpi: int = 300):
        """
        :param kb_array: (num_kb,) float64 array
            FRACTIONAL coordinates of the loop along b-axis
        :param phases: (num_kpt, num_occ) float64 array
            phases of WF centers
        :param scatter: whether to do scatter plot instead of line plot
        :param polar: whether to plot in polar coordinate system
        :param x_label: label for x-axis
        :param y_label: label for y-axis
        :param color: line color
        :param linewidth: line width
        :param fig_name: file name of figure
        :param fig_dpi: dpi of output figure
        :return: None
        """
        if self.is_master:
            if polar:
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            else:
                fig, ax = plt.subplots()

            for ib in range(phases.shape[1]):
                if polar:
                    x, y = phases[:, ib], kb_array
                else:
                    x, y = kb_array, phases[:, ib]
                if scatter:
                    ax.scatter(x, y, s=1, color=color, linewidth=linewidth)
                else:
                    ax.plot(x, y, c=color)

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            self.__output(fig_name, fig_dpi)

    def plot_scalar(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    scatter: bool = False, site_size: int = 5,
                    num_grid: Tuple[int, int] = (200, 200),
                    cmap: str = "viridis", with_colorbar: bool = False,
                    fig_name: str = None, fig_dpi: int = 300):
        """
        Plot 2d scalar field z = f(x, y).

        :param x: x-component of Cartesian coordinates of data points
        :param y: y-component of Cartesian coordinates of data points
        :param z: z-value of data points
        :param scatter:  whether to plot the wave function as scatter
        :param site_size: site size for scatter plot
        :param num_grid: number of grid-points for interpolation along x and y
            directions when plotting the wave function
        :param cmap: color map for plotting the wave function
        :param with_colorbar: whether to add colorbar to figure
        :param fig_name: image file name
        :param fig_dpi: dpi of output figure
        :return: None
        """
        if self.is_master:
            # Plot data
            fig, ax = plt.subplots()
            if scatter:
                img = ax.scatter(x, y, c=z, s=site_size, cmap=cmap)
            else:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                x_fi = np.linspace(x_min, x_max, num_grid[0])
                y_fi = np.linspace(y_min, y_max, num_grid[1])
                x_grid, y_grid = np.meshgrid(x_fi, y_fi)
                xy_fi = np.c_[x_grid.ravel(), y_grid.ravel()]
                z_fi = griddata((x, y), z, xy_fi, method="cubic")
                z_fi = z_fi.reshape(num_grid)
                img = ax.imshow(z_fi, cmap=cmap, interpolation="none",
                                origin="lower",
                                extent=(x_min, x_max, y_min, y_max))

            if with_colorbar:
                plt.colorbar(img)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.autoscale()

            # Output figure
            self.__output(fig_name, fig_dpi)

    def plot_vector(self, x: np.ndarray, y: np.ndarray,
                    u: np.ndarray, v: np.ndarray,
                    fig_name: str = None, fig_dpi: int = 300, **kwargs):
        """
        Plot 2d vector field [u(x, y), v(x, y)].

        :param x: x coordinates of arrow locations
        :param y: y coordinates of arrow locations
        :param u: x component of arrow directions
        :param v: y component of arrow directions
        :param fig_name: image file name
        :param fig_dpi: dpi of output figure
        :param kwargs: keyword arguments for quiver function
        :return: None
        """
        if self.is_master:
            fig, ax = plt.subplots()
            c = np.linalg.norm(u.real + 1j * v.real)
            ax.quiver(x, y, u, v, c, **kwargs)
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.autoscale()
            self.__output(fig_name, fig_dpi)

    def plot_wfc(self, sample: Sample, wfc: np.ndarray, **kwargs):
        """
        Plot wave function in real space.

        :param sample: sample under study
        :param wfc: (num_orb_sample,) float64 array, projection of
            wave function on all the sites
        :param kwargs: keyword arguments for plot_z
        :return: None
        """
        # Get site locations
        sample.init_orb_pos()
        x = np.array(sample.orb_pos[:, 0])
        y = np.array(sample.orb_pos[:, 1])

        # Plot wfc
        self.plot_scalar(x, y, wfc, **kwargs)
