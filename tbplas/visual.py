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

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .builder import Sample


class Visualizer:
    """
    Class for visualizing data.

    Attributes
    ----------
    rank: integer
        rank of this process
    """
    def __init__(self, enable_mpi=False):
        """
        :param enable_mpi: boolean
            whether to enable mpi parallelism
        """
        if enable_mpi:
            from .parallel import MPIEnv
            self.rank = MPIEnv().rank
        else:
            self.rank = 0

    def __output(self, fig_name=None, fig_dpi=300):
        """
        Show or save the figure and close.

        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        if self.rank == 0:
            if fig_name is not None:
                plt.savefig(fig_name, dpi=fig_dpi)
            else:
                plt.show()
            plt.close()

    def __plot_xy(self, x: np.ndarray, y: np.ndarray, x_label=None,
                  y_label=None, fig_name=None, fig_dpi=300):
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
        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        if self.rank == 0:
            plt.plot(x, y)
            if x_label is not None:
                plt.xlabel(x_label)
            if y_label is not None:
                plt.ylabel(y_label)
            plt.tight_layout()
            self.__output(fig_name, fig_dpi)

    def plot_bands(self, k_len: np.ndarray, bands: np.ndarray,
                   k_idx: np.ndarray, k_label: List[str],
                   x_label: str = "k (1/nm)", y_label: str = "Energy (eV)",
                   fig_name=None, fig_dpi=300):
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
        :param y_label:
            label of y-axis
        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        if self.rank == 0:
            # Plot band structure
            num_bands = bands.shape[1]
            for i in range(num_bands):
                plt.plot(k_len, bands[:, i], color="r", linewidth=1.0)

            # Label highly-symmetric k-points
            for idx in k_idx:
                plt.axvline(k_len[idx], color='k', linewidth=1.0)

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
                 fig_name=None, fig_dpi=300):
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
        :param fig_name: string
            file name of figure to save
        :param fig_dpi: integer
            resolution of figure
        :return: None
        """
        self.__plot_xy(energies, dos, x_label, y_label, fig_name, fig_dpi)

    def plot_wf2(self, sample: Sample, wf2: np.ndarray, scatter=True,
                 site_size=5, num_grid=(200, 200), cmap="viridis",
                 with_colorbar=False, fig_name=None, fig_dpi=300):
        """
        Plot squared wave function in real space.

        :param sample: instance of 'Sample' class
            sample under study
        :param wf2: (num_orb_sample,) float64 array
            squared projection of wave function on all the sites
        :param scatter: boolean
            whether to plot the wave function as scatter
        :param site_size: float
            site size
        :param num_grid: (num_grid_x, num_grid_y)
            number of grid-points for interpolation along x and y directions
            when plotting the wave function
        :param cmap: string
            color map for plotting the wave function
        :param with_colorbar: boolean
            whether to add colorbar to figure
        :param fig_name: string
            image file name
        :param fig_dpi: float
            dpi of output figure
        :return: None
        """
        if self.rank == 0:
            # Get site locations
            sample.init_orb_pos()
            x = np.array(sample.orb_pos[:, 0])
            y = np.array(sample.orb_pos[:, 1])

            # Plot data
            fig, ax = plt.subplots()
            if scatter:
                img = ax.scatter(x, y, c=wf2, s=site_size, cmap=cmap)
            else:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                x_fi = np.linspace(x_min, x_max, num_grid[0])
                y_fi = np.linspace(y_min, y_max, num_grid[1])
                x_grid, y_grid = np.meshgrid(x_fi, y_fi)
                xy_fi = np.c_[x_grid.ravel(), y_grid.ravel()]
                z_fi = griddata((x, y), wf2, xy_fi,
                                method="cubic").reshape(num_grid)
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
