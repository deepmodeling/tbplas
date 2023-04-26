"""Utilities for visualizing results from exact diagonalization or TBPM."""

from typing import List, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .builder import Sample
from .parallel import MPIEnv


__all__ = ["Visualizer"]


class Visualizer(MPIEnv):
    """Class for data visualization."""
    def __init__(self, enable_mpi: bool = False,
                 echo_details: bool = False) -> None:
        """
        :param enable_mpi: whether to enable mpi parallelism
        :param echo_details: whether to output parallelization details
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=echo_details)

    def _output(self, fig_name: str = None, fig_dpi: int = 300) -> None:
        """
        Show or save the figure and close.

        :param fig_name: file name of figure to save
        :param fig_dpi: resolution of figure
        :return: None
        """
        if self.is_master:
            if fig_name is not None:
                plt.savefig(fig_name, dpi=fig_dpi)
            else:
                plt.show()
            plt.close()

    def plot_xy(self, x: np.ndarray,
                y: np.ndarray,
                x_label: str = None,
                y_label: str = None,
                x_lim: Tuple[float, float] = None,
                y_lim: Tuple[float, float] = None,
                color: str = "r",
                linewidth: float = 1.2,
                beautifier: Callable = None,
                fig_name: str = None,
                fig_dpi: int = 300,
                **kwargs) -> None:
        """
        Plot y as function of x.

        :param x: (num_data,) float64 array
            data x for plot
        :param y: (num_data,) float64 array
            data y for plot
        :param x_label: label for x-axis
        :param y_label: label for y-axis
        :param x_lim: (x_min, x_max) range of x
        :param y_lim: (y_min, y_max) range of y
        :param color: line color
        :param linewidth: line width
        :param beautifier: function for improving the plot
        :param fig_name: file name of figure to save
        :param fig_dpi: resolution of figure
        :param kwargs: parameters for plt.plot()
        :return: None
        """
        if self.is_master:
            plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)
            if x_label is not None:
                plt.xlabel(x_label)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_lim is not None:
                plt.xlim(x_lim)
            if y_lim is not None:
                plt.ylim(y_lim)
            if beautifier is not None:
                beautifier()
            plt.tight_layout()
            self._output(fig_name, fig_dpi)

    def plot_bands(self, k_len: np.ndarray,
                   bands: np.ndarray,
                   k_idx: np.ndarray,
                   k_label: List[str],
                   x_label: str = "k (1/nm)",
                   y_label: str = "Energy (eV)",
                   color: str = "r",
                   linewidth: float = 1.2,
                   beautifier: Callable = None,
                   fig_name: str = None,
                   fig_dpi: int = 300,
                   **kwargs) -> None:
        """
        Plot band structure.

        :param k_len: (num_kpt,) float64 array
            distance of k-path in reciprocal space
        :param bands: (num_kpt, num_band) float64 array
            energies corresponding to k_len
        :param k_idx: (num_hsk,) int32 array
            indices of highly-symmetric k-points in k_len
        :param k_label: (num_hsk,) labels of highly-symmetric k-points
        :param x_label: label of x-axis
        :param y_label: label of y-axis
        :param color: line color
        :param linewidth: line width
        :param beautifier: function for improving the plot
        :param fig_name: file name of figure to save
        :param fig_dpi: resolution of figure
        :param kwargs: parameters for plt.plot()
        :return: None
        """
        if self.is_master:
            # Plot band structure
            num_bands = bands.shape[1]
            for i in range(num_bands):
                plt.plot(k_len, bands[:, i], color=color, linewidth=linewidth,
                         **kwargs)

            # Label highly-symmetric k-points
            for idx in k_idx:
                plt.axvline(k_len[idx], color="k",
                            linewidth=plt.rcParams['axes.linewidth'])

            # Adjustment
            plt.xlim((0, np.amax(k_len)))
            plt.xticks(k_len[k_idx], k_label)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            if beautifier is not None:
                beautifier()
            plt.tight_layout()

            # Output figure
            self._output(fig_name, fig_dpi)

    def plot_dos(self, energies: np.ndarray,
                 dos: np.ndarray,
                 x_label: str = "Energy (eV)",
                 y_label: str = "DOS (1/eV)",
                 **kwargs) -> None:
        """
        Plot density of states.

        :param energies: (num_eng,) float64 array
            energy grids
        :param dos: (num_eng,) float64 array
            density of states
        :param x_label: label for x-axis
        :param y_label: label for y-axis
        :param kwargs: arguments for 'plot_xy'
        :return: None
        """
        self.plot_xy(energies, dos, x_label, y_label, **kwargs)

    def plot_phases(self, kb_array: np.ndarray,
                    phases: np.ndarray,
                    scatter: bool = True,
                    polar: bool = False,
                    x_label: str = "$k_b (G_b)$",
                    y_label: str = r"$\theta$",
                    color: str = "r",
                    linewidth: float = 1.2,
                    beautifier: Callable = None,
                    fig_name: str = None,
                    fig_dpi: int = 300,
                    **kwargs) -> None:
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
        :param beautifier: function for improving the plot
        :param fig_name: file name of figure
        :param fig_dpi: dpi of output figure
        :param kwargs: parameters for plt.plot() and plt.scatter()
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
                    ax.scatter(x, y, s=1, color=color, linewidth=linewidth,
                               **kwargs)
                else:
                    ax.plot(x, y, c=color, **kwargs)

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            if beautifier is not None:
                beautifier()
            plt.tight_layout()
            self._output(fig_name, fig_dpi)

    def plot_scalar(self, x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray,
                    scatter: bool = False,
                    site_size: int = 5,
                    num_grid: Tuple[int, int] = (200, 200),
                    cmap: str = "viridis",
                    with_colorbar: bool = False,
                    beautifier: Callable = None,
                    fig_name: str = None,
                    fig_dpi: int = 300,
                    **kwargs) -> None:
        """
        Plot 2d scalar field z = f(x, y).

        :param x: (num_data,) float64 array
            x-component of Cartesian coordinates of data points
        :param y: (num_data,) float64 array
            y-component of Cartesian coordinates of data points
        :param z: (num_data,) float64 array
            z-value of data points
        :param scatter: whether to plot the wave function as scatter
        :param site_size: site size for scatter plot
        :param num_grid: (nx, ny) number of grid-points for interpolation along
            x and y directions when plotting the wave function
        :param cmap: color map for plotting the wave function
        :param with_colorbar: whether to add colorbar to figure
        :param beautifier: function for improving the plot
        :param fig_name: image file name
        :param fig_dpi: dpi of output figure
        :param kwargs: parameters for plt.scatter() and plt.imshow()
        :return: None
        """
        if self.is_master:
            # Plot data
            fig, ax = plt.subplots()
            if scatter:
                img = ax.scatter(x, y, c=z, s=site_size, cmap=cmap, **kwargs)
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
                                extent=(x_min, x_max, y_min, y_max), **kwargs)

            if with_colorbar:
                plt.colorbar(img)
            plt.axis('equal')
            plt.axis('off')
            if beautifier is not None:
                beautifier()
            plt.tight_layout()
            plt.autoscale()

            # Output figure
            self._output(fig_name, fig_dpi)

    def plot_vector(self, x: np.ndarray,
                    y: np.ndarray,
                    u: np.ndarray,
                    v: np.ndarray,
                    beautifier: Callable = None,
                    fig_name: str = None,
                    fig_dpi: int = 300,
                    **kwargs) -> None:
        """
        Plot 2d vector field [u(x, y), v(x, y)].

        :param x: (num_data,) float64 array
            x-component of Cartesian coordinates of data points
        :param y: (num_data,) float64 array
            y-component of Cartesian coordinates of data points
        :param u: (num_data,) float64 array
            x component of arrow directions
        :param v: (num_data,) float64 array
            y component of arrow directions
        :param beautifier: function for improving the plot
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
            if beautifier is not None:
                beautifier()
            plt.tight_layout()
            plt.autoscale()
            self._output(fig_name, fig_dpi)

    def plot_wfc(self, sample: Sample, wfc: np.ndarray, **kwargs) -> None:
        """
        Plot wave function in real space.

        :param sample: sample under study
        :param wfc: (num_orb_sample,) float64 array
            projection of wave function on all the sites
        :param kwargs: arguments for 'plot_scalar'
        :return: None
        """
        # Get site locations
        sample.init_orb_pos()
        x = np.array(sample.orb_pos[:, 0])
        y = np.array(sample.orb_pos[:, 1])

        # Plot wfc
        self.plot_scalar(x, y, wfc, **kwargs)
