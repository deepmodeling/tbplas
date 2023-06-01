"""Utilities for visualizing results from exact diagonalization or TBPM."""

from typing import List, Tuple, Callable, Union, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from scipy.interpolate import griddata

from .builder import PrimitiveCell, Sample
from .parallel import MPIEnv


__all__ = ["Visualizer"]


class Visualizer(MPIEnv):
    """
    Class for quick data visualization.

    NOTES
    -----
    1. Order of arguments

    The arguments should follow:
        data (x, y, z, u, v, ...),
        figure settings (fig_name, fig_size, fig_dpi)
        model [optional],
        model settings [optional],
        axis settings (x_label, y_label, x_lim, y_lim)
        beautifier,
        arguments for low level matplotlib functions
    """
    def __init__(self, enable_mpi: bool = False,
                 echo_details: bool = False) -> None:
        """
        :param enable_mpi: whether to enable mpi parallelism
        :param echo_details: whether to output parallelization details
        """
        super().__init__(enable_mpi=enable_mpi, echo_details=echo_details)

    def _output(self, fig_name: str = None, fig_dpi: int = 300) -> None:
        """
        Save the figure or show it on the screen.

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
                fig_name: str = None,
                fig_size: Tuple[float, float] = None,
                fig_dpi: int = 300,
                x_label: str = None,
                y_label: str = None,
                x_lim: Tuple[float, float] = None,
                y_lim: Tuple[float, float] = None,
                beautifier: Callable = None,
                color: str = "r",
                linewidth: float = 1.2,
                **kwargs) -> None:
        """
        Plot y as function of x.

        :param x: (num_data,) float64 array
            data x for plot
        :param y: (num_data,) float64 array
            data y for plot
        :param fig_name: file name of figure to save
        :param fig_size: width and height of the figure
        :param fig_dpi: resolution of figure
        :param x_label: label for x-axis
        :param y_label: label for y-axis
        :param x_lim: (x_min, x_max) range of x
        :param y_lim: (y_min, y_max) range of y
        :param beautifier: function for improving the plot
        :param color: line color
        :param linewidth: line width
        :param kwargs: parameters for plt.plot()
        :return: None
        """
        if self.is_master:
            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
            if x_label is not None:
                ax.set_xlabel(x_label)
            if y_label is not None:
                ax.set_ylabel(y_label)
            if x_lim is not None:
                ax.set_xlim(x_lim)
            if y_lim is not None:
                ax.set_ylim(y_lim)
            if beautifier is not None:
                beautifier()
            fig.tight_layout()
            self._output(fig_name, fig_dpi)

    def plot_bands(self, k_len: np.ndarray,
                   bands: np.ndarray,
                   k_idx: np.ndarray,
                   k_label: List[str],
                   fig_name: str = None,
                   fig_size: Tuple[float, float] = None,
                   fig_dpi: int = 300,
                   x_label: str = "k (1/nm)",
                   y_label: str = "Energy (eV)",
                   beautifier: Callable = None,
                   color: str = "r",
                   linewidth: float = 1.2,
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
        :param fig_name: file name of figure to save
        :param fig_size: width and height of the figure
        :param fig_dpi: resolution of figure
        :param x_label: label of x-axis
        :param y_label: label of y-axis
        :param beautifier: function for improving the plot
        :param color: line color
        :param linewidth: line width
        :param kwargs: parameters for plt.plot()
        :return: None
        """
        if self.is_master:
            fig, ax = plt.subplots(figsize=fig_size)

            # Plot band structure
            num_bands = bands.shape[1]
            for i in range(num_bands):
                ax.plot(k_len, bands[:, i], color=color, linewidth=linewidth,
                        **kwargs)

            # Label highly-symmetric k-points
            for idx in k_idx:
                ax.axvline(k_len[idx], color="k",
                           linewidth=plt.rcParams['axes.linewidth'])

            # Adjustment
            ax.set_xlim((0, np.amax(k_len)))
            ax.set_xticks(k_len[k_idx])
            ax.set_xticklabels(k_label)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if beautifier is not None:
                beautifier()
            fig.tight_layout()

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
        self.plot_xy(energies, dos, x_label=x_label, y_label=y_label, **kwargs)

    def plot_phases(self, kb_array: np.ndarray,
                    phases: np.ndarray,
                    fig_name: str = None,
                    fig_size: Tuple[float, float] = None,
                    fig_dpi: int = 300,
                    scatter: bool = True,
                    polar: bool = False,
                    x_label: str = "$k_b (G_b)$",
                    y_label: str = r"$\theta$",
                    beautifier: Callable = None,
                    color: str = "r",
                    linewidth: float = 1.2,
                    **kwargs) -> None:
        """
        :param kb_array: (num_kb,) float64 array
            FRACTIONAL coordinates of the loop along b-axis
        :param phases: (num_kpt, num_occ) float64 array
            phases of WF centers
        :param fig_name: file name of figure
        :param fig_size: width and height of the figure
        :param fig_dpi: dpi of output figure
        :param scatter: whether to do scatter plot instead of line plot
        :param polar: whether to plot in polar coordinate system
        :param x_label: label for x-axis
        :param y_label: label for y-axis
        :param beautifier: function for improving the plot
        :param color: line color
        :param linewidth: line width
        :param kwargs: parameters for plt.plot() and plt.scatter()
        :return: None
        """
        if self.is_master:
            if polar:
                fig, ax = plt.subplots(figsize=fig_size,
                                       subplot_kw={'projection': 'polar'})
            else:
                fig, ax = plt.subplots(figsize=fig_size)

            # Plot phases
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

            # Adjustment
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if beautifier is not None:
                beautifier()
            fig.tight_layout()

            # Output figure
            self._output(fig_name, fig_dpi)

    def _plot_model(self, ax: plt.axes,
                    model: Union[PrimitiveCell, Sample],
                    model_style: Dict[str, Any] = None) -> None:
        """
        Plot model onto axes.

        :param ax: axes on which the model will be plotted
        :param model: model to plot
        :param model_style: style of the model
        :return: None
        """
        if self.is_master:
            if isinstance(model, PrimitiveCell):
                orb_pos = model.orb_pos_nm
                hop_i = model.hop_ind[:, 3]
                dr = model.dr_nm
            else:
                model.init_orb_pos()
                model.init_hop()
                orb_pos = model.orb_pos
                hop_i, dr = model.hop_i, model.dr
            hop_lc = []
            for i_h in range(hop_i.shape[0]):
                pos_i = orb_pos[hop_i.item(i_h)]
                pos_j = pos_i + dr[i_h]
                hop_lc.append((pos_i[:2], pos_j[:2]))
            if model_style is None:
                model_style = {"color": "gray", "alpha": 0.5}
            ax.add_collection(mc.LineCollection(hop_lc, **model_style))

    def plot_scalar(self, x: np.ndarray,
                    y: np.ndarray,
                    z: np.ndarray,
                    fig_name: str = None,
                    fig_size: Tuple[float, float] = None,
                    fig_dpi: int = 300,
                    model: Union[PrimitiveCell, Sample] = None,
                    model_style: Dict[str, Any] = None,
                    scatter: bool = False,
                    site_size: Union[int, np.ndarray] = 5,
                    site_color: Union[str, List[str]] = "cmap",
                    num_grid: Tuple[int, int] = (200, 200),
                    cmap: str = "viridis",
                    with_colorbar: bool = False,
                    beautifier: Callable = None,
                    **kwargs) -> None:
        """
        Plot 2d scalar field z = f(x, y).

        :param x: (num_data,) float64 array
            x-component of Cartesian coordinates of data points
        :param y: (num_data,) float64 array
            y-component of Cartesian coordinates of data points
        :param z: (num_data,) float64 array
            z-value of data points
        :param fig_name: image file name
        :param fig_size: width and height of the figure
        :param fig_dpi: dpi of output figure
        :param model: model with which the data will be plotted
        :param model_style: style of the model
        :param scatter: whether to plot the wave function as scatter
        :param site_size: size of the sites for scatter plot
        :param site_color: color of the sites, use colormap from z-value if set
            to camp, otherwise monochromatic
        :param num_grid: (nx, ny) number of grid-points for interpolation along
            x and y directions when plotting the wave function
        :param cmap: color map for plotting the wave function
        :param with_colorbar: whether to add colorbar to figure
        :param beautifier: function for improving the plot
        :param kwargs: parameters for plt.scatter() and plt.imshow()
        :return: None
        """
        if self.is_master:
            fig, ax = plt.subplots(figsize=fig_size)
            if model is not None:
                self._plot_model(ax, model, model_style)

            # Plot data
            if scatter:
                if site_color == "cmap":
                    img = ax.scatter(x, y, c=z, s=site_size, cmap=cmap,
                                     **kwargs)
                else:
                    img = ax.scatter(x, y, c=site_color, s=site_size,
                                     **kwargs)
            else:
                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)
                x_fi = np.linspace(x_min, x_max, num_grid[0])
                y_fi = np.linspace(y_min, y_max, num_grid[1])
                x_grid, y_grid = np.meshgrid(x_fi, y_fi)
                xy_fi = np.c_[x_grid.ravel(), y_grid.ravel()]
                z_fi = griddata((x, y), z, xy_fi, method="cubic")
                z_fi = z_fi.reshape(num_grid)
                extent = (x_min, x_max, y_min, y_max)
                img = ax.imshow(z_fi, cmap=cmap, interpolation="none",
                                origin="lower", extent=extent, **kwargs)

            # Adjustment
            if with_colorbar:
                fig.colorbar(img)
            ax.set_aspect('equal')
            for key in ("top", "bottom", "left", "right"):
                ax.spines[key].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if beautifier is not None:
                beautifier()
            fig.tight_layout()
            ax.autoscale()

            # Output figure
            self._output(fig_name, fig_dpi)

    def plot_vector(self, x: np.ndarray,
                    y: np.ndarray,
                    u: np.ndarray,
                    v: np.ndarray,
                    fig_name: str = None,
                    fig_size: Tuple[float, float] = None,
                    fig_dpi: int = 300,
                    model: Union[PrimitiveCell, Sample] = None,
                    model_style: Dict[str, Any] = None,
                    arrow_color: Union[str, List[str]] = "cmap",
                    cmap: str = "viridis",
                    with_colorbar: bool = False,
                    beautifier: Callable = None,
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
        :param fig_name: image file name
        :param fig_size: width and height of the figure
        :param fig_dpi: dpi of output figure
        :param model: model with which the data will be plotted
        :param model_style: style of the model
        :param arrow_color: color for the arrows
        :param cmap: color map for plotting the arrows
        :param with_colorbar: whether to add colorbar to figure
        :param beautifier: function for improving the plot
        :param kwargs: keyword arguments for quiver function
        :return: None
        """
        if self.is_master:
            fig, ax = plt.subplots(figsize=fig_size)
            if model is not None:
                self._plot_model(ax, model, model_style)

            # Plot data
            if arrow_color == "cmap":
                c = np.sqrt(np.abs(u)**2 + np.abs(v)**2)
                img = ax.quiver(x, y, u, v, c, cmap=cmap, **kwargs)
            else:
                img = ax.quiver(x, y, u, v, color=arrow_color, **kwargs)

            # Adjustment
            if with_colorbar:
                fig.colorbar(img)
            ax.set_aspect('equal')
            for key in ("top", "bottom", "left", "right"):
                ax.spines[key].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if beautifier is not None:
                beautifier()
            fig.tight_layout()
            ax.autoscale()

            # Output figure
            self._output(fig_name, fig_dpi)

    def plot_wfc(self, model: Union[PrimitiveCell, Sample],
                 wfc: np.ndarray,
                 with_model: bool = True,
                 **kwargs) -> None:
        """
        Plot wave function in real space.

        :param model: model to which the wave function belongs
        :param with_model: whether to plot model along with the wave function
        :param wfc: (num_orb,) float64 array
            projection of wave function on all the sites
        :param kwargs: arguments for 'plot_scalar'
        :return: None
        """
        # Get site locations
        if isinstance(model, PrimitiveCell):
            orb_pos = model.orb_pos_nm
        else:
            model.init_orb_pos()
            orb_pos = model.orb_pos
        x = np.array(orb_pos[:, 0])
        y = np.array(orb_pos[:, 1])

        # Plot wfc
        if with_model:
            self.plot_scalar(x, y, wfc, model=model, **kwargs)
        else:
            self.plot_scalar(x, y, wfc, model=None, **kwargs)
