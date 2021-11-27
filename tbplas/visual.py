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

    def plot_band(self, k_len: np.ndarray, bands: np.ndarray,
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
        :param solver: instance of 'BaseSolver' and derive classes
            solver containing MPI environment
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

            # Show or save the figure
            if fig_name is not None:
                plt.savefig(fig_name, dpi=fig_dpi)
            else:
                plt.show()
            plt.close()

    def plot_wf2(self, sample: Sample, wf2, site_size=5, with_colorbar=False,
                 fig_name=None, fig_dpi=300):
        """
        Plot squared wave function in real space.

        :param sample: instance of 'Sample' class
            sample under study
        :param wf2: (n_indptr-1,) float64 array
            squared projection of wave function on all the sites
        :param site_size: float
            site size
        :param with_colorbar: boolean
            whether to add colorbar to figure
        :param fig_name: string
            image file name
        :param fig_dpi: float
            dpi of output figure
        :param solver: instance of 'BaseSolver' and derive classes
            solver containing MPI environment
        :return: None
        """
        if self.rank == 0:
            # Get site locations
            sample.init_orb_pos()
            x = np.array(sample.orb_pos[:, 0])
            y = np.array(sample.orb_pos[:, 1])

            # Get absolute square of wave function and sort
            z = wf2
            sorted_idx = z.argsort()
            x, y, z = x[sorted_idx], y[sorted_idx], z[sorted_idx]

            # make plot
            fig, ax = plt.subplots()
            sc = ax.scatter(x, y, c=z, s=site_size, edgecolor='')
            plt.axis('equal')
            plt.axis('off')
            if with_colorbar:
                plt.colorbar(sc)
            plt.tight_layout()
            plt.autoscale()

            # Show or save
            if fig_name is not None:
                plt.savefig(fig_name, dpi=fig_dpi)
            else:
                plt.show()
            plt.close()
