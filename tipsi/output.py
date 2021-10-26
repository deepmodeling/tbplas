"""output.py contains tools for data output.

Functions
----------
    plot_wf
        Plot wavefunction.
"""

################
# dependencies
################

# numerics & math
import numpy as np
import matplotlib.pyplot as plt


def plot_wf(wfsq, sample, filename, site_size=5, fig_dpi=300, colorbar=False):
    """Plot wavefunction

    Parameters
    ----------
    wfsq : list of positive real numbers
        wave function squared
    sample : Sample object
        geometric information of the sample
    filename : string
        image file name
    site_size : float
        site size; default 5
    fig_dpi : float
        dpi of output figure; default 300
    colorbar : bool
        add colorbar to figure; default False
    """

    # get site locations
    sample.init_orb_pos()
    x = np.array(sample.orb_pos[:, 0])
    y = np.array(sample.orb_pos[:, 1])

    # get absolute square of wave function and sort
    z = wfsq
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # make plot
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=site_size, edgecolor='')
    plt.axis('equal')
    plt.axis('off')
    if colorbar:
        plt.colorbar(sc)
    plt.draw()
    plt.savefig(filename, dpi=fig_dpi)
    plt.close()
