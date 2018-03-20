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

# plotting
import matplotlib.pyplot as plt

def plot_wf(wf, sample, filename, site_size=5, fig_dpi=300, colorbar=False):
    """Plot wavefunction
    
    Parameters
    ----------
    wf : list of complex floats
        wavefunction
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
    x = np.array(sample.site_x)
    y = np.array(sample.site_y)
    
    # get absolute square of wave function and sort
    z = np.power(np.array(np.abs(wf)), 2)
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