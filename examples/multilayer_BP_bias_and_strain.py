import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..")
import tipsi
from tipsi.materials import black_phosphorus

def bp_lat_hop(n_layers = 1, bias = 0., strain = 0.):
    """Lattice-HopDict pair for multilayer black phosphorus with bias and strain.
    
    Parameters
    ----------
    n_layers : integer
        number of layers, optional (default 1)
    bias : float
        bias (in units eV/nm), optional (default 0.)
    strain : float
        strain (in percentage), optional (default 0.)
        
    Returns
    ----------
    lat : tipsi.Lattice object
        Black phosphorus lattice.
    hops : tipsi.HopDict object
        Black phosphorus hopping dictionary.
    """
    
    # create lattice, hop_dict and pbc_wrap
    lat = black_phosphorus.lattice()
    hops = black_phosphorus.hop_dict()
    
    # add strain to hop_dict and lattice
    strain_tensor = np.diag([1.0 - 0.002 * strain, \
                             1.0 + 0.01 * strain, \
                             1.0 - 0.002 * strain])
    beta = 4.5
    lat, hops = tipsi.uniform_strain(lat, hops, strain_tensor, beta)
    
    # extend unit cell and add bias
    lat, hops = tipsi.extend_unit_cell(lat, hops, 2, n_layers)
    for i in range(4 * n_layers):
        z = lat.site_pos((0, 0, 0), i)[2]
        onsite_pot = z * bias
        hops.set_element((0, 0, 0), (i, i), onsite_pot)
    
    # remove redundant z-direction hoppings
    hops.remove_z_hoppings()
    
    # done
    return lat, hops

def bp_sample(W, H, n_layers, lat, hops, nr_processes = 1, pbc = 'b'):
    """Create a sample of multilayer rectangular black phosphorus.
    
    Parameters
    ----------
    W : integer
        width of the sample, in unit cells
    H : integer
        height of the sample, in unit cells
    n_layers : integer
        number of layers, optional (default 1)
    lat : tipsi.Lattice object
        Black phosphorus lattice.
    hops : tipsi.HopDict object
        Black phosphorus hopping dictionary.
    nr_processes : integer
        number of processes for sample building, optional (default 1)
    pbc : string
        'b' gives periodic boundary conditions in both directions
        'a' gives an armchair ribbon
        'z' gives a zigzag ribbon
        'n' (or anything else) gives no periodic boundary conditions
        
    Returns
    ----------
    sample : tipsi.Sample object
        Black phosphorus sample.
    """
    
    # create SiteSet object
    site_set = tipsi.SiteSet()
    for z in range(n_layers):
        for x in range(W):
            for y in range(H):
                for i in range(4 * n_layers):
                    site_set.add_site((x, y, z), i)
    
    # remove bearded edge for zigzag ribbons
    if pbc == 'z':
        for z in range(n_layers):
            for x in range(W):
                for i in range(n_layers):
                    site_set.delete_site((x, 0, z), i * 4)
                for i in range(n_layers):
                    site_set.delete_site((x, H - 1, z), i * 4 + 3)
        
    # define pbc
    if pbc == 'b':
        def pbc_func(unit_cell_coords, orbital):
            x, y, z = unit_cell_coords
            return (x % W, y % H, z), orbital
    elif pbc == 'a':
        def pbc_func(unit_cell_coords, orbital):
            x, y, z = unit_cell_coords
            return (x, y % H, z), orbital
    elif pbc == 'z':
        def pbc_func(unit_cell_coords, orbital):
            x, y, z = unit_cell_coords
            return (x % W, y, z), orbital
    else:
        def pbc_func(unit_cell_coords, orbital):
            return unit_cell_coords, orbital
    
    # make sample
    sample = tipsi.Sample(lat, site_set, pbc_func, nr_processes)

    # apply hop_dict
    sample.add_hop_dict(hops)
    
    # rescale Hamiltonian
    sample.rescale_H(10.)
    
    return sample

def main():
    
    # parameters
    W = 10
    H = 10
    n_layers = 2
    bias = 0
    strain = -5
    nr_processes = 8
    k_res = 50
    pbc = 'b'
    
    output_bands_img = "bp_bands_size_"+str(W)+"x"+str(H)+"_layers_"+str(n_layers)+"_bias_"+str(bias)+"_strain_"+str(strain)+".png"
    output_ac_dat = "bp_ac_size_"+str(W)+"x"+str(H)+"_layers_"+str(n_layers)+"_bias_"+str(bias)+"_strain_"+str(strain)+".dat"
    output_ac_img= "bp_ac_size_"+str(W)+"x"+str(H)+"_layers_"+str(n_layers)+"_bias_"+str(bias)+"_strain_"+str(strain)+".png"
    
    # get Lattice, HopDict pair
    lat, hops = bp_lat_hop(n_layers, bias, strain)
    
    
    
    # BAND STRUCTURE
    
    # define symmetry points
    G = np.array([0. ,0. ,0. ])
    X = lat.reciprocal_latt()[1] / 2
    Y = lat.reciprocal_latt()[0] / 2
    S = X + Y
    kpoints = [G, Y, S, X, G]
    ticktitles = ["G","Y","S","X","G"]
    kpoints, kvals, ticks = tipsi.interpolate_k_points(kpoints, k_res)
    
    # get band structure
    bands = tipsi.band_structure(hops, lat, kpoints)
    mu = (bands[0,3] + bands[0,4]) / 2.
    bands = bands - mu

    # plot
    for i in range(len(bands[0,:])):
        plt.plot(kvals, bands[:,i], color='k')
    for tick in ticks:
        plt.axvline(tick, color='k', linewidth=0.5)
    plt.xticks(ticks, ticktitles)
    plt.xlim((0., np.amax(kvals)))
    plt.xlabel("k (1/nm)")
    plt.ylabel("E (eV)")
    plt.savefig(output_bands_img)
    plt.close()
    
    # TBPM
    
    # sample
    sample = bp_sample(W, H, n_layers, lat, hops, nr_processes, pbc)
    #sample.plot()
    
    # config object
    config = tipsi.Config(sample)
    config.generic['mu'] = mu
    config.save(directory = 'sim_data', \
        prefix = config.output['timestamp'])

    # get AC conductivity
    corr_AC = tipsi.corr_AC(sample, config)
    omegas_AC, AC = tipsi.analyze_corr_AC(config, corr_AC)
    AC_real_xx = AC[0]
    AC_imag_xx = tipsi.AC_imag(AC_real_xx)
    AC_real_yy = AC[3]
    AC_imag_yy = tipsi.AC_imag(AC_real_yy)
    
    # write
    data = np.column_stack((omegas_AC, AC_real_xx, AC_imag_xx, AC_real_yy, AC_imag_yy))
    header = "hbar * omega, AC_real_xx, AC_imag_xx, AC_real_yy, AC_imag_yy"
    np.savetxt(output_ac_dat, data, header = header)
    
    # plot
    plt.plot(omegas_AC, AC_real_xx, label = "Re(sigma_xx)")
    plt.plot(omegas_AC, AC_imag_xx, label = "Im(sigma_xx)")
    plt.plot(omegas_AC, AC_real_yy, label = "Re(sigma_yy)")
    plt.plot(omegas_AC, AC_imag_yy, label = "Im(sigma_yy)")
    plt.plot(omegas_AC, AC_imag_xx * AC_imag_yy, label = "Im(sigma_xx) * Im(sigma_yy)")
    plt.xlabel("hbar * omega (eV)")
    plt.ylabel("sigma (sigma_0)")
    plt.xlim((0., 4.))
    plt.ylim((-3., 3.))
    plt.legend()
    plt.savefig(output_ac_img)
    plt.close()
    
if __name__ == '__main__':
    main()
