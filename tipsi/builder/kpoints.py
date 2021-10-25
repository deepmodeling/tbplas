"""Functions for k-point operations."""

import numpy as np
from .lattice import gen_reciprocal_vectors, frac2cart


def gen_kpath(hs_kpoints, num_kpoints):
    """
    Generate path in the reciprocal space connecting highly symmetric k-points.

    :param hs_kpoints: 3*N array, list of highly symmetric k-points
    :param num_kpoints: N-1 list of numbers of intermediate k-points
            between two highly symmetric k-points
    :return: k_path: array with 3 columns
    """
    assert hs_kpoints.shape[0] == len(num_kpoints) + 1
    kpath = []
    for i in range(len(num_kpoints)):
        k0 = hs_kpoints[i]
        k1 = hs_kpoints[i+1]
        nk = num_kpoints[i]
        for j in range(nk):
            kpath.append(k0 + j * 1.0 / nk * (k1 - k0))
    kpath.append(hs_kpoints[-1])
    return np.array(kpath)


def gen_kdist(lattice_vectors, kpoints):
    """
    Convert k_path generated ny gen_path into distances in reciprocal space.

    :param lattice_vectors: 3*3 array, lattice vectors
    :param kpoints: N*3 array, fractional coordinates of kpoints
    :return: kdist: N*1 array, distance in reciprocal space
    """
    reciprocal_vectors = gen_reciprocal_vectors(lattice_vectors)
    kpoints_cartesian = frac2cart(reciprocal_vectors, kpoints)
    kdist = np.zeros(kpoints.shape[0])
    for i in range(1, kpoints.shape[0]):
        dk = kpoints_cartesian[i] - kpoints_cartesian[i-1]
        kdist[i] = kdist[i-1] + np.sqrt(np.sum(dk**2))
    return kdist


def gen_kmesh(grid_size):
    """
    Generate uniform meshgrid in the first Brillouin zone.

    :param grid_size:list with 3 integers specifying the dimension along each
            direction
    :return: kmesh: (grid_size[0] * grid_size[1] * grid_size[2]) *3 array
    """
    assert len(grid_size) == 3
    kmesh = np.array([[kx, ky, kz]
                     for kx in np.linspace(0, 1-1./grid_size[0], grid_size[0])
                     for ky in np.linspace(0, 1-1./grid_size[1], grid_size[1])
                     for kz in np.linspace(0, 1-1./grid_size[2], grid_size[2])])
    return kmesh
