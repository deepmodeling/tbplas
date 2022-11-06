import unittest
from math import exp

import numpy as np
from numpy.random import random
from numpy.linalg import norm

import tbplas as tb


def calc_hop_mos2(sk, rij, label_i, label_j):
    """
    Evaluate the hopping integral <i,0|H|j,r> for single layer MoS2.

    Reference:
    [1] https://www.mdpi.com/2076-3417/6/10/284
    [2] https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.075409

    :param tb.SK sk: SK instance
    :param np.ndarray rij: displacement vector from orbital i to j in nm
    :param str label_i: label of orbital i
    :param str label_j: label of orbital j
    :return: hopping integral in eV
    :rtype: complex
    """
    # Parameters from ref. 1
    v_pps, v_ppp = 1.225, -0.467
    v_pds, v_pdp = 3.689, -1.241
    v_dds, v_ddp, v_ddd = -0.895, 0.252, 0.228

    # # Parameters from ref. 2
    # v_pps, v_ppp = 0.696, 0.278
    # v_pds, v_pdp = -2.619, -1.396
    # v_dds, v_ddp, v_ddd = -0.933, -0.478, -0.442

    return sk.eval(r=rij, label_i=label_i, label_j=label_j,
                   v_pps=v_pps, v_ppp=v_ppp,
                   v_pds=v_pds, v_pdp=v_pdp,
                   v_dds=v_dds, v_ddp=v_ddp, v_ddd=v_ddd)


def calc_hop_bp(sk, rij, label_i, label_j):
    """
    Evaluate the hopping integral <i,0|H|j,r> for single layer black phosphorus.

    Reference:
    https://www.sciencedirect.com/science/article/pii/S0927025617306705

    :param tb.SK sk: SK instance
    :param np.ndarray rij: displacement vector from orbital i to j in nm
    :param str label_i: label of orbital i
    :param str label_j: label of orbital j
    :return: hopping integral in eV
    :rtype: complex
    """
    r = norm(rij)
    r0 = 0.2224
    v_sss = -1.59 * exp(-(r - r0) / 0.033)
    v_sps = 2.39 * exp(-(r - r0) / 0.053)
    v_pps = 4.03 * exp(-(r - r0) / 0.058)
    v_ppp = -1.14 * exp(-(r - r0) / 0.053)
    return sk.eval(r=rij, label_i=label_i, label_j=label_j,
                   v_sss=v_sss, v_sps=v_sps,
                   v_pps=v_pps, v_ppp=v_ppp)


class TestSK(unittest.TestCase):
    def test_eval(self):
        """Test if the common interface 'eval' works as expected."""
        sk = tb.SK()
        p_labels = ("px", "py", "pz")
        d_labels = ("dx2-y2", "dxy", "dyz", "dzx", "dz2")

        def _check(*t):
            for j in range(len(t)-1):
                x1, x2 = t[j], t[j+1]
                self.assertAlmostEqual(x1.real, x2.real)
                self.assertAlmostEqual(x1.imag, x2.imag)

        for i in range(1000):
            # Generate random vectors and parameters
            r = (random(3) - 0.5) * 10
            v_sss, v_sps, v_sds = random(3)
            v_pps, v_ppp, v_pds, v_pdp = random(4)
            v_dds, v_ddp, v_ddd = random(3)

            # Test ss
            t1 = sk.ss(v_sss=v_sss)
            t2 = sk.eval(r=r, label_i="s", label_j="s", v_sss=v_sss)
            _check(t1, t2)

            # Test pp
            for p1 in p_labels:
                for p2 in p_labels:
                    t1 = sk.pp(r, label_i=p1, label_j=p2, v_pps=v_pps, v_ppp=v_ppp)
                    t2 = sk.eval(r, label_i=p1, label_j=p2, v_pps=v_pps, v_ppp=v_ppp)
                    _check(t1, t2)

            # Test dd
            for i1 in range(5):
                d1 = d_labels[i1]
                for i2 in range(i1, 5):
                    d2 = d_labels[i2]
                    t1 = sk.dd(r, label_i=d1, label_j=d2, v_dds=v_dds, v_ddp=v_ddp, v_ddd=v_ddd)
                    t2 = sk.eval(r, label_i=d1, label_j=d2, v_dds=v_dds, v_ddp=v_ddp, v_ddd=v_ddd)
                    _check(t1, t2)

            # Test sp and ps
            for p_label in p_labels:
                t1 = sk.sp(r, label_p=p_label, v_sps=v_sps)
                t2 = sk.eval(r, label_i="s", label_j=p_label, v_sps=v_sps)
                _check(t1, t2)

                t1 = sk.ps(r, label_p=p_label, v_sps=v_sps)
                t2 = sk.eval(r, label_i=p_label, label_j="s", v_sps=v_sps)
                _check(t1, t2)

            # Test sd and ds
            for d_label in d_labels:
                t1 = sk.sd(r, label_d=d_label, v_sds=v_sds)
                t2 = sk.eval(r, label_i="s", label_j=d_label, v_sds=v_sds)
                _check(t1, t2)

                t1 = sk.ds(r, label_d=d_label, v_sds=v_sds)
                t2 = sk.eval(r, label_i=d_label, label_j="s", v_sds=v_sds)
                _check(t1, t2)

            # Test pd and dp
            for p in p_labels:
                for d in d_labels:
                    t1 = sk.pd(r, label_p=p, label_d=d, v_pds=v_pds, v_pdp=v_pdp)
                    t2 = sk.eval(r, label_i=p, label_j=d, v_pds=v_pds, v_pdp=v_pdp)
                    _check(t1, t2)

                    t1 = sk.dp(r, label_p=p, label_d=d, v_pds=v_pds, v_pdp=v_pdp)
                    t2 = sk.eval(r, label_i=d, label_j=p, v_pds=v_pds, v_pdp=v_pdp)
                    _check(t1, t2)

    def test_mos2(self):
        """Test 'pp', 'pd' and 'dd' methods of SK using MoS2 as the example."""
        # Lattice vectors
        vectors = np.array([
            [3.181400000, 0.000000000, 0.000000000],
            [-1.590690383, 2.755178772, 0.000000000],
            [0.000000000, 0.000000000, 15.900000000],
        ])

        # Orbital coordinates
        coord_mo = [0.666670000,  0.333330000,  0.578250000]
        coord_s1 = [0.000000000, -0.000000000,  0.480070000]
        coord_s2 = [0.000000000, -0.000000000,  0.676430000]
        orbital_coord = [coord_mo for _ in range(5)]
        orbital_coord.extend([coord_s1 for _ in range(3)])
        orbital_coord.extend([coord_s2 for _ in range(3)])
        orbital_coord = np.array(orbital_coord)

        # Orbital labels
        d_orbitals = ("dz2", "dzx", "dyz", "dx2-y2", "dxy")
        p_orbitals = ("px", "py", "pz")
        orbital_label = d_orbitals + p_orbitals * 2

        # Orbital energies
        # Parameters from ref. 1:
        orbital_energy = {"dz2": -1.094, "dzx": -0.050, "dyz": -0.050,
                          "dx2-y2": -1.511, "dxy": -1.511,
                          "px": -3.559, "py": -3.559, "pz": -6.886}

        # # Parameters from ref. 2:
        # orbital_energy = {"dz2": -1.512, "dzx": 0.0, "dyz": 0.0,
        #                   "dx2-y2": -3.025, "dxy": -3.025,
        #                   "px": -1.276, "py": -1.276, "pz": -8.236}

        # Create the primitive cell and add orbitals
        cell = tb.PrimitiveCell(lat_vec=vectors, unit=tb.ANG)
        for i, label in enumerate(orbital_label):
            coord = orbital_coord[i]
            energy = orbital_energy[label]
            cell.add_orbital(coord, energy=energy, label=label)

        # Get hopping terms in the nearest approximation
        neighbors = tb.find_neighbors(cell, cell, a_max=2, b_max=2,
                                      max_distance=0.32)

        # Add hopping terms
        sk = tb.SK()
        for term in neighbors:
            i, j = term.pair
            label_i = cell.get_orbital(i).label
            label_j = cell.get_orbital(j).label
            hop = calc_hop_mos2(sk, term.rij, label_i, label_j)
            cell.add_hopping(term.rn, i, j, hop)

        # Plot band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [1./2, 0.0, 0.0],
            [1./3, 1./3, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "M", "K", "G"]
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = cell.calc_bands(k_path)
        tb.Visualizer().plot_bands(k_len, bands, k_idx, k_label)

    def test_bp(self):
        """Test 'ss', 'sp' and 'pp' methods of SK using BP as the example."""
        # Lattice vectors
        vectors = np.array([
            [4.623311297, 0.000000000, 0.000000000],
            [0.000000000, 3.299154095, 0.000000000],
            [0.000000000, 0.000000000, 20.478000000],
        ])

        # Orbital coordinates
        coord_p = np.array([
            [0.839001067, 0.250000000, 0.448627136],
            [0.339001577, 0.750000000, 0.551372864],
            [0.660998423, 0.250000000, 0.551372864],
            [0.160998933, 0.750000000, 0.448627136],
        ])
        orbital_coord = [row for row in coord_p for _ in range(4)]

        # Orbital labels
        orbital_label = ("s", "px", "py", "pz") * 4

        # Orbital energies
        orbital_energy = {"s": -8.80, "px": 0.0, "py": 0.0, "pz": 0.0}

        # Create primitive cell and add orbital
        cell = tb.PrimitiveCell(lat_vec=vectors, unit=tb.ANG)
        for i, label in enumerate(orbital_label):
            coord = orbital_coord[i]
            energy = orbital_energy[label]
            cell.add_orbital(coord, energy=energy, label=label)

        # Get hopping terms in the nearest approximation
        neighbors = tb.find_neighbors(cell, cell, a_max=2, b_max=2,
                                      max_distance=1.0)

        # Add hopping terms
        sk = tb.SK()
        for term in neighbors:
            i, j = term.pair
            label_i = cell.get_orbital(i).label
            label_j = cell.get_orbital(j).label
            hop = calc_hop_bp(sk, term.rij, label_i, label_j)
            cell.add_hopping(term.rn, i, j, hop)

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0]
        ])
        k_label = ["G", "X", "S", "Y", "G"]
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40, 40])
        k_len, bands = cell.calc_bands(k_path)
        tb.Visualizer().plot_bands(k_len, bands, k_idx, k_label)


if __name__ == "__main__":
    unittest.main()
