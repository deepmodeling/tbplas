#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb


def calc_hop(sk: tb.SK, rij: np.ndarray, distance: float,
             label_i: str, label_j: str, sk_params: np.ndarray) -> complex:
    """
    Calculate hopping integral based on Slater-Koster formulation.

    :param sk: SK instance
    :param rij: displacement vector from orbital i to j in nm
    :param distance: norm of rij
    :param label_i: label of orbital i
    :param label_j: label of orbital j
    :param sk_params: array containing SK parameters
    :return: hopping integral in eV
    """
    # 1st and 2nd hopping distances in nm
    d1 = 0.1419170044439990
    d2 = 0.2458074906840380
    if abs(distance - d1) < 1.0e-5:
        v_sss, v_sps, v_pps, v_ppp = sk_params[2:6]
    elif abs(distance - d2) < 1.0e-5:
        v_sss, v_sps, v_pps, v_ppp = sk_params[6:10]
    else:
        raise ValueError(f"Too large distance {distance}")
    return sk.eval(r=rij, label_i=label_i, label_j=label_j,
                   v_sss=v_sss, v_sps=v_sps,
                   v_pps=v_pps, v_ppp=v_ppp)


class MyFit(tb.ParamFit):
    def calc_bands_ref(self) -> np.ndarray:
        """
        Get reference band data for fitting.

        :return: band structure on self.k_points
        """
        cell = tb.wan2pc("graphene", hop_eng_cutoff=0.0)
        k_len, bands = cell.calc_bands(self.k_points)
        return bands

    def calc_bands_fit(self, sk_params: np.ndarray) -> np.ndarray:
        """
        Get band data of the model from given parameters.

        :param sk_params: array containing SK parameters
        :return: band structure on self.k_points
        """
        # Lattice constants and orbital info.
        lat_vec = np.array([
            [2.458075766398899, 0.000000000000000, 0.000000000000000],
            [-1.229037883199450, 2.128755065595607, 0.000000000000000],
            [0.000000000000000, 0.000000000000000, 15.000014072326660],
        ])
        orb_pos = np.array([
            [0.000000000, 0.000000000, 0.000000000],
            [0.666666667, 0.333333333, 0.000000000],
        ])
        orb_label = ("s", "px", "py", "pz")

        # Create the cell and add orbitals
        e_s, e_p = sk_params[0], sk_params[1]
        cell = tb.PrimitiveCell(lat_vec, unit=tb.ANG)
        for pos in orb_pos:
            for label in orb_label:
                if label == "s":
                    cell.add_orbital(pos, energy=e_s, label=label)
                else:
                    cell.add_orbital(pos, energy=e_p, label=label)

        # Add Hopping terms
        neighbors = tb.find_neighbors(cell, a_max=5, b_max=5,
                                      max_distance=0.25)
        sk = tb.SK()
        for term in neighbors:
            i, j = term.pair
            label_i = cell.get_orbital(i).label
            label_j = cell.get_orbital(j).label
            hop = calc_hop(sk, term.rij, term.distance, label_i, label_j,
                           sk_params)
            cell.add_hopping(term.rn, i, j, hop)

        # Evaluate band structure
        k_len, bands = cell.calc_bands(self.k_points)
        return bands


def main():
    # Fit the sk parameters
    # Reference:
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.245412
    k_points = tb.gen_kmesh((100, 100, 1))
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    fit = MyFit(k_points, weights)
    sk0 = np.array([-8.370, 0.0,
                    -5.729, 5.618, 6.050, -3.070,
                    0.102, -0.171, -0.377, 0.070])
    sk1 = fit.fit(sk0)

    # Plot fitted band structure
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [1./3, 1./3, 0.0],
        [1./2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    fit = MyFit(k_path, weights)
    bands_ref = fit.calc_bands_ref()
    bands_fit = fit.calc_bands_fit(sk1)
    num_bands = bands_ref.shape[1]
    for i in range(num_bands):
        plt.plot(bands_ref[:, i], color="red", linewidth=1.0)
        plt.plot(bands_fit[:, i], color="blue", linewidth=1.0)
    plt.show()


if __name__ == "__main__":
    main()