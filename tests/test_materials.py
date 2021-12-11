#! /usr/bin/env python

import unittest

import numpy as np
import matplotlib.pyplot as plt

import tbplas as tb
from tbplas.materials.xs2 import _gen_orb_labels


class TestMaterials(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test00_graphene(self):
        """
        Test utilities for building graphene sample.

        :return: None
        """
        prim_cell = tb.make_graphene_diamond()
        prim_cell.plot()
        prim_cell = tb.make_graphene_rect(from_scratch=True)
        prim_cell.plot()
        prim_cell = tb.make_graphene_rect(from_scratch=False)
        prim_cell.plot()

        # Test band structure
        prim_cell = tb.make_graphene_diamond()
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "K", "M", "G"]
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        tb.Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = tb.gen_kmesh((120, 120, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Test orbital labels
        prim_cell = tb.make_graphene_diamond()
        label_ref = ["C_pz" for _ in range(2)]
        label_test = [orb.label for orb in prim_cell.orbital_list]
        self.assertListEqual(label_ref, label_test)

        prim_cell = tb.make_graphene_rect(from_scratch=True)
        label_ref = ["C_pz" for _ in range(4)]
        label_test = [orb.label for orb in prim_cell.orbital_list]
        self.assertListEqual(label_ref, label_test)

        prim_cell = tb.make_graphene_rect(from_scratch=False)
        label_ref = ["C_pz" for _ in range(4)]
        label_test = [orb.label for orb in prim_cell.orbital_list]
        self.assertListEqual(label_ref, label_test)

    def test01_black_phosphorus(self):
        """
        Test utilities for constructing black phosphorus sample.

        :return: None
        """
        prim_cell = tb.make_black_phosphorus()

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        k_label = ["G", "X", "S", "Y", "G"]
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        tb.Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = tb.gen_kmesh((100, 100, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Test plot utilities
        sample = tb.Sample(tb.SuperCell(prim_cell, dim=(7, 5, 1)))
        for view in ("ab", "ba", "bc", "cb", "ca", "ac"):
            prim_cell.plot(view=view)
            sample.plot(view=view)

        # Test orbital labels
        label_ref = ["P_pz" for _ in range(4)]
        label_test = [orb.label for orb in prim_cell.orbital_list]
        self.assertListEqual(label_ref, label_test)

    def test02_antimonene(self):
        """
        Test utilities for constructing antimonene.

        :return: None
        """
        prim_cell = tb.make_antimonene(with_soc=True)
        prim_cell.plot()

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [1./2, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "M", "K", "G"]
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        tb.Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = tb.gen_kmesh((120, 120, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Test labels
        prim_cell = tb.make_antimonene(with_soc=False)
        label_ref = ["p11", "p12", "p13", "p21", "p22", "p23"]
        label_test = [orb.label for orb in prim_cell.orbital_list]
        self.assertListEqual(label_ref, label_test)

        prim_cell = tb.make_antimonene(with_soc=True)
        label_ref = ["p11+", "p12+", "p13+", "p11-", "p12-", "p13-",
                     "p21+", "p22+", "p23+", "p21-", "p22-", "p23-"]
        label_test = [orb.label for orb in prim_cell.orbital_list]

    def test03_tmdc(self):
        """
        Test utilities for constructing TMDC.

        :return: None
        """
        prim_cell = tb.make_tmdc("MoS2")

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [1./2, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "M", "K", "G"]
        k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        tb.Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = tb.gen_kmesh((120, 120, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Test orbital labels
        for material in ("MoS2", "MoSe2", "WS2", "WSe2"):
            label_ref = _gen_orb_labels(material)
            prim_cell = tb.make_tmdc(material)
            label_test = [orb.label for orb in prim_cell.orbital_list]
            self.assertListEqual(label_ref, label_test)


if __name__ == "__main__":
    unittest.main()
