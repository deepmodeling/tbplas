#! /usr/bin/env python

import unittest

import numpy as np
import matplotlib.pyplot as plt

import tbplas.builder.kpoints as kpt
from tbplas.materials.graphene import (make_graphene_diamond,
                                       make_graphene_rect)
from tbplas.materials.phosphorene import make_black_phosphorus
from tbplas.materials.antimonene import make_antimonene
from tbplas.materials.xs2 import make_tmdc
from tbplas.builder import SuperCell, Sample
from tbplas.visual import Visualizer


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
        prim_cell = make_graphene_diamond()
        prim_cell.plot()
        prim_cell = make_graphene_rect(from_scratch=True)
        prim_cell.plot()
        prim_cell = make_graphene_rect(from_scratch=False)
        prim_cell.plot()

        # Test band structure
        prim_cell = make_graphene_diamond()
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [1./2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "K", "M", "G"]
        k_path, k_idx = kpt.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = kpt.gen_kmesh((120, 120, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

    def test01_black_phosphorus(self):
        """
        Test utilities for constructing black phosphorus sample.

        :return: None
        """
        prim_cell = make_black_phosphorus()

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        k_label = ["G", "X", "S", "Y", "G"]
        k_path, k_idx = kpt.gen_kpath(k_points, [40, 40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = kpt.gen_kmesh((100, 100, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

        # Test plot utilities
        sample = Sample(SuperCell(prim_cell, dim=(7, 5, 1)))
        for view in ("ab", "ba", "bc", "cb", "ca", "ac"):
            prim_cell.plot(view=view)
            sample.plot(view=view)

    def test02_antimonene(self):
        """
        Test utilities for constructing antimonene.

        :return: None
        """
        prim_cell = make_antimonene(with_soc=True)
        prim_cell.plot()

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [1./2, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "M", "K", "G"]
        k_path, k_idx = kpt.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = kpt.gen_kmesh((120, 120, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

    def test03_tmdc(self):
        """
        Test utilities for constructing TMDC.

        :return: None
        """
        prim_cell = make_tmdc("MoS2")

        # Test band structure
        k_points = np.array([
            [0.0, 0.0, 0.0],
            [1./2, 0.0, 0.0],
            [2./3, 1./3, 0.0],
            [0.0, 0.0, 0.0],
        ])
        k_label = ["G", "M", "K", "G"]
        k_path, k_idx = kpt.gen_kpath(k_points, [40, 40, 40])
        k_len, bands = prim_cell.calc_bands(k_path)
        Visualizer().plot_band(k_len, bands, k_idx, k_label)

        # Test DOS
        k_points = kpt.gen_kmesh((120, 120, 1))
        energies, dos = prim_cell.calc_dos(k_points)
        plt.plot(energies, dos)
        plt.show()

if __name__ == "__main__":
    unittest.main()
