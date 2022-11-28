#! /usr/bin/env python
"""Example for reading structure from lammps output."""

from scipy.spatial import KDTree
from ase import io

import tbplas as tb

# Read the last image of lammps dump
atoms = io.read("struct.atom", format="lammps-dump-text", index=-1)

# Get cell lattice vectors in Angstroms
lattice_vectors = atoms.cell

# Create the cell
prim_cell = tb.PrimitiveCell(lat_vec=atoms.cell, unit=tb.ANG)

# Add orbitals
for atom in atoms:
    prim_cell.add_orbital(atom.scaled_position)

# Get Cartesian Coordinates of all orbitals
orb_pos_ang = prim_cell.orb_pos_ang

# Detect nearest neighbours using KDTree
kd_tree = KDTree(orb_pos_ang)
pairs = kd_tree.query_pairs(r=1.45)

# Add hopping terms
for pair in pairs:
    prim_cell.add_hopping((0, 0, 0), pair[0], pair[1], energy=-2.7)

# Plot the cell
prim_cell.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)
