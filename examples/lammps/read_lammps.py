#! /usr/bin/env python

from scipy.spatial import KDTree
from ase import io

import tbplas as tb
from tbplas.builder.base import Hopping

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

# Safe but slow
# for pair in pairs:
#     prim_cell.add_hopping((0, 0, 0), pair[0], pair[1], energy=-2.7)

# Dangerous but fast (10^4 times faster)
# Luckily enough, KDTree returns only pairs with i < j. So there is no
# need to worry about redundant hopping terms.
hop_list = [Hopping((0, 0, 0), pair[0], pair[1], -2.7) for pair in pairs]
prim_cell.hopping_list = hop_list

# Plot the cell
prim_cell.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)
