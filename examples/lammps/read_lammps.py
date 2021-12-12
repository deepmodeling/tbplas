#! /usr/bin/env python

from ase import io


# Read the last image of lammps dump
atoms = io.read("output.atom", format="lammps-dump-text", index=-1)

# Get cell lattice vectors in Angstroms
lattice_vectors = atoms.cell

# Get symbols of atoms
atom_symbols = atoms.symbols

# Get Cartesian coordinates of atoms in Angstrom
cart_coord = atoms.get_positions()

# Get fractional coordinates
frac_coord = atoms.get_scaled_positions()
