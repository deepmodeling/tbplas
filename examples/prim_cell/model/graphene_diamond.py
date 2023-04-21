#! /usr/bin/env python

import tbplas as tb


# In this tutorial, we show how to create the diamond-shaped primitive cell of
# graphene. Firstly, we need to generate the lattice vectors. They can be input
# manually, or generated with the 'gen_lattice_vectors' function. Lengths are in
# Angstroms and angles are in degrees.
vectors = tb.gen_lattice_vectors(a=2.46, b=2.46, gamma=60)
print("lattice vectors in angstroms:")
print(vectors)

# Then we create a primitive cell and add orbitals. Note that coordinates of
# orbitals should be fractional. Since we are lattice vectors with the angle
# between a1 and a2 being 60 degrees, the fractional coordinates are (1/3, 1/3, 0)
# and (2/3, 2/3, 0).
cell = tb.PrimitiveCell(vectors, unit=tb.ANG)
cell.add_orbital((1./3, 1./3), 0.0)
cell.add_orbital((2./3, 2./3), 0.0)

# # Alternatively, we can add the orbitals using cartesian coordinates.
# cell.add_orbital_cart((1.23, 0.71014083), energy=0.0, unit=tb.ANG)
# cell.add_orbital_cart((2.46, 1.42028166), energy=0.0, unit=tb.ANG)
# print(cell.orb_pos_ang)

# Then we add hopping terms. Conjugate relation <i,0|H|j,R> = <j,0|H|i,-R>* are
# handled automatically. So we need to add only half of all the hopping terms.
cell.add_hopping((0, 0), 0, 1, -2.7)
cell.add_hopping((1, 0), 1, 0, -2.7)
cell.add_hopping((0, 1), 1, 0, -2.7)

# Now we can view the primitive cell just created.
cell.plot()
