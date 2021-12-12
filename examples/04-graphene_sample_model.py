#! /usr/bin/env python

import tbplas as tb


# First we import the primitive cell of graphene from the repository.
prim_cell = tb.make_graphene_diamond()

# Then we create a supercell from the primitive cell with size of 6*6*1
# and open boundary conditions.
super_cell = tb.SuperCell(prim_cell, dim=(6, 6, 1), pbc=(False, False, False))

# Then we create the sample and have a look at it.
sample = tb.Sample(super_cell)
sample.plot()

# For large sample it is recommended to disable some features to boost
# the plot. Pay attention to the differences.
sample.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)

# Now we create a sample with periodic boundary conditions along x and y
# directions. Pay attention to the difference.
super_cell_pbc = tb.SuperCell(prim_cell, dim=(6, 6, 1), pbc=(True, True, True))
sample_pbc = tb.Sample(super_cell_pbc)
sample_pbc.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False)
