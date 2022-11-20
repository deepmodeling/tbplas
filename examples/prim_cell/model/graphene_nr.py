#! /usr/bin/env python

import tbplas as tb


# To build graphene nano-ribbons we need the rectangular cell.
# We import it from the material repository. Alternatively, we
# can reuse the model we created in previous example.
rect_cell = tb.make_graphene_rect()

# Then we extend the rectangular cell by calling 'extend_prim_cell'.
gnr_am = tb.extend_prim_cell(rect_cell, dim=(3, 3, 1))

# We make an armchair graphene nano-ribbon by removing hopping terms along x
# direction.
gnr_am.apply_pbc(pbc=(False, True, False))
gnr_am.plot()

# Similarly, we can make a zigzag nano-ribbon by removing hopping terms along
# y direction.
gnr_zz = tb.extend_prim_cell(rect_cell, dim=(3, 3, 1))
gnr_zz.apply_pbc(pbc=(True, False, False))
gnr_zz.plot()
