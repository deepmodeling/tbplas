#! /usr/bin/env python

import tbplas as tb


# In previous example we have demonstrated how to build graphene nano-ribbons
# in armchair and zigzag configuration at primitive cell level using
# 'extend_prim_cell' and 'apply_pbc'. However, these functions are intended
# to small cells. For large cells, the 'SuperCell' and 'Sample' classes are
# recommended, which will be shown in this tutorial.

# Just as in the example of primitive cell, we need the rectangular cell to
# build nano-ribbons.
rect_cell = tb.make_graphene_rect()

# Create armchair graphene nano-ribbon.
gnr_am = tb.Sample(tb.SuperCell(rect_cell, dim=(3, 3, 1),
                                pbc=(False, True, False)))
gnr_am.plot()

# Similar for zigzag nano-ribbon.
gnr_zz = tb.Sample(tb.SuperCell(rect_cell, dim=(3, 3, 1),
                   pbc=(True, False, False)))
gnr_zz.plot()
