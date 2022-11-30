Quasi-crystal
=============

In this tutorial, we show how to construct the quasi-crystal, in which we also need to shift
twist, reshape and merge the cells. Taking bilayer graphene quasicrystal as an example, a
quasicrystal with 12-fold symmtery is formed by twisting one layer by :math:`30^\circ` with respect
to the center of :math:`\mathbf{c} = \frac{2}{3}\mathbf{a}_1 + \frac{2}{3}\mathbf{a}_2`, where
:math:`\mathbf{a}_1` and :math:`\mathbf{a}_2` are the lattice vectors of the primitive cell of
fixed layer. The script can be found at ``examples/advanced/quasi_crystal.py``. We begin with
importing the packages and defining the geometric parameters:

.. code-block:: python

    import math

    import numpy as np
    from numpy.linalg import norm

    import tbplas as tb


    angle = 30 / 180 * math.pi
    center = (2./3, 2./3, 0)
    radius = 3.0
    shift = 0.3349
    dim = (33, 33, 1)

Here ``angle`` is the twisting angle and ``center`` is the fractional coordinate of twisting
center. The radius of the quasicrystal is controlled by ``radius``, while ``shift`` specifies the
interlayer distance. We need a large cell to hold the quasicrystal, whose dimension is given in
``dim``. After introducing the parameters, we build the fixed and twisted layers by:

.. code-block:: python

    # Build the layers
    prim_cell = tb.make_graphene_diamond()
    layer_fixed = tb.extend_prim_cell(prim_cell, dim=dim)
    layer_twisted = tb.extend_prim_cell(prim_cell, dim=dim)

Then we shift and rotate the twisted layer with respect to the center and reshape it to the lattice
vectors of fixed layer:

.. code-block:: python

    # Get the Cartesian coordinate of twisting center
    center = np.array([dim[0]//2, dim[1]//2, 0]) + center
    center = np.matmul(center, prim_cell.lat_vec)

    # Twist, shift and reshape top layer
    tb.spiral_prim_cell(layer_twisted, angle=angle, center=center, shift=shift)
    conv_mat = np.matmul(layer_fixed.lat_vec, np.linalg.inv(layer_twisted.lat_vec))
    layer_twisted = tb.reshape_prim_cell(layer_twisted, conv_mat)

Since we have extended the primitive cell by :math:`33\times33\times1` times, and we want the
quasicrystal to be located in the center of the cell, we need to convert the coordinate of twisting
center in line 2-3. The twisting operation is done by the :func:`.spiral_prim_cell` function, where
the Cartesian coordinate of the center is given in the ``center`` argument. The fixed and twisted
layers have the same lattice vectors after reshaping, so we can merge them safely:

.. code-block:: python

    # Merge the layers
    final_cell = tb.merge_prim_cell(layer_twisted, layer_fixed)

Then we remove unnecessary orbitals to produce a round quasicrystal with finite radius. This is
done by a loop over orbital positions to collect the indices of unnecessary orbitals, and function
calls to ``remove_orbitals`` and ``trim`` functions:

.. code-block:: python

    # Remove unnecessary orbitals
    idx_remove = []
    orb_pos = final_cell.orb_pos_nm
    for i, pos in enumerate(orb_pos):
        if np.linalg.norm(pos[:2] - center[:2]) > radius:
            idx_remove.append(i)
    final_cell.remove_orbitals(idx_remove)

    # Remove dangling orbitals
    final_cell.trim()

Finally, we extend the hoppings and visualize the quasicrystal:

.. code-block:: python

    # Extend and visualize the model
    extend_hop(final_cell)
    final_cell.plot(with_cells=False, with_orbitals=False, hop_as_arrows=False, hop_eng_cutoff=0.3)

where the ``extend_hop`` function is defined in :ref:`hetero_model`. The output is shown in
following figure:

.. figure:: images/quasi_crystal/quasi_crystal.png
    :align: center
    :scale: 50%

    Plot of the quasicrystal formed from the incommensurate :math:`30^\circ` twisted bilayer
    graphene with a radius of 3 nm.