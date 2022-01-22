Setting up primitive cell
=========================

In this tutorial we show how to set up a primitive cell from the :class:`.PrimitiveCell` class taking
monolayer graphene as the example. First of all, we need to import all necessary packages:

.. code-block:: python

    import math
    import numpy as np
    import tbplas as tb

From the :ref:`background` section you already know the ingredients of a primitive cell are lattice vectors,
orbitals and hopping terms. We begin with generating the lattice vectors from lattice parameters, namely
:math:`a`, :math:`b`, :math:`c`, :math:`\alpha`, :math:`\beta` and :math:`\gamma`. Graphene has two equivalent
sets of lattice parameters, with :math:`a=b=2.46 \overset{\circ}{\mathrm {A}}` and :math:`\alpha=\beta=90^\circ`.
The only difference is that :math:`\gamma` can be either :math:`60^\circ` or :math:`120^\circ`. Also, we
need to provide an arbitrary lattice paramter :math:`c` since all primitive cells are regarded as 3-dimensional
in TBPLaS. We will take :math:`\gamma=60^\circ` and :math:`c=10 \overset{\circ}{\mathrm {A}}`. The lattice vectors
can be generated with the :func:`gen_lattice_vectors` function

.. code-block:: python

    vectors = tb.gen_lattice_vectors(a=2.46, b=2.46, c=10.0, gamma=60)

Alternatively, we can specify them manually:

.. code-block:: python

    a = 2.46
    c = 10.0
    sqrt3 = math.sqrt(3)

    vectors = np.array([
        [a, 0, 0,],
        [0.5*a, 0.5*sqrt3*a, 0,],
        [0, 0, c]
    ])

Now we can create an empty primitive cell from the lattice vectors:

.. code-block:: python

    cell = tb.PrimitiveCell(vectors)

Then we add the orbitals. The primitive cell of graphene has two carbon atoms, with fractional coordinates of
:math:`(\frac{1}{3}, \frac{1}{3}, 0)` and :math:`(\frac{2}{3}, \frac{2}{3}, 0)`. In the 2-band model, each
carbon atom carries 1 :math:`p_z` orbital with on-site energy being zero. So we can add the orbitals in fractinal
coordinates:

.. code-block:: python

    cell.add_orbital([1./3, 1./3], 0.0)
    cell.add_orbital([2./3, 2./3], 0.0)

or alternatively in Cartesian coordinates

.. code-block:: python

    cell.add_orbital_cart([1.23, 0.71014083], unit=tb.ANG, energy=0.0)
    cell.add_orbital_cart([2.46, 1.42028166], unit=tb.ANG, energy=0.0)

Note that we must use keyword arguments ``unit`` and ``energy`` instead of positional arguments in this case.
Otherwise TBPLaS may be confused.

With the orbitals ready, we can add the hopping terms. As shown in the figure below, there are 6 hopping terms
between :math:`(0, 0)` and neighbouring cells in the nearest approximation:

* :math:`(0, 0) \leftarrow (0, 0), i=0, j=1`
* :math:`(0, 0) -> (0, 0), i=1, j=0`

.. figure:: images/graph_prim.png
    :align: center

    Hopping terms within :math:`(0, 0, 0)` cell and neighbouring cells.
