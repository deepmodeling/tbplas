.. _prim_bands:

Bands and DOS of primitive cell
===============================

In this tutorial we show how to calculate the band structure and density of states (DOS) of a primitive cell
using monolayer graphene as the example. First of all, we need to import all necessary packages:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import tbplas as tb

Of course, we can reuse the model built in previous tutorial. But this time we will import it from the materials
repository for convenientce:

.. code-block:: python

    cell = tb.make_graphene_diamond()

The first Brillouin zone of a hexagonal lattice with :math:`\gamma=60^\circ` is shown as below:

.. figure:: images/prim_bands/FBZ.png
    :align: center
    :scale: 50%

    Schematic plot of the first Brillouin zone of a hexagonal lattice. :math:`b_1` and :math:`b_2` are basis
    vectors of reciprocal lattice. Dashed hexagon indicates the edges of first Brillouin zone. The path of
    :math:`\Gamma \rightarrow M \rightarrow K \rightarrow \Gamma` is indicated with red arrows.

We generate a path of :math:`\Gamma \rightarrow M \rightarrow K \rightarrow \Gamma`, with 40 interpolated
k-points along each segment, using the following commands:

.. code-block:: python

    k_points = np.array([
        [0.0, 0.0, 0.0],    # Gamma
        [1./2, 0.0, 0.0],   # M
        [2./3, 1./3, 0.0],  # K
        [0.0, 0.0, 0.0],    # Gamma
    ])
    k_label = ["G", "M", "K", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])

Here ``k_points`` is an array contaning the fractional coordinates of highly-symmetric k-points, which
are then passed to :func:`.gen_kpath` for interpolation. ``k_path`` is an array containing interpolated
coordinates and ``k_idx`` contains the indices of highly-symmetric k-points in ``k_path``. ``k_label``
is a list of strings to label the k-points in the plot of band structure.

The band structure can be obtained by:

.. code-block:: python

    k_len, bands = cell.calc_bands(k_path)

and plotted by calling the :func:`plot_band` method of :class:`.Visualizer` class:

.. code-block:: python

    vis = tb.Visualizer()
    vis.plot_band(k_len, bands, k_idx, k_label)

or alternatively, using ``matplotlib`` directly:

.. code-block:: python

    num_bands = bands.shape[1]
    for i in range(num_bands):
        plt.plot(k_len, bands[:, i], color="r", linewidth=1.0)
    for idx in k_idx:
        plt.axvline(k_len[idx], color='k', linewidth=1.0)
    plt.xlim((0, np.amax(k_len)))
    plt.xticks(k_len[k_idx], k_label)
    plt.xlabel("k (1/nm)")
    plt.ylabel("Energy (eV)")
    plt.tight_layout()
    plt.show()
    plt.close()

.. figure:: images/prim_bands/bands.png
    :align: center

    Band structure of monolayer graphene.

To evaluate density of states (DOS) we need to generate a uniform k-mesh in the first Brillouin zone using the
:func:`.gen_kmesh` function:

.. code-block:: python

    k_mesh = tb.gen_kmesh((120, 120, 1))  # 120*120*1 uniform meshgrid

Then we can calculate and visulize DOS with the :func:`calc_dos` method of :class:`.Visualizer` class:

.. code-block:: python

    energies, dos = cell.calc_dos(k_mesh)
    vis.plot_dos(energies, dos)

Of course, we can also plot DOS using ``matplotlib`` directly:

.. code-block:: python

    plt.plot(energies, dos, linewidth=1.0)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (1/eV")
    plt.tight_layout()
    plt.show()
    plt.close()

.. figure:: images/prim_bands/dos.png
    :align: center
    :scale: 30%
