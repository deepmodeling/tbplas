Analytical Hamiltonian
======================

**NOTE: The diagonal terms of analytical Hamiltonian produced by print_hk is incorrect in TBPLaS v1.4.
Make sure you have installed v1.4.1 or newer versions when running this example.**

In this tutorial, we show how to get the analytical Hamiltonian and how to use it in calculations.
The :class:`.PrimitiveCell` class has a ``print_hk`` method which can print the formula of
Hamiltonian, while the :class:`.DiagSolver` class can utilize user-defined analytical Hamiltonian
to calculate the eigenvalues and eigenstates. The script is located at ``examples/advanced/analy_ham.py``.
We begin with importing the necessary packages:

.. code-block:: python

    from math import cos, sin, pi
    import numpy as np
    import tbplas as tb

Derivation of Hamiltonian
-------------------------

The analytical Hamiltonian can be obtained by calling the ``print_hk`` method, after the primitive
cell has been successfully created. We show this with the 2-band model of monolayer graphene:

.. code-block:: python
    :linenos:

    cell = tb.make_graphene_diamond()
    cell.print_hk(convention=1)
    cell.print_hk(convention=2)

The ``convention`` argument specifies the convention to evaluate the elements of Hamiltonian, and
should be either 1 or 2. For convention 1 the elements are

.. math::

    H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} H_{ij}(\mathbf{R})\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot(\mathbf{R}+\tau_j-\tau_i)}

while for convention 2

.. math::

    H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} H_{ij}(\mathbf{R})\mathrm{e}^{\mathrm{i}\mathbf{k}\cdot\mathbf{R}}


The output for convention 1 should look like

.. code-block:: text
    :linenos:

    ham[0, 0] = (0.0)
    ham[1, 1] = (0.0)
    ham[0, 1] = ((-2.7+0j) * exp_i(0.3333333333333333 * ka + 0.3333333333333333 * kb)
     + (-2.7-0j) * exp_i(-0.6666666666666667 * ka + 0.3333333333333333 * kb)
     + (-2.7-0j) * exp_i(0.3333333333333333 * ka - 0.6666666666666667 * kb))
    ham[1, 0] = ham[0, 1].conjugate()
    with exp_i(x) := cos(2 * pi * x) + 1j * sin(2 * pi * x)

and for convention 2

.. code-block:: text
    :linenos:

    ham[0, 0] = (0.0)
    ham[1, 1] = (0.0)
    ham[0, 1] = ((-2.7+0j) * 1
     + (-2.7-0j) * exp_i(-ka)
     + (-2.7-0j) * exp_i(-kb))
    ham[1, 0] = ham[0, 1].conjugate()
    with exp_i(x) := cos(2 * pi * x) + 1j * sin(2 * pi * x)

where the formula for each element of the Hamiltonian matrix is printed. Note that we have defined the
function :math:`\exp(\mathrm{i} \cdot 2\pi \cdot x)` using Euler formula since the ``exp``
function of ``numpy`` is rather slow if the input is not an array. We will utilize this function in
the next section.

Usage of Hamiltonian
--------------------

TBPLaS provides the :class:`.FakePC` class for holding the analytical Hamiltonian. Users should derive
their own class from this class and implement the analytical Hamiltonian in the ``set_ham_dense`` and
``set_ham_csr`` methods. The ``set_ham_dense`` method should accept the fractional coordinate of k-point
and the Hamiltonian matrix as input, and modifies the Hamiltonian in-place. On the contrary, the
``set_ham_csr`` function accepts only the fractional coordinate of k-point as input, and returns a new
Hamiltonian matrix in CSR format.

We define the following function and class from the analytical Hamiltonians in previous section

.. code-block:: python
    :linenos:

    def exp_i(x: float) -> complex:
        """
        Evaluate exp(i*2pi*x) using Euler formula.

        :param x: incoming x
        :return: exp(i*2pi*x)
        """
        return cos(2 * pi * x) + 1j * sin(2 * pi * x)


    class FakePC(tb.FakePC):
        def set_ham_dense(self, kpt: np.ndarray,
                          ham: np.ndarray,
                          convention: int = 1) -> None:
            ka, kb = kpt.item(0), kpt.item(1)
            ham[0, 0] = 0.0
            ham[1, 1] = 0.0
            if convention == 1:
                ham[0, 1] = -2.7 * (exp_i(1. / 3 * ka + 1. / 3 * kb) +
                                    exp_i(-2. / 3 * ka + 1. / 3 * kb) +
                                    exp_i(1. / 3 * ka - 2. / 3 * kb))
                ham[1, 0] = ham[0, 1].conjugate()
            else:
                ham[0, 1] = -2.7 * (1.0 + exp_i(-ka) + exp_i(-kb))
                ham[1, 0] = ham[0, 1].conjugate()


To demonstrate the usage of analytical Hamiltonian, we create a fake graphene primitive cell with 2
orbitals. Then we create a ``solver`` from the :class:`.DiagSolver` class using the fake primitive cell
with analytical Hamiltonian

.. code-block:: python
    :linenos:

    # Create fake primitive cell, solver and visualizer
    vectors = tb.gen_lattice_vectors(a=0.246, b=0.246, c=1.0, gamma=60)
    cell = FakePC(num_orb=2, lat_vec=vectors, unit=tb.NM)
    solver = tb.DiagSolver(cell)
    vis = tb.Visualizer()

The band structure from the analytical Hamiltonian can be obtained by calling the ``calc_bands``
method of the :class:`.DiagSolver` class

.. code-block:: python
    :linenos:

    # Evaluation of band structure
    k_points = np.array([
        [0.0, 0.0, 0.0],
        [2. / 3, 1. / 3, 0.0],
        [1. / 2, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    k_label = ["G", "M", "K", "G"]
    k_path, k_idx = tb.gen_kpath(k_points, [40, 40, 40])
    for convention in (1, 2):
        k_len, bands = solver.calc_bands(k_path, convention)[:2]
        vis.plot_bands(k_len, bands, k_idx, k_label)

The evaluation of DOS is similar

.. code-block:: python
    :linenos:

    # Evaluation of DOS
    k_mesh = tb.gen_kmesh((120, 120, 1))
    for convention in (1, 2):
        energies, dos = solver.calc_dos(k_mesh, convention=convention)
        vis.plot_dos(energies, dos)

Both conventions produce the same band structure and DOS as in :ref:`prim_bands`.
