Overview
========

In this section we give an overview on the concepts of TBPLaS, as well as the common workflow, to provide
the users and developers a general picture of its design philosophy.

Concepts
--------

TBPLaS has been implemented in an intuitive and comprehensible object-oriented manner, with classes directly
related to concepts of tight-binding theory. The layout of classes, as well as a more detailed diagram are
shown below. At the root of the hierarchy there are the classes of :class:`.Orbital` and :class:`.Hopping`,
representing the orbitals and hopping terms in a tight-binding model, respectively. From the orbitals and
hopping terms, as well as lattice vectors, a primitive cell can be created as instance of the :class:`.PrimitiveCell`
class. As mentioned in previous sections, primitive cells are the primary building blocks of bulk materials.
Many properties, including band structure, density of states, polarizability, dielectric function, optical
conductivity, can be obtained at primitive cell level, either by calling proper methods of :class:`.PrimitiveCell`
class, or with the help of :class:`.Lindhard` class.

.. figure:: images/class_layout.png
    :alt: class layout
    :align: center

    Layout of user classes of TBPLaS. Classes of same level in the hierarchy share the same color.

.. figure:: images/class_diagram.png
    :alt: class diagram
    :align: center

    User class diagram of TBPLaS. Only essential attributes are shown for clarity.

:class:`.SuperCell` / :class:`.InterHopping` / :class:`.Sample` are a set of classes specially designed
for constructing large complex models, especially for TBPM calculations. The computational expensive parts
of these classes are written in Cython, making them orders of magnitudes faster than the purely python-based
tools. For example, it takes 0.01 second to construct a graphene model with 100,000 primitive cells using
:class:`.SuperCell` / :class:`.Sample` class, while the python-based tools take 22.63 seconds. At :class:`.SuperCell`
level the user can specify the number of replicated primitive cells, boundary conditions, vacancies, additional
intra-hopping terms (instance of :class:`.IntraHopping` class), and modifiers to orbital positions. Heterogenous
systems, e.g., slabs with adatoms or hetero-structures with multiple layers, are modeled as separate supercells
plus inter-hopping terms (instance of :class:`.InterHopping` class). The :class:`.Sample` class is a unified
interface to both homogenous and heterogenous systems, from which the band structure and density of states can
be obtained via exact-diagonalization. Different kinds of perturbations, e.g., electric and magnetic fields,
strain, can be specified at :class:`.Sample` level. Also, it acts as the starting point for TBPM calculations.

The parameters of TBPM calculation are stored in the :class:`.Config` class. Based on the sample and configuration,
a solver and an analyzer can be created from :class:`.Solver` and :class:`.Analyzer` classes, respectively.
The main purpose of solver is to obtain correlation functions, which are then analyzed by the analyzer to yield
density of states, local density of states, AC conductivity, DC conductivity, Hall conductance, quasi-eigenstates,
etc. The results from TBPM calculations, as well as the results from exact-diagonalization at either :class:`.PrimitiveCell`
or :class:`.Sample` level, can be visualized using matplotlib, or alternatively, with the :class:`.Visualizer` class,
which is a wrapper over matplotlib functions.

Workflow
--------

The common workflow of TBPLaS is shown below. All calculations using TBPLaS begin with creating the primitive cell,
which involves specifying the lattice vectors, adding orbitals and adding hopping terms. TBPLaS uses translational
symmetry and conjugate relation to reduce the number of hopping terms, so only half of the terms are needed.
From the primitive cell we can calculate the band structure, density of states, polarizability, dielectric function,
and AC conductivity via exact diagonalization or Lindhard functions. We can also build complex models of moderate
size using Python-based tools, and evaluate properties like band structure and density of states in the same approach.
Strains and external fields can be implemented by directly modifying the orbital positions, on-site energies,
hopping terms and other attributes of the primitive cell instance.

If the model is much larger, we need to use the Cython-based :class:`.SuperCell` / :class:`.InterHopping` / :class:`.Sample`
classes to create a sample. Strains and external fields can be implemented in the same approach as for primitive cell.
From the sample we can band structure and density of states via exact diagonalization, or by TBPM with solver/analyzer.
Finally, we can visualize the results with the help of visualizer.

.. figure:: images/workflow.png
    :alt: workflow
    :align: center

    Workflow of TBPLaS
