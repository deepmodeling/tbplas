Introduction
============

This directory contains various of examples demonstrating the features and usages of TBPLaS.
All the examples can be run with ``./foo.py`` or ``python ./foo.py`` with ``foo.py`` being
the name of script. The example scripts using MPI should be run with ``mpirun -np N ./foo.py``
or ``mpirun -np N python ./foo.py``, where ``N`` is the number of MPI processes.


List of examples
----------------

* advanced
  * fractal: construction of Sierpinski carpet
  * quasi_crystal: construction of quasi-crystal from twisted bilayer graphene
  * tbg: construction of twisted bilayer graphene
* interface:
  * lammps: construction of primitive cell from the output of LAMMPS
  * wannier90: construction of primitive cell from the output of Wannier90, for graphene, MoS2
    and InSe, with parameters fitting example for graphene
* prim_cell:
  * band_dos: calculation of band structure and density of states for graphene and graphene
    nano-ribbon
  * model: construction of graphene primitive cell with diamond and rectangular shapes and
    graphene nano-ribbon at PrimitiveCell model
  * z2: calculation of Z2 topological invariant for graphene and bilayer bismuth
  * lindhard: calculation of response properties of graphene using lindhard function
  * sk_soc: usage of Slater-Koster formulation and intra-atom spin-orbital coupling
* sample:
  * band_dos: calculation of band structure and density of states for graphene and graphene
    nano-ribbon
  * model: construction of graphene, graphene nano-ribbon and models with vacancies and deformation
    at Sample level
  * tbpm: calculation of various properties using TBPM algorithms

The recommended order of studying the examples is: ``prim_cell`` -> ``sample`` -> ``advanced``.
For ``prim_cell`` and ``sample``, the order should be ``model`` -> ``band_dos`` -> other examples.
