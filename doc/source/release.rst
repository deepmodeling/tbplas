Release Notes
=============

v1.6 | 2024-01-17
-----------------

New features
------------

* ``plot_wfc3d`` of :class:`.Visualizer` class now works for periodic systems.
* New :class:`.SOCTable2` class for evaluating spin-orbital coupling terms.
* New build system based on scikit-build-core and cmake.

Changes
-------

* Dropped support for Python 3.6. Now Python >= 3.7 is required to install TBPLaS.

v1.5 | 2023-08-17
-----------------

New features
^^^^^^^^^^^^

* :class:`.Visualizer` get as new method ``plot_wfc3d`` to plot three-dimensional wavefunctions
  as Gaussian cubes.
* All exact-diagonalization methods support non-orthogonal basis sets with the help of new
  :class:`.Overlap` class.
* Added interface for reading Hamiltonian and overlap from the output of DeepH. See
  ``examples/interface/deeph`` for more details.

Improvements
^^^^^^^^^^^^

* Efficiency of :func:`.wan2pc` significantly enhanced.

Changes
^^^^^^^

* Analytical Hamiltonian now should be implemented using the :class:`.FakePC` class.

Bugs fixed
^^^^^^^^^^

* Fixed incorrect restriction on the dimension of supercell.

v1.4.2 | 2023-07-25
-------------------

New features
^^^^^^^^^^^^

* Orbitals can be colored using user-defined coloring function via the ``orb_color`` argument
  for the ``plot`` method of :class:`.PrimitiveCell` and :class:`.Sample` classes.
* New :func:`.make_mos2_soc` function for making MoS2 primitive cell with SOC.
* :class:`.Lindhard` and :class:`.Analyzer` classes have a new ``calc_epsilon_q0`` method to
  calculate epsilon from AC conductivity for q=0.
* :class:`.Lindhard` gets a new ``energy_min`` argument for setting the lower bound of energy
  grid during initialization.

Improvements
^^^^^^^^^^^^

* Improved output of ``print`` and ``print_hk`` methods of :class:`.PrimitiveCell` class.
* Simplified :class:`.SOC` and :class:`.SOCTable` classes.
* Improved the observer pattern for keeping data consistency.

Changes
^^^^^^^

* :class:`.PCInterHopping` and :class:`.SCInterHopping` classes moved to ``primitive`` and
  ``super`` modules.

Bugs fixed
^^^^^^^^^^

* Hamiltonian from ``set_ham_dense`` and ``set_ham_csr`` methods of :class:`.Sample` class
  does not consider the rescaling factor.

v1.4.1 | 2023-06-14
-------------------

Improvements
^^^^^^^^^^^^

* Simplified ``sync_array`` methods of :class:`.PrimitiveCell` and :class:`.SuperCell`.
* :class:`.PrimitiveCell` and relevant modeling tools are more robust for empty primitive cells.

Changes
^^^^^^^

* Array attributes of :class:`.PrimitiveCell` and :class:`.SuperCell` are initialized as empty
  arrays rather than None.

Bugs fixed
^^^^^^^^^^

* Diagonal terms in output of ``print_hk`` of :class:`.PrimitiveCell` is incorrect.

Examples
^^^^^^^^

* Add example for analytical Hamiltonian.

v1.4 | 2023-06-08
-----------------

New features
^^^^^^^^^^^^

.. rubric:: Modeling tools

* The :class:`.PrimitiveCell` class gets a new attribute ``origin`` for representing the origin
  of lattice vectors and a new method ``reset_lattice`` to reset the lattice vectors. Setting up
  complex models is much easier and more flexible.
* The :class:`.PrimitiveCell` class gets a new method ``print_hk`` for printing the analytical
  Hamiltonian of the model.
* New :class:`.SOCTable` class for boosting the evaluation of intra-atom spin-orbital coupling terms.
* New :func:`.make_graphene_soc` function for getting the graphene model with Rashba and Kane-Mele
  spin-orbital coupling.
* Models built from the :class:`.Sample` class can be saved to and loaded from files with the
  ``save_array`` and ``load_array`` methods.
* The k-point of Hamiltonian of models of :class:`.Sample` class can be set up with the
  ``set_k_point`` method.
* The :class:`.Visualizer` class can plot scalar and vector fields with the ``plot_scalar`` and
  ``plot_vector`` methods, which are particularly useful for visualizing spin textures.

.. rubric:: Property calculators

* New :class:`.SpinTexture` class for calculating spin texture.
* New :class:`.DiagSolver` class for calculating energies, wavefunctions and density of states, which
  supports analytical Hamiltonian.

Improvments
^^^^^^^^^^^

* Legacy :class:`.HopDict` class has been refactored to support dictionary-like operations.
* New algorithm for building the hopping terms of :class:`.SuperCell` in general cases (100 times faster).
* The ``plot`` method of :class:`.Sample` class can plot conjugate hopping terms as well.
* Timestep for the ``calc_psi_t`` method of :class:`.Solver` class can be specified through the
  ``dt_scale`` argument.
* The ``plot_wfc`` method of :class:`.Visualizer` class can show the model alongside the wavefunction.

Changes
^^^^^^^

* ``get_dr`` methods of :class:`.SuperCell` and :class:`.SCInterHopping` classes have beem merged into
  ``get_hop`` method.
* ``init_dr`` method of :class:`.Sample` class has been merged into ``init_hop`` method accordingly.

Bugs fixed
^^^^^^^^^^

* ``read_config`` does not back up the names of legal parameters.

Examples
^^^^^^^^

* All examples have been reviewed and updated to the latest API.
* New example for calculating spin texture of graphene with Rashba and Kane-Mele SOC.
* New example for calculating quasi_eigenstates.

For developers
^^^^^^^^^^^^^^

* Added type hints for all the classes and functions.
* Implemented observer pattern for keeping data consistency. The original top-down approach has also been
  reviewed and improved.
* Redesigned the interfaces of all the classes, with instance attributes made private whenever possible.
  Now the attributes should accessed via the ``get_*`` methods or as properties.
* The ``get_*`` methods and properties of :class:`.PrimitiveCell` and :class:`.SuperCell` call ``sync_array``
  automatically. No need to call ``sync_array`` manually any more.
* Reorganized package structure

  * Physical constants, lattice and k-point utilities have been moved to the ``base`` package.
  * Interfaces to other codes have been moved to the ``adapter`` package.
  * Cython extension has been broken into smaller parts and moved to the ``Cython`` package.
  * Exact diagonalization modules have been moved to the ``diaognal`` package.
  * TBPM modules have been moved to the ``tbpm`` package.

* All methods involving exact diagonalization are now based the :class:`.DiagSolver` class. User-defined
  calculators should be derived from this class.

v1.3 | 2022-12-01
-----------------

New features
^^^^^^^^^^^^

* Added :class:`.SK` class for setting hopping integrals with Slater-Koster formulation
* Added :class:`.ParamFit` class for fitting on-site energies and hopping integrals
* Added :class:`.SOC` class for adding intra-atom spin-orbital coupling
* Added :func:`.make_graphene_sp` for making the 8-orbital model of graphene
* :class:`.Config`, :class:`.Solver` and :class:`.Analyzer` now checks for undefined parameters
* New algorithm for building the hopping terms of :class:`.SuperCell` (50 times faster)
* :class:`.Visualizer` gets a new ``plot_phases`` method to plot the topological phases from Z2

Improvments
^^^^^^^^^^^

* Redesigned :class:`.Z2` for calculating and analyzing the Z2 topological invariant
* Updated the tutorials with a lot of new examples demonstrating the new features

v1.2 | 2022-09-02
-----------------

New features
^^^^^^^^^^^^

* Added example for calculating Z2 topological invariant
* Added ``log`` method to :class:`.Lindhard`, :class:`.Solver` and :class:`.Analyzer`
  for reporting time and date

Improvments
^^^^^^^^^^^

* Removed unnecessary MPI_Allreduce calls in :class:`.Lindhard`

Changes
^^^^^^^

* Legacy :class:`.HopDict` class no longer handles conjugate terms automatically.

v1.1 | 2022-08-13
-----------------

New features
^^^^^^^^^^^^

* New :class:`.Lindhard` class for evaluating response properties using Lindhard function.
* Implemented LDOS calculation based exact diagonalization.
* Implemented propagation of wave function from initial condition.
* Implemented evaluation of diffusion coeffcients from DC correlation function.
* Added MPI support for band structure and DOS calculation.
* Added support for 64-bit array indices (samples can be much larger).

Improvments
^^^^^^^^^^^

* A lot of classes have been refactored for simplicity, maintainability and efficiency.
* The default values of common parameters and the units of outputs have been unified for exact
  diagonalization, Lindhard and TBPM subroutines.
* References to papers discussing the methodologies have been revised.
* :func:`merge_prim_cell` checks lattice vectors before merging cells.
* ``plot`` method of :class:`.Sample` accepts lists of colors for plotting the supercells and
  inter-cell hopping terms.
* DC conductivity subroutine is refactored and much faster.

Changes
^^^^^^^

* The ``IntraHopping`` class has beem removed. Modifications to hopping terms are now handled
  by the supercell itself.
* The ``InterHopping`` class has been renamed to :class:`.SCInterHopping`.
* The ``InterHopDict`` class has been renamed to :class:`.PCInterHopping`.
* ``apply_pbc`` and ``trim_prim_cell`` functions are moved to :class:`.PrimitiveCell` class.
* The output unit of AC conductivity from TBPM has been changed from e^2/(4*h_bar) to e^2/h_bar,
  for consistency with the :class:`.Lindhard` class.

Bugs fixed
^^^^^^^^^^

* :func:`merge_prim_cell` does not set the ``extend`` attribute properly.
* ``reset_array`` method of :class:`.Sample` class does not reset the ``rescale`` attribute.
* The FORTRAN subroutine ``norm`` produces L^1 norm instead of L^2 for complex vectors.
* The FORTRAN subroutine ``tbpm_ldos`` does not set initial state properly.

Misc.
^^^^^

* Updated documentation, examples and configuration files.
* Added more examples.

v1.0 | 2022-02-18
-----------------

First public release of TBPLaS.

New features
^^^^^^^^^^^^

* The ``builder`` module is rewritten from scratch. Now it is much easier to use and
  orders of magnitudes faster.
* The workflow of setting up a sample is simplified, with many handy tools provided.
* Added options to specify the timestep and thresthold for checking wavefunction norm
  during tbpm calculation.

Changes
^^^^^^^

* Refactored existing code into :class:`.Solver`, :class:`.Analyzer` and :class:`.Visualizer`
  classes.
* Simplified :class:`.Config`. Now it is not dependent on the Sample.
* Rewritten ``materials`` module with the new builder.
* Converted output from txt files to numpy format. Add ``-DDEBUG`` to f90flags if you don't
  like this feature.
* Many bug fixes, efficiency improvments and security enhancements.

Bugs fixed
^^^^^^^^^^

* csr.F90:
  
  The subtle bug that ``amxpby_d`` and ``amxpby_z`` do not behave as expected has been fixed.
  This bug is effective when using built-in sparse matrix library, and causes ``Fermi``
  subroutine to yield diverging results, which affects many calculations, e.g. AC conductivity.

* funcs.F90:

  Removed SIMD instructions that will cause ``ifort`` to abort during compilation.

* tbpm.f90:
  
  Fixed incorrect initial norm when checking wave function.

v0.9.8 | 2021-06-06
-------------------

New features
^^^^^^^^^^^^

* Most of the subroutines involving wave function propagation will check the
  norm of wave function after 128 steps of propagation. The program will abort
  and a error message is casted to prompt the user to increase ``rescale`` if
  NaN, Inf or large derivation from 1 of the norm is detected.

* MPI parallelization has been implemented for ``corr_DOS``, ``corr_LDOS``,
  ``corr_AC``, ``corr_dyn_pol``, ``corr_DC``, ``mu_Hall``, ``quasi_eigenstates``, 
  which may boost the calculation by approximately 20%. A new module ``parallel``
  has been introduced for this purpose, as well as necessary adjustments in modules
  of ``config``, ``correlation``, ``f2py.pyf`` and ``tbpm.f90``. Hybrid MPI+OpenMP
  parallelization is also possible by setting ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``
  properly.

* A new module ``utils`` has been introduced, which provides classes for times
  profiling, progress reporting, random number seeds generating, message
  printing, etc.

Changes
^^^^^^^

* setup.cfg:

  * Optimization flags for ifort has changed to ``-xHost``. Tests should be taken
    to avoid aggressive and unstable optimizaitons.
  * Compiler name of ``gnu95`` has been changed to ``gfortran``.

* config:

  The logic workflow has been unified and simplified. A new key ``prefix`` has
  replaced the old key ``timestamp``. Default argument values for ``set_output``
  and ``save`` methos have also been changed in according to the new workflow.

* tbpm.f90

  Some temporary arrays in subroutines ``tbpm_dccond`` and ``tbpm_eigenstates``
  have been changed from row-major to column-major, which may boosts the
  calculation by approximately 12%.

Bugs fixed
^^^^^^^^^^

* analysis.f90:

  Error of ``index out range`` has been fixed for function ``analyze_corr_DC``,
  which is due to the incomplete update of the length of ``corr_DOS``. Maybe in
  the future we may find a more elegant solution to this problem.

* propagation.f90:

  Subroutine ``cheb_wf_timestep_inv`` had not worked properly as due to a typo
  in the starting range of loop over Bessel coeffcients. Now it has been fixed
  and shares the same subroutine as cheb_wf_timestep. An argument ``fwd`` has
  been introduced to distinguish forward and backward propagation.

* random.f90:
  
  Subroutine ``random_state`` had not been thread-safe, which would lead to
  different results with different number of OpenMP threads, especially for
  AC and DC conductivity. Now the OpenMP instructions have been removed and
  the subroutine is made serial, thus being thread-safe.
