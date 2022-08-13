Release Notes
=============

v1.1 | 2022-08-13
-----------------

.. rubric:: New features

* New :class:`.Lindhard` class for evaluating response properties using Lindhard function.
* Implemented LDOS calculation based exact diagonalization.
* Implemented propagation of wave function from initial condition.
* Implemented evaluation of diffusion coeffcients from DC correlation function.
* Added MPI support for band structure and DOS calculation.
* Added support for 64-bit array indices (samples can be much larger).

.. rubric:: Improvments

* A lot of classes have been refactored for simplicity, maintainability and efficiency.
* The default values of common parameters and the units of outputs have been unified for exact
  diagonalization, Lindhard and TBPM subroutines.
* References to papers discussing the methodologies have been revised.
* Function ``merge_prim_cell`` checks lattice vectors before merging cells.
* ``plot`` method of :class:`.Sample` accepts lists of colors for plotting the supercells and
  inter-cell hopping terms.
* DC conductivity subroutine is refactored and much faster.

.. rubric:: Changes

* The ``IntraHopping`` class has beem removed. Modifications to hopping terms are now handled
  by the supercell itself.
* The ``InterHopping`` class has been renamed to :class:`.SCInterHopping`.
* The ``InterHopDict`` class has been renamed to :class:`.PCInterHopping`.
* ``apply_pbc`` and ``trim_prim_cell`` functions are moved to :class:`.PrimitiveCell` class.
* The output unit of AC conductivity from TBPM has been changed from e^2/(4*h_bar) to e^2/h_bar,
  for consistency with the :class:`.Lindhard` class.

.. rubic:: Bugs fixed

* :function:`merge_prim_cell` does not set the ``extend`` attribute properly.
* ``reset_array`` method of :class:`.Sample` class does not reset the ``rescale`` attribute.
* The FORTRAN subroutine ``norm`` produces L^1norm instead of L^2 for complex vectors.
* The FORTRAN subroutine ``tbpm_ldos`` does not set initial state properly.

v1.0 | 2022-02-18
-----------------

First public release of TBPLaS.

.. rubric:: New features

* The ``builder`` module is rewritten from scratch. Now it is much easier to use and
  orders of magnitudes faster.
* The workflow of setting up a sample is simplified, with many handy tools provided.
* Added options to specify the timestep and thresthold for checking wavefunction norm
  during tbpm calculation.

.. rubric:: Changes

* Refactored existing code into :class:`.Solver`, :class:`.Analyzer` and :class:`.Visualizer`
  classes.
* Simplified :class:`.Config`. Now it is not dependent on the Sample.
* Rewritten ``materials`` module with the new builder.
* Converted output from txt files to numpy format. Add ``-DDEBUG`` to f90flags if you don't
  like this feature.
* Many bug fixes, efficiency improvments and security enhancements.

.. rubric:: Bugs fixed

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

.. rubric:: New features

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

.. rubric:: Changes

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

.. rubric:: Bugs fixed

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
