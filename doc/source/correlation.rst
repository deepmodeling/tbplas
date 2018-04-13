=================================
Calculating correlation functions
=================================

Now that we have created a Sample, and defined the simulation configuration
parameters, we are ready to perform an actual calculation by calling the 
fortran subroutines. The resulting correlation functions are
automatically written to files in the ``sim_data`` folder. The
correlation functions are also returned to Python::

    # DOS correlation, fortran call
    corr_DOS = tipsi.corr_DOS(sample, config)

    # AC conductivity correlation, fortran call
    corr_AC = tipsi.corr_AC(sample, config)

In the current version, the following physical quantities can
be calculated:

- DOS
- quasi-eigenstates
- DC conductivity
- AC conductivity
- dynamical polarization
- dielectric function

DOS
----

.. autofunction:: tipsi.correlation.corr_DOS()

AC conductivity
---------------

.. autofunction:: tipsi.correlation.corr_AC()

DC conductivity
---------------

.. autofunction:: tipsi.correlation.corr_DC()

Dynamical polarization
----------------------

.. autofunction:: tipsi.correlation.corr_dyn_pol()

Quasi-eigenstates
-----------------

.. autofunction:: tipsi.correlation.quasi_eigenstates()