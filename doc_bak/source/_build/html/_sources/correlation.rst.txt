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
- LDOS
- quasi-eigenstates
- DC conductivity
- AC conductivity
- dynamical polarization
- dielectric function

DOS
----

.. autofunction:: tipsi.correlation.corr_DOS

Relevant config parameters::

    config.generic['Bessel_precision']
    config.generic['Bessel_max']
    config.generic['seed']
    config.generic['nr_time_steps']
    config.generic['nr_random_samples']
    config.output['corr_DOS']

LDOS
----

.. autofunction:: tipsi.correlation.corr_LDOS

Relevant config parameters::
    
    config.generic['Bessel_precision']
    config.generic['Bessel_max']
    config.generic['seed']
    config.generic['nr_time_steps']
    config.generic['nr_random_samples']
    config.LDOS['site_indices']
    config.LDOS['wf_weights']
    config.output['corr_LDOS']
    
AC conductivity
---------------

.. autofunction:: tipsi.correlation.corr_AC

Relevant config parameters::

    config.generic['Bessel_precision']
    config.generic['Bessel_max']
    config.generic['beta']
    config.generic['mu']
    config.generic['seed']
    config.generic['nr_time_steps']
    config.generic['nr_random_samples']
    config.generic['nr_Fermi_fft_steps']
    config.generic['Fermi_cheb_precision']
    config.output['corr_AC']
        
DC conductivity
---------------

.. autofunction:: tipsi.correlation.corr_DC

Relevant config parameters::

    config.generic['Bessel_precision']
    config.generic['Bessel_max']
    config.generic['beta']
    config.generic['mu']
    config.generic['seed']
    config.generic['nr_time_steps']
    config.generic['nr_random_samples']
    config.DC_conductivity['energy_limits']
    config.output['corr_DOS']
    config.output['corr_DC']

Dynamical polarization
----------------------

.. autofunction:: tipsi.correlation.corr_dyn_pol
    
Relevant config parameters::

    config.generic['Bessel_precision']
    config.generic['Bessel_max']
    config.generic['beta']
    config.generic['mu']
    config.generic['seed']
    config.generic['nr_time_steps']
    config.generic['nr_random_samples']
    config.generic['nr_Fermi_fft_steps']
    config.generic['Fermi_cheb_precision']
    config.dyn_pol['q_points']
    config.output['corr_dyn_pol']

Quasi-eigenstates
-----------------

.. autofunction:: tipsi.correlation.quasi_eigenstates
    
This function does not write correlation functions to the ``sim_data`` folder, but only
returns wave functions to python. Relevant config parameters::

    config.generic['Bessel_precision']
    config.generic['Bessel_max']
    config.generic['seed']
    config.generic['nr_time_steps']
    config.generic['nr_random_samples']
    config.quasi_eigenstates['energies']
