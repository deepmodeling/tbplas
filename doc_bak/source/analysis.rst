========
Analysis
========

We can get get the correlation functions directly from \textsc{fortran},
and perform the subsequent analysis with::

    # DOS correlation, fortran call
    corr_DOS = tipsi.corr_DOS(sample, config)

    # DOS correlation analysis
    energies_DOS, DOS = tipsi.analyze_corr_DOS(config, corr_DOS)

    # AC conductivity correlation, fortran call
    corr_AC = tipsi.corr_AC(sample, config)

    # AC conductivity correlation analysis
    omegas_AC, AC = tipsi.analyze_corr_AC(config, corr_AC)

Alternatively, we can read the correlation functions from file in a 
separate Python script::

    timestamp = "1522172330" # set to output timestamp

    # read Config object
    config = tipsi.read_config("sim_data/" + timestamp + "config.pkl")

    # get DOS
    corr_DOS = tipsi.read_corr_DOS("sim_data/" + timestamp + "corr_DOS.dat")
    energies_DOS, DOS = tipsi.analyze_corr_DOS(config, corr_DOS)

Correlation function analysis
-----------------------------

.. autofunction:: tipsi.analysis.analyze_corr_DOS

Relevant config parameters::

    config.generic['nr_time_steps']
    config.generic['correct_spin']

.. autofunction:: tipsi.analysis.analyze_corr_LDOS

Relevant config parameters::

    config.generic['nr_time_steps']
    config.generic['correct_spin']
    
.. autofunction:: tipsi.analysis.analyze_corr_AC

Relevant config parameters::

    config.generic['nr_time_steps']
    config.generic['correct_spin']
    
.. autofunction:: tipsi.analysis.AC_imag
.. autofunction:: tipsi.analysis.analyze_corr_DC

Relevant config parameters::

    config.generic['nr_time_steps']
    config.generic['correct_spin']
    config.DC_conductivity['energy_limits']
    
.. autofunction:: tipsi.analysis.analyze_corr_dyn_pol

Relevant config parameters::

    config.generic['nr_time_steps']
    config.generic['correct_spin']
    config.dyn_pol['q_points']
    
.. autofunction:: tipsi.analysis.get_dielectric_function

Relevant config parameters::

    config.generic['nr_time_steps']
    config.dyn_pol['coulomb_constant']
    config.dyn_pol['q_points']

Reading from file
-----------------

.. autofunction:: tipsi.input.read_config
.. autofunction:: tipsi.input.read_sample
.. autofunction:: tipsi.input.read_corr_DOS
.. autofunction:: tipsi.input.read_corr_LDOS
.. autofunction:: tipsi.input.read_corr_AC
.. autofunction:: tipsi.input.read_corr_DC
.. autofunction:: tipsi.input.read_corr_dyn_pol

Window functions
----------------

.. autofunction:: tipsi.analysis.window_Hanning
.. autofunction:: tipsi.analysis.window_exp
.. autofunction:: tipsi.analysis.window_exp_ten

Output
-----------------------------

.. autofunction:: tipsi.output.plot_wf
