Simulation parameters
=================================

Before we can run a simulation we need to set the parameters for the TBPM
calculations. For example, if we want to use 1024 time steps, 4 random
samples, an energy range from -10 to 10 eV and we want to correct for spin
in the final result::

    config = tipsi.Config(sample)
    config.generic['nr_time_steps'] = 1024
    config.generic['nr_random_samples'] = 4
    config.generic['energy_range'] = 20.
    config.generic['correct_spin'] = True
    config.save()
    
The last line ensures that the configuration object is saved to file
in the ``sim_data`` folder, with the same timestamp prefix as the correlation files.

Each correlation function calculation has its own set of configuration
parameters. Moreover, the Config object also contains output options.

.. autoclass:: tipsi.config.Config
   :members:
