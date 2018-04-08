Running
=================================

First run
#################################

When installed correctly tipsi can be included in all python scripts and programs with a simple::

    import tipsi

This will load all tipsi modules.

Making a sample(wrong now, need to modify)
********************************

The first step in to create a sample with a particular lattice. Simple examples are already included, such as a honeycomb lattice::

    sys = tipsi.honeycomb_sheet(512,512)

Which will create a honeycomb lattice with 512 by 512 unitcells (giving 524288 atoms in our system).
Here you can add potential disorder if you want.
Now we finalize the sites so we can add hoppings, using nearest neighbor hopping::

    sys.finalize_sites()
    hopping_value = 1.
    sys.neighbor_hopping(-hopping_value)

Here you can add hopping disorder if you want. Also, you can plot the system with *sys.plot()*
Now we finalize the hoppings::

    sys = sys.finalize()

Simulation Parameters
********************************

The next step is to set your parameters needed for the simulation. The values are initialized by::

    tbpm = tipsi.tbpm_config(sys)

Default values will be set. To change the defaults one uses (for example)::

    # number of random samples
    tbpm.generic['rannr'] = 8
    # rescaling of Hamiltonian so that abs(highest eigenvalue) <= 1
    tbpm.generic['hoprescale'] = 4.
    # energy range
    tbpm.generic['energyrange'] = 12.

Starting simulation
********************************

Now we're ready to start the calculations. To calculate the DOS one types::

    energies, dos = tipsi.get_dos(sys,tbpm)
