Welcome to TiPSi's documentation!
=================================

TiPSi is a package for Python 3 to make large-scale tight-binding Hamiltonians and 
run Tight Binding Propagation Method (TBPM) calculations. TiPSi is optimized for
usage on cluster nodes. It uses FORTRAN code to do the number crunching, and 
f2py to interface with Python.

In general, a simulation consists of the following steps:

- Make a tight-binding Hamiltonian
- Calculate correlation functions
- Analyze correlation functions

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gettingstarted
   firstrun
   builder
   materials
   parameters
   correlation
   analysis
   examples

Full documentation
==================

* :ref:`genindex`
