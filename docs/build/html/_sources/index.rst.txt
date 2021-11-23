.. pytreegrav documentation master file, created by
   sphinx-quickstart on Mon Nov 22 10:52:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pytreegrav's documentation!
======================================
pytreegrav is a package for computing the gravitational potential and/or field of a set of particles. It includes methods for brute-force direction summation and for the fast, approximate Barnes-Hut treecode method. For the Barnes-Hut method we implement an oct-tree as a numba jitclass to achieve much higher peformance than the equivalent pure Python implementation, without writing a single line of C or Cython.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/installation
   usage/quickstart
   Nbody_simulation
   frontend_API
   community

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
