---
title: 'pytreegrav: a python package for fast tree-based gravity and spatial algorithms'
tags:
  - Python
  - physics
  - gravity
  - simulations
authors:
  - name: Michael Y. Grudić
    orcid: 0000-0002-1655-5604
    affiliation: 1
  - name: Alexander B. Gurvich
    affiliation: 1
affiliations:
 - name: Department of Physics & Astronomy and CIERA, Northwestern University
   index: 1
date: 9 June 2021
bibliography: paper.bib
---

# Summary

Gravity is important in a wide variety of science problems. In particular, astrophysics problems nearly all involve gravity and can have large ($>>10^4$) numbers of gravitating masses, such as the stars in a cluster or galaxy, or the discrete fluid mass elements in a hydrodynamics simulation. Often the gravitational field of such a large number of masses can be too expensive to compute naïvely (i.e. by directly summing the contribution of every single element).
``pytreegrav`` is a Python package for computing gravitational fields and potentials using a fast, approximate tree-based method that can be orders of magnitude faster than the naïve method. It supports the computation of fields and potentials from arbitrary particle distributions at arbitrary points, with arbitrary softening/smoothing lengths, and is parallelized with OpenMP. It can also leverage its tree structure to perform common statistical computations, including spatial correlation functions and velocity structure functions.

# Statement of need

The problem addressed by ``pytreegrav`` is the following: given an arbitrary set of "source" masses $m_i$ distributed at 3D coordinates $\mathbf{x}_i$, optionally each having a finite spatial extent $h_i$ (the _softening radius_), compute either the gravitational potential $\Phi$ and/or the gravitational field $\mathbf{g}$ at an arbitrary set of "target" points in space $\mathbf{y}_i$. This task must be performed in all N-body simulations (wherein $\mathbf{y}_i=\mathbf{x}_i$). It is also often useful for _analyzing_ simulation results in post-processing -- $\Phi$ and $\mathbf{g}$ are often not saved in simulation snapshots, and even when they are it is often useful to analyze the gravitational interactions between specific _subsets_ of the mass elements in the simulations. Computing the potential is also important for generating equilibrium _initial conditions_ for N-body simulations [@galic], and for identifying interesting gravitationally-bound structures such as halos, star clusters, and giant molecular clouds.

Many gravity simulation codes (or multi-physics simulation codes _including_ gravity) have been written that address the problem of gravity computation in a variety of ways for their own internal purposes [@aarseth_nbody;@dehnen]. However, ``pykdgrav`` was the first Python package to offer a generic, modular, trivially-installable gravity solver that could be easily integrated into any other Python code, using the fast, approximate tree-based method pioneered by [@barneshut] to be practical for large particle numbers. ``pykdgrav`` used a KD-tree implementation accelerated with ``numba`` to achieve high performance in the potential/field evaluation, however the prerequisite tree-building step had relatively high overhead and a very large memory footprint, because the entire dataset was redundantly stored at every level in the tree hierarchy. This made it difficult to scale to various practical research problems, such as analyzing high-resolution galaxy simulations [@fire2,@fire_pressurebalance]. ``pytreegrav`` is a full refactor of ``pykdgrav`` that addresses these shortcomings, with drastically reduced tree-build time and memory footprint, and a more efficient non-recursive tree traversal for field summation.

# Methods

``pytreegrav`` can compute $\Phi$ and $\mathbf{g}$ using one of two methods: by "brute force" (explcitly summing the field of every particle, which is exact to machine precision), or using the tree (which is approximate, but much faster for large particle numbers). The brute-force methods are 