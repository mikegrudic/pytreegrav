---
title: 'pytreegrav: a python package for fast tree-based gravity and spatial algorithms'
tags:
  - Python
  - physics
  - gravity
  - simulations
authors:
  - name: Michael Y. Grudić^[first author] 
    orcid: 0000-0002-1655-5604
    affiliation: 1
  - name: Alexander B. Gurvich^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 1
affiliations:
 - name: Department of Physics & Astronomy and CIERA, Northwestern University
   index: 1
date: 9 June 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Gravity is important in a wide variety of science problems. In particular, astrophysics problems nearly all involve gravity and can have large ($>>10^4$) numbers of gravitating masses, such as the stars in a cluster or galaxy, or the discrete fluid mass elements in a hydrodynamics simulation. The gravitational field of a large number of masses can be too expensive to computel naïvely (i.e. by directly summing the contribution of every single element). ``pytreegrav`` is a Python package for computing gravitational fields and potentials using a fast, approximate tree-based method that can be orders of magnitude faster than the naïve method. It supports the computation of fields and potentials from arbitrary particle distributions at arbitrary points, with arbitrary softening/smoothing lengths, and is parallelized with OpenMP.

# Statement of need

The problem addressed by ``pytreegrav`` is the following: given an arbitrary set of "source" masses $m_i$ distributed at 3D coordinates $\mathbf{x}_i$, optionally each having a finite spatial extent $h_i$ (the _softening radius_), compute either the gravitational potential $\Phi$ or the gravitational field $\mathbf{g}$ at an arbitrary set of points in space $\mathbf{y}_i$. Many gravity simulation codes (or multi-physics simulation codes _including_ gravity) have been written that address the problem of gravity computation in a variety of ways for their own internal purposes [@aarseth_nbody,@2011EPJP..126...55D]. However, ``pykdgrav`` was the first Python package offering a generic, modular, trivially-installable gravity solver that could be easily integrated into any other Python code, that could use the fast, approximate tree-based method pioneered by [@barneshut] to scale up to be practical for large particle numbers.

## It takes advantage of JIT compilation by numba to achieve high speeds comparable to state-of-the-art simulation codes, and is capable of OpenMP parallelism for the field evaluation. 
