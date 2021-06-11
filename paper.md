---
title: '``pytreegrav``: A fast Python gravity solver'
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
    orcid: 0000-0002-6145-3674
    affiliation: 1
affiliations:
 - name: Department of Physics & Astronomy and CIERA, Northwestern University, 1800 Sherman Ave, Evanston, IL 60201, USA
   index: 1
date: 9 June 2021
bibliography: paper.bib
---

# Summary

Gravity is important in a wide variety of science problems. Astrophysics problems in particular nearly all involve gravity, and can have large ($>>10^4$) numbers of gravitating masses, such as the stars in a cluster or galaxy, or the discrete fluid mass elements in a hydrodynamics simulation. Often the gravitational field of such a large number of masses can be too expensive to compute naïvely (i.e. by directly summing the contribution of every single element).

``pytreegrav`` is a multi-method Python package for computing gravitational fields and potentials. It includes an optimized brute-force solver, and a fast, approximate tree-based method that can be orders of magnitude faster than the naïve method. It supports the computation of fields and potentials from arbitrary particle distributions at arbitrary points, with arbitrary softening/smoothing lengths, and is parallelized with OpenMP.

# Statement of need

The problem addressed by ``pytreegrav`` is the following: given an arbitrary set of "source" masses $m_i$ distributed at 3D coordinates $\mathbf{x}_i$, optionally each having a finite spatial extent $h_i$ (the _softening radius_), compute either the gravitational potential $\Phi$ and/or the gravitational field $\mathbf{g}$ at an arbitrary set of "target" points in space $\mathbf{y}_i$. This task must be performed in all N-body simulations (wherein $\mathbf{y}_i=\mathbf{x}_i$). It is also often useful for _analyzing_ simulation results in post-processing -- $\Phi$ and $\mathbf{g}$ are often not saved in simulation snapshots, and even when they are it is often useful to analyze the gravitational interactions between specific _subsets_ of the mass elements in the simulations. Computing $\mathbf{g}$ is also important for generating equilibrium _initial conditions_ for N-body simulations [@makedisk;@galic], and for identifying interesting gravitationally-bound structures such as halos, star clusters, and giant molecular clouds [@rockstar;@grudic2018;@guszejnov2020].

Many gravity simulation codes (or multi-physics simulation codes _including_ gravity) have been written that address the problem of gravity computation in a variety of ways for their own internal purposes [@aarseth_nbody;@dehnen]. However, ``pykdgrav`` was the first Python package to offer a generic, modular, trivially-installable gravity solver that could be easily integrated into any other Python code, using the fast, approximate tree-based @barneshut method to be practical for large particle numbers. ``pykdgrav`` used a KD-tree implementation accelerated with ``numba`` [@numba] to achieve high performance in the potential/field evaluation, however the prerequisite tree-building step had relatively high overhead and a very large memory footprint, because the entire dataset was redundantly stored at every level in the tree hierarchy. This made it difficult to scale to various practical research problems, such as analyzing high-resolution galaxy simulations [@fire_pressurebalance]. ``pytreegrav`` is a full refactor of ``pykdgrav`` that addresses these shortcomings with a new octree implementation, with drastically reduced tree-build time and memory footprint, and a more efficient non-recursive tree traversal for field summation. This makes it suitable for post-processing datasets from state-of-the-art astrophysics simulations, with upwards of $10^8$ particles in the region of interest. 

# Methods

``pytreegrav`` can compute $\Phi$ and $\mathbf{g}$ using one of two methods: by "brute force" (explcitly summing the field of every particle, which is exact to machine precision), or using the fast, approximate @barneshut tree-based method (which is approximate, but much faster for large particle numbers). In an $N$-body problem where the fields at all particle positions must be known, the cost of the brute-force method scales $\propto N^2$, while the cost of the tree-based method scales less steeply, $\propto N \log N$ (\autoref{fig:cputime}).

![Wall-clock time per particle running ``pytreegrav`` on a sample of $N$ particles from a @plummer distribution for various $N$. Test was run on a an Intel i9 9900K workstation on a single core (_left_) and in parallel on 16 logical cores (_right_).\label{fig:cputime}](CPU_Time_both.png){ width=100% }

The brute-force methods are often fastest for small ($<10^3$ particle) point sets because they lack the overheads of tree construction and traversal, while the tree-based methods will typically be faster for larger datasets. Both methods are optimized with ``numba``, and the basic ``Accel`` and ``Potential`` front-end functions will use whichever method is likely to be faster, based on this heuristic crossover point of $10^3$ particles. Both methods can also optionally be parallelized with OpenMP, via the ``numba`` interface.

The implementation of the tree build and tree-based field summation largely follows that of ``GADGET-2`` [@gadget2]. Starting with an initial cube enclosing all particles, particles are inserted into the tree one at a time, creating sub-nodes by dividing nodes into 8 octants until each subnode contains at most one particle. The indices of the 8 subnodes of each node are stored for an initial recursive traversal of the completed tree, but the tree traversal only needs to know the _first_ subnode (if the node is to be refined) and the index of the next branch of the tree (if the field due to the node is summed directly), so these indices are recorded in the initial recursive tree traversal, and the 8 explicit subnode indices are then deleted, saving memory and removing any empty nodes from consideration. Once these "next branch" and "first subnode" indices are known, the tree field summations can be done in a single ``while`` loop with no recursive function calls, which also saves memory and generally increases performance.

The field summation itself uses the @barneshut geometric opening criterion, with improvements suggested by @dubinski: for a node of side length $L$ with centre of mass located at distance $r$ from the target point, its contribution is summed using the monopole approximation (treating the whole node as a point mass) only if $r > L/\Theta + \delta$, where $\Theta=0.7$ by default (giving $\sim 1\%$ RMS error in $\mathbf{g}$), $\delta$ is the distance from the node's geometric center to its center of mass. If the conditions for approximation are not satisfied, the node's subnodes are considered in turn, until the field contribution of all mass within the node is summed.

``pytreegrav`` supports gravitational softening by assuming the mass distribution of each particle takes the form of a standard M4 cubic spline kernel, which is $=0$ outside of the softening radius $h$ (and hence the field reduces to that of a point mass). Explicit expressions for this form of the softened gravitational potential and field are given in @gizmo. $h$ can vary from particle to particle, and when summing the field the larger of the source or the target softening is used (symmetrizing the force between overlapping particles). When softenings are nonzero, the largest softening $h_{\rm max}$ of all particles in a node is stored, and a node is always opened in the field summation if $r < 0.6L + \max\left(h_{\rm target}, h_{\rm max}\right) + \delta$, where $h_{\rm target}$ is the softening of the target particle where the field is being summed. This ensures that any interactions between physically-overlapping particles are summed directly with the softening kernel.

# Acknowledgements

We acknowledge code contributions from Martin Beroiz and helpful feedback from Elisa Bortolas, Thorsten García, and GitHub user ``herkesg`` during the development of ``pykdgrav``, which were incorporated into ``pytreegrav``.

# References