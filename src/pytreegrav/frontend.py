import numpy as np
import warnings
from numpy import zeros_like, zeros
from .kernel import *
from .octree import *
from .dynamic_tree import *
from .treewalk import *
from .grouped_treewalk import AccelTarget_grouped, PotentialTarget_grouped, _morton_order
from .bruteforce import *
from .bruteforce_symmetric import Potential_bruteforce_symmetric, Accel_bruteforce_symmetric, SYMMETRIC_NMIN
from .misc import *


def _f64(a):
    """Cast to float64 without collapsing a length-1 array to a scalar.

    ``np.float64(a)`` looks like a dtype cast, but on any array of size 1 numpy applies its *scalar constructor* instead and returns a 0-d float64 (deprecated since numpy 1.25, fixed only in 2.4). A caller passing a single particle therefore had ``m`` and ``softening`` silently turned into scalars, and the jitclass then failed to type ``masses[oi]`` (github issue #31). ``asarray`` never does this.
    """
    return np.asarray(a, dtype=np.float64)


def checkTreeQuadrupoles(tree, quadrupole):
    """Raise ValueError if a quadrupole treewalk is requested on a tree that lacks them.

    The walk kernels index tree.Quadrupoles based only on the walk's quadrupole flag, but that array is allocated only when the tree itself was built with quadrupole=True. The mismatch is an out-of-bounds read, which numba does not bounds-check, so without this guard it segfaults instead of raising.
    """
    if quadrupole and not tree.HasQuads:
        raise ValueError(
            "quadrupole=True requires a tree built with quadrupole moments: pass a tree from "
            "ConstructTree(..., quadrupole=True), or evaluate with quadrupole=False."
        )


def valueTestMethod(method):
    """Raise TypeError/ValueError unless method is one of 'adaptive', 'bruteforce', 'tree'."""
    methods = ["adaptive", "bruteforce", "tree"]

    ## check if method is a str
    if type(method) != str:
        raise TypeError("Invalid method type %s, must be str" % type(method))

    ## check if method is a valid method
    if method not in methods:
        raise ValueError("Invalid method %s. Must be one of: %s" % (method, str(methods)))


def warn_if_coincident_positions(tree, softening=None):
    """Warn if the build found particle positions coincident to floating-point precision.

    Asks the tree rather than scanning the input: the build cannot subdivide coincident points forever, so it already detects them exactly, and reading its flag is free. The np.unique-per-dimension scan this replaces cost over half the build time at N=1e7 and was wrong besides, testing whether any single *coordinate* repeated rather than any *position* -- so it warned about any lattice or grid whose positions were in fact all distinct.
    """
    if not tree.HasCoincidentPoints:
        return

    if softening is not None and np.any(softening > 0):
        warnings.warn(
            "Warning: Particle positions are non-unique. Softening will determine the answer for overlapping particles."
        )
        return

    warnings.warn(
        "Warning: Particle positions are non-unique. The answer will be singular or garbage for overlapping particles."
    )
    return


def ConstructTree(
    pos,
    m=None,
    softening=None,
    quadrupole=False,
    vel=None,
    compute_moments=True,
    morton_order=True,
    radix=True,
):
    """Builds a tree containing particle data, for subsequent potential/field evaluation

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like or None, optional
        shape (N,) array of particle masses - if None then zeros will be used (e.g. if all you need the tree for is spatial algorithms)
    softening: array_like or None, optional
        shape (N,) array of particle softening lengths - these give the radius of compact support of the M4 cubic spline mass distribution of each particle
    quadrupole: bool, optional
        Whether to store quadrupole moments (default False)
    vel: array_like or None, optional
        shape (N,3) array of particle velocities. If provided, a DynamicOctree that also stores node-averaged velocities is built (used by the velocity correlation/structure functions); if None, a static Octree is built (default None)
    compute_moments: bool, optional
        Whether to compute node multipole moments (centers of mass, masses, softenings, and quadrupoles if enabled). Set False to build only the spatial structure, e.g. for purely geometric queries (default True; forced False when m is None)
    morton_order: bool, optional
        Whether to store particles in Morton (depth-first traversal) order for cache-efficient treewalks (default True)
    radix: bool, optional
        Whether to use the radix-sort tree build (faster, default True). Set False to use the legacy insertion build. Ignored when vel is provided (dynamic tree always uses insertion), or when morton_order is False.

    Returns
    -------
    tree: Octree or DynamicOctree
        tree instance built from the particle data
    """

    # Coerce here rather than trusting the caller: the jitclass reports a shape/dtype mismatch as a
    # numba TypingError deep in the build, which is unreadable. atleast_1d/2d in particular rescue a
    # single particle, whose arrays a caller can easily have had collapsed to scalars -- see _f64.
    pos = np.atleast_2d(_f64(pos))
    if m is None:
        m = zeros(len(pos))
        compute_moments = False
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = zeros_like(m)
    softening = np.atleast_1d(_f64(softening))
    if vel is not None:
        vel = np.atleast_2d(_f64(vel))
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(m)) and np.all(np.isfinite(softening))):
        print("Invalid input detected - aborting treebuild to avoid going into an infinite loop!")
        raise

    if vel is None:
        tree = Octree(
            pos,
            m,
            softening,
            quadrupole=quadrupole,
            compute_moments=compute_moments,
            morton_order=morton_order,
            radix=radix,
        )
    else:
        tree = DynamicOctree(pos, m, softening, vel, quadrupole=quadrupole)

    # the build detects coincident positions as a side effect, so this costs nothing
    warn_if_coincident_positions(tree, softening)
    return tree


def Potential(
    pos,
    m,
    softening=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
    device="cpu",
):
    """Gravitational potential calculation

    Returns the gravitational potential for a set of particles with positions x and masses m, at the positions of those particles, using either brute force or tree-based methods depending on the number of particles.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    G: float, optional
        gravitational constant (default 1.0)
    softening: None or array_like, optional
        shape (N,) array containing kernel support radii for gravitational softening - these give the radius of compact support of the M4 cubic spline mass distribution - set to 0 by default
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    device: str, optional
        'cpu' (default) or 'cuda'. 'cuda' needs pytreegrav[cuda] and an NVIDIA GPU, and covers the monopole tree and brute-force methods. It is float32, but its error against the CPU path stays below theta's own truncation error. Uploads the tree (or sources) on every call; for repeated evaluation hold a pytreegrav.cuda.CudaPotential/CudaAccel or their Bruteforce counterparts instead.

    Returns
    -------
    phi: array_like
        shape (N,) array of potentials at the particle positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    # Coerce once, up front, so every method sees the same thing. Only the tree path used to coerce
    # (via ConstructTree), so lists worked with method="tree" and raised a numba TypingError with
    # method="bruteforce"; asarray is a no-op for arrays that are already float64.
    pos = np.atleast_2d(_f64(pos))
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = np.zeros_like(m)
    softening = np.atleast_1d(_f64(softening))

    # figure out which method to use
    if method == "adaptive":
        # A supplied tree pins the method: brute force cannot use it and would silently sum only the
        # particles in pos, a different quantity whenever the tree holds a different set. Otherwise,
        # threaded brute force stays competitive to larger N -- 4000 is where the parallel curves cross
        # on 32 threads (Plummer, theta=0.7): at N=3447 brute-force accel is 0.57 us/particle against
        # the tree's 0.67, by N=6092 it is 0.95 against 0.68. Potential crosses nearer 5500, so this
        # favours acceleration, the more common call and the steeper curve to be wrong about.
        if tree is not None or len(pos) > (4000 if parallel else 1000):
            method = "tree"
        else:
            method = "bruteforce"

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and quadrupole:
        raise ValueError("device='cuda' is monopole only; pass quadrupole=False")

    if method == "bruteforce":  # we're using brute force
        if device == "cuda":
            from .cuda import CudaPotentialBruteforce  # lazy: numba-cuda is an optional extra

            phi = CudaPotentialBruteforce(pos, m, softening)(pos, softening, G=G)
        elif parallel:
            # the symmetrized kernel runs two parallel regions and per-thread buffers;
            # that only pays for itself once there are enough interactions to amortize it
            if len(pos) >= SYMMETRIC_NMIN:
                phi = Potential_bruteforce_symmetric(pos, m, softening, G=G)
            else:
                phi = Potential_bruteforce_parallel(pos, m, softening, G=G)
        else:
            phi = Potential_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos),
                _f64(m),
                _f64(softening),
                quadrupole=quadrupole,
            )  # build the tree if needed
            idx = tree.TreewalkIndices  # built from pos, so its walk order already indexes pos
        else:
            # SORT pos; don't apply the tree's stored permutation. TreewalkIndices is a fixed sigma
            # over whatever built the tree, and is not an involution -- pos already in tree order
            # becomes X[sigma^2]: right answer, but grouping's acceptance padding inflates 94x in a
            # STARFORGE snapshot's densest decile (p90 group extent 0.115 -> 10.9 pc), 15 s -> 285 s.
            # A larger supplied tree raises IndexError outright. Sorting is idempotent, fixing both.
            idx = _morton_order(_f64(pos))
        checkTreeQuadrupoles(tree, quadrupole)

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos, idx, axis=0)
        h_sorted = np.take(softening, idx)

        if device == "cuda":
            from .cuda import CudaPotential  # lazy: numba-cuda is an optional extra

            phi = CudaPotential(tree)(pos_sorted, h_sorted, G=G, theta=theta)
        else:
            # pos_sorted is in Morton order, so consecutive targets are spatially compact groups
            phi = PotentialTarget_grouped(
                pos_sorted,
                h_sorted,
                tree,
                group_size=group_size,
                G=G,
                theta=theta,
                quadrupole=quadrupole,
                parallel=parallel,
            )

        # now reorder phi back to the order of the input positions
        phi = np.take(phi, idx.argsort())

    if return_tree:
        return phi, tree
    else:
        return phi


def PotentialTarget(
    pos_target,
    pos_source,
    m_source,
    softening_target=None,
    softening_source=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
):
    """Gravitational potential calculation for general N+M body case

    Returns the gravitational potential for a set of M particles with positions x_source and masses m_source, at the positions of a set of N particles that need not be the same.

    Parameters
    ----------
    pos_target: array_like
        shape (N,3) array of target particle positions where you want to know the potential
    pos_source: array_like
        shape (M,3) array of source particle positions (positions of particles sourcing the gravitational field)
    m_source: array_like
        shape (M,) array of source particle masses
    softening_target: array_like or None, optional
        shape (N,) array of target particle softening radii - these give the radius of compact support of the M4 cubic spline mass distribution
    softening_source: array_like or None, optional
        shape (M,) array of source particle radii  - these give the radius of compact support of the M4 cubic spline mass distribution
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    Returns
    -------
    phi: array_like
        shape (N,) array of potentials at the target positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    ## allow user to pass in tree without passing in source pos and m
    ##  but catch if they don't pass in the tree.
    if tree is None and (pos_source is None or m_source is None):
        raise ValueError("Must pass either pos_source & m_source or source tree.")

    if softening_target is None:
        softening_target = zeros(len(pos_target))
    if softening_source is None and pos_source is not None:
        softening_source = zeros(len(pos_source))

    # figure out which method to use
    if method == "adaptive":
        if pos_source is None or len(pos_target) * len(pos_source) > 10**6:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            phi = PotentialTarget_bruteforce_parallel(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        else:
            phi = PotentialTarget_bruteforce(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos_source),
                _f64(m_source),
                _f64(softening_source),
                quadrupole=quadrupole,
            )  # build the tree if needed
        checkTreeQuadrupoles(tree, quadrupole)
        # external targets are not spatially ordered; Morton-sort them so grouping is effective
        tsort = _morton_order(_f64(pos_target))
        phi_sorted = PotentialTarget_grouped(
            _f64(pos_target)[tsort],
            _f64(softening_target)[tsort],
            tree,
            group_size=group_size,
            G=G,
            theta=theta,
            quadrupole=quadrupole,
            parallel=parallel,
        )
        phi = np.empty_like(phi_sorted)
        phi[tsort] = phi_sorted  # undo the Morton permutation

    if return_tree:
        return phi, tree
    else:
        return phi


def Accel(
    pos,
    m,
    softening=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
    device="cpu",
):
    """Gravitational acceleration calculation

    Returns the gravitational acceleration for a set of particles with positions x and masses m, at the positions of those particles, using either brute force or tree-based methods depending on the number of particles.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    G: float, optional
        gravitational constant (default 1.0)
    softening: None or array_like, optional
        shape (N,) array containing kernel support radii for gravitational softening - these give the radius of compact support of the M4 cubic spline mass distribution - set to 0 by default
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    device: str, optional
        'cpu' (default) or 'cuda'. 'cuda' needs pytreegrav[cuda] and an NVIDIA GPU, and covers the monopole tree and brute-force methods. It is float32, but its error against the CPU path stays below theta's own truncation error. Uploads the tree (or sources) on every call; for repeated evaluation hold a pytreegrav.cuda.CudaPotential/CudaAccel or their Bruteforce counterparts instead.

    Returns
    -------
    g: array_like
        shape (N,3) array of acceleration vectors at the particle positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    # Coerce once, up front, so every method sees the same thing. Only the tree path used to coerce
    # (via ConstructTree), so lists worked with method="tree" and raised a numba TypingError with
    # method="bruteforce"; asarray is a no-op for arrays that are already float64.
    pos = np.atleast_2d(_f64(pos))
    m = np.atleast_1d(_f64(m))
    if softening is None:
        softening = np.zeros_like(m)
    softening = np.atleast_1d(_f64(softening))

    # figure out which method to use
    if method == "adaptive":
        # see the method-selection note in Potential
        if tree is not None or len(pos) > (4000 if parallel else 1000):
            method = "tree"
        else:
            method = "bruteforce"

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda" and quadrupole:
        raise ValueError("device='cuda' is monopole only; pass quadrupole=False")

    if method == "bruteforce":  # we're using brute force
        if device == "cuda":
            from .cuda import CudaAccelBruteforce  # lazy: numba-cuda is an optional extra

            g = CudaAccelBruteforce(pos, m, softening)(pos, softening, G=G)
        elif parallel:
            # see the note in Potential: small problems stay on the simpler kernel
            if len(pos) >= SYMMETRIC_NMIN:
                g = Accel_bruteforce_symmetric(pos, m, softening, G=G)
            else:
                g = Accel_bruteforce_parallel(pos, m, softening, G=G)
        else:
            g = Accel_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos),
                _f64(m),
                _f64(softening),
                quadrupole=quadrupole,
            )  # build the tree if needed
            idx = tree.TreewalkIndices  # built from pos, so its walk order already indexes pos
        else:
            # SORT pos; don't apply the tree's stored permutation. TreewalkIndices is a fixed sigma
            # over whatever built the tree, and is not an involution -- pos already in tree order
            # becomes X[sigma^2]: right answer, but grouping's acceptance padding inflates 94x in a
            # STARFORGE snapshot's densest decile (p90 group extent 0.115 -> 10.9 pc), 15 s -> 285 s.
            # A larger supplied tree raises IndexError outright. Sorting is idempotent, fixing both.
            idx = _morton_order(_f64(pos))
        checkTreeQuadrupoles(tree, quadrupole)

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos, idx, axis=0)
        h_sorted = np.take(softening, idx)

        if device == "cuda":
            from .cuda import CudaAccel  # lazy: numba-cuda is an optional extra

            g = CudaAccel(tree)(pos_sorted, h_sorted, G=G, theta=theta)
        else:
            # pos_sorted is in Morton order, so consecutive targets are spatially compact groups
            g = AccelTarget_grouped(
                pos_sorted,
                h_sorted,
                tree,
                group_size=group_size,
                G=G,
                theta=theta,
                quadrupole=quadrupole,
                parallel=parallel,
            )

        # now g is in the tree-order: reorder it back to the original order
        g = np.take(g, idx.argsort(), axis=0)

    if return_tree:
        return g, tree
    else:
        return g


def AccelTarget(
    pos_target,
    pos_source,
    m_source,
    softening_target=None,
    softening_source=None,
    G=1.0,
    theta=0.7,
    tree=None,
    return_tree=False,
    parallel=False,
    method="adaptive",
    quadrupole=False,
    group_size=8,
):
    """Gravitational acceleration calculation for general N+M body case

    Returns the gravitational acceleration for a set of M particles with positions x_source and masses m_source, at the positions of a set of N particles that need not be the same.

    Parameters
    ----------
    pos_target: array_like
        shape (N,3) array of target particle positions where you want to know the acceleration
    pos_source: array_like
        shape (M,3) array of source particle positions (positions of particles sourcing the gravitational field)
    m_source: array_like
        shape (M,) array of source particle masses
    softening_target: array_like or None, optional
        shape (N,) array of target particle softening radii - these give the radius of compact support of the M4 cubic spline mass distribution
    softening_source: array_like or None, optional
        shape (M,) array of source particle radii - these give the radius of compact support of the M4 cubic spline mass distribution
    G: float, optional
        gravitational constant (default 1.0)
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, giving ~0.2% RMS acceleration error on a Plummer sphere; 0.5 gives ~0.1%)
    parallel: bool, optional
        If True, will parallelize the force summation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)
    method: str, optional
        Which summation method to use: 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    quadrupole: bool, optional
        Whether to use quadrupole moments in tree summation (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the dominant traversal cost (default 8, ~2-3x faster than 1 at equal-or-better accuracy; much larger values slow down again as group bounding boxes open more nodes). 1 reproduces the per-particle walk. Only affects the tree method.

    Returns
    -------
    phi: array_like
        shape (N,3) array of accelerations at the target positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    ## allow user to pass in tree without passing in source pos and m
    ##  but catch if they don't pass in the tree.
    if tree is None and (pos_source is None or m_source is None):
        raise ValueError("Must pass either pos_source & m_source or source tree.")

    if softening_target is None:
        softening_target = zeros(len(pos_target))
    if softening_source is None and pos_source is not None:
        softening_source = zeros(len(pos_source))

    # figure out which method to use
    if method == "adaptive":
        if pos_source is None or len(pos_target) * len(pos_source) > 10**6:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            g = AccelTarget_bruteforce_parallel(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        else:
            g = AccelTarget_bruteforce(
                pos_target,
                softening_target,
                pos_source,
                m_source,
                softening_source,
                G=G,
            )
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                _f64(pos_source),
                _f64(m_source),
                _f64(softening_source),
                quadrupole=quadrupole,
            )  # build the tree if needed
        checkTreeQuadrupoles(tree, quadrupole)
        # external targets are not spatially ordered; Morton-sort them so grouping is effective
        tsort = _morton_order(_f64(pos_target))
        g_sorted = AccelTarget_grouped(
            _f64(pos_target)[tsort],
            _f64(softening_target)[tsort],
            tree,
            group_size=group_size,
            G=G,
            theta=theta,
            quadrupole=quadrupole,
            parallel=parallel,
        )
        g = np.empty_like(g_sorted)
        g[tsort] = g_sorted  # undo the Morton permutation

    if return_tree:
        return g, tree
    else:
        return g


def DensityCorrFunc(
    pos,
    m,
    rbins=None,
    max_bin_size_ratio=100,
    theta=1.0,
    tree=None,
    return_tree=False,
    parallel=False,
    boxsize=0,
    weighted_binning=False,
):
    """Computes the average amount of mass in radial bin [r,r+dr] around a point, provided a set of radial bins.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    rbins: array_like or None, optional
        1D array of radial bin edges - if None will use heuristics to determine sensible bins. Otherwise MUST BE LOGARITHMICALLY SPACED (default None)
    max_bin_size_ratio: float, optional
        controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0)
    parallel: bool, optional
        If True, will parallelize the correlation function computation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        if True will return the generated or used tree for future use (default False)
    boxsize: float, optional
        finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning: bool, optional
        (experimental) if True will distribute mass among radial bings with a weighted kernel (default False)

    Returns
    -------
    rbins: array_like
        array containing radial bin edges
    mbins: array_like
        array containing mean mass in radial bins, averaged over all points
    """

    if rbins is None:
        r = np.sort(np.sqrt(np.sum((pos - np.median(pos, axis=0)) ** 2, axis=1)))
        rbins = 10 ** np.linspace(np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3)))

    built_tree = tree is None
    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(_f64(pos), _f64(m), _f64(softening))  # build the tree if needed
    # Sort pos rather than applying the tree's stored permutation -- see the note in Potential. The
    # output is a binned aggregate, so sigma^2 was still a valid sample; the real failure was that a
    # supplied tree of a different size indexed out of bounds.
    idx = tree.TreewalkIndices if built_tree else _morton_order(_f64(pos))
    pos_sorted = np.take(pos, idx, axis=0)

    if parallel:
        mbins = DensityCorrFunc_tree_parallel(
            pos_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )
    else:
        mbins = DensityCorrFunc_tree(
            pos_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )

    if return_tree:
        return rbins, mbins, tree
    else:
        return rbins, mbins


def VelocityCorrFunc(
    pos,
    m,
    v,
    rbins=None,
    max_bin_size_ratio=100,
    theta=1.0,
    tree=None,
    return_tree=False,
    parallel=False,
    boxsize=0,
    weighted_binning=False,
):
    """Computes the weighted average product v(x).v(x+r), for a vector field v, in radial bins

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    v: array_like
        shape (N,3) of vector quantity (e.g. velocity, magnetic field, etc)
    rbins: array_like or None, optional
        1D array of radial bin edges - if None will use heuristics to determine sensible bins. Otherwise MUST BE LOGARITHMICALLY SPACED (default None)
    max_bin_size_ratio: float, optional
        controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0)
    parallel: bool, optional
        If True, will parallelize the correlation function computation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        if True will return the generated or used tree for future use (default False)
    boxsize: float, optional
        finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning: bool, optional
        (experimental) if True will distribute mass among radial bings with a weighted kernel (default False)

    Returns
    -------
    rbins: array_like
        array containing radial bin edges
    corr: array_like
        array containing correlation function values in radial bins
    """

    if rbins is None:
        r = np.sort(np.sqrt(np.sum((pos - np.median(pos, axis=0)) ** 2, axis=1)))
        rbins = 10 ** np.linspace(np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3)))

    built_tree = tree is None
    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(_f64(pos), _f64(m), _f64(softening), vel=v)  # build the tree if needed
    # Sort pos rather than applying the tree's stored permutation -- see the note in Potential. The
    # output is a binned aggregate, so sigma^2 was still a valid sample; the real failure was that a
    # supplied tree of a different size indexed out of bounds.
    idx = tree.TreewalkIndices if built_tree else _morton_order(_f64(pos))
    pos_sorted = np.take(pos, idx, axis=0)
    v_sorted = np.take(v, idx, axis=0)
    wt_sorted = np.take(m, idx, axis=0)
    if parallel:
        corr = VelocityCorrFunc_tree_parallel(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )
    else:
        corr = VelocityCorrFunc_tree(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )

    if return_tree:
        return rbins, corr, tree
    else:
        return rbins, corr


def VelocityStructFunc(
    pos,
    m,
    v,
    rbins=None,
    max_bin_size_ratio=100,
    theta=1.0,
    tree=None,
    return_tree=False,
    parallel=False,
    boxsize=0,
    weighted_binning=False,
):
    """Computes the structure function for a vector field: the average value of (v(x) - v(x+r))^2, in radial bins for r

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    v: array_like
        shape (N,3) of vector quantity (e.g. velocity, magnetic field, etc)
    rbins: array_like or None, optional
        1D array of radial bin edges - if None will use heuristics to determine sensible bins. Otherwise MUST BE LOGARITHMICALLY SPACED (default None)
    max_bin_size_ratio: float, optional
        controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta: float, optional
        cell opening angle used to control accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0)
    parallel: bool, optional
        If True, will parallelize the correlation function computation over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        if True will return the generated or used tree for future use (default False)
    boxsize: float, optional
        finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning: bool, optional
        (experimental) if True will distribute mass among radial bings with a weighted kernel (default False)

    Returns
    -------
    rbins: array_like
        array containing radial bin edges
    Sv: array_like
        array containing structure function values
    """

    if rbins is None:
        r = np.sort(np.sqrt(np.sum((pos - np.median(pos, axis=0)) ** 2, axis=1)))
        rbins = 10 ** np.linspace(np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3)))

    built_tree = tree is None
    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(_f64(pos), _f64(m), _f64(softening), vel=v)  # build the tree if needed
    # Sort pos rather than applying the tree's stored permutation -- see the note in Potential. The
    # output is a binned aggregate, so sigma^2 was still a valid sample; the real failure was that a
    # supplied tree of a different size indexed out of bounds.
    idx = tree.TreewalkIndices if built_tree else _morton_order(_f64(pos))
    pos_sorted = np.take(pos, idx, axis=0)
    v_sorted = np.take(v, idx, axis=0)
    wt_sorted = np.take(m, idx, axis=0)
    if parallel:
        Sv = VelocityStructFunc_tree_parallel(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )
    else:
        Sv = VelocityStructFunc_tree(
            pos_sorted,
            v_sorted,
            wt_sorted,
            tree,
            rbins,
            max_bin_size_ratio=max_bin_size_ratio,
            theta=theta,
            boxsize=boxsize,
            weighted_binning=weighted_binning,
        )

    if return_tree:
        return rbins, Sv, tree
    else:
        return rbins, Sv


def ColumnDensity(
    pos,
    m,
    radii,
    rays=None,
    randomize_rays=False,
    healpix=False,
    tree=None,
    theta=0.5,
    return_tree=False,
    parallel=False,
    group_size=16,
    device="cpu",
):
    """Ray-traced or angle-binned column density calculation.

    Returns an estimate of the column density from the position of each particle integrated to infinity, assuming the particles are represented by uniform spheres. Note that optical depth can be obtained by supplying "sigma = opacity * mass" in place of mass, useful in situations where opacity is highly variable.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    radii: array_like
        shape (N,) array containing particle radii of the uniform spheres that we use to model the particles' mass distribution
    rays: None, int, or array_like, optional
        Which ray directions to raytrace the columns (default None). None uses the angular-binned method with 6 bins on the sky; an integer uses that many rays, with 6 giving the standard 6-ray grid and other numbers sampling random directions; a (N_rays,3) array gives the directions explicitly, and is normalized automatically.
    healpix: int, optional
        If nonzero, use a healpix ray grid with this resolution level NSIDE (default False)
    randomize_rays: bool, optional
        Randomize the orientation of the ray-grid *for each particle* (default False)
    parallel: bool, optional
        If True, will parallelize the column density over all available cores. (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    theta: float, optional
        Opening angle for the beam-traced angular bin estimator (default 0.5)
    return_tree: bool, optional
        return the tree used for future use (default False)
    group_size: int, optional
        Targets sharing one tree traversal, amortizing the descent (default 16). Bit-identical to the per-target walk for the ray-traced path; for rays=None the group opens a superset of nodes, so the answer shifts slightly and is somewhat more accurate. Ignored with randomize_rays or below COLUMN_GROUP_MIN_TARGETS targets; 1 gives the per-target walk.
    device: str, optional
        'cpu' (default) or 'cuda'. 'cuda' needs pytreegrav[cuda] and an NVIDIA GPU, and applies only to the ray-traced path (rays given, randomize_rays off). It is float32: relative error against this path is ~2e-6 typical, reaching ~1e-2 on the densest sightlines. It repacks and uploads the tree on every call -- for repeated evaluation hold a pytreegrav.cuda.CudaColumnDensity instead.

    Returns
    -------
    columns: array_like
        shape (N,N_rays) float array of column densities from particle centers integrated along the rays
    """

    if np.any(np.asarray(radii) <= 0):
        # Point-like, so no cross-section and nothing obscured -- the right h -> 0 limit, but it
        # drops their mass from the column, which should not pass unnoticed.
        warnings.warn(
            "Some particle radii are <= 0; these particles are point-like and contribute no column "
            "density. Supply a nonzero radius if their mass should be included."
        )

    pos = _f64(pos)
    if tree is None:
        tree = ConstructTree(
            pos,
            _f64(m),
            _f64(radii),
        )  # build the tree if needed
        # the tree was just built from pos, so its walk order already is pos's Morton order
        idx = tree.TreewalkIndices
    else:
        # A supplied tree may hold different particles than the targets, so its TreewalkIndices do
        # not index pos.  Sort the targets on their own, as PotentialTarget/AccelTarget do.
        idx = _morton_order(pos)
    pos_sorted = pos[idx]

    if type(rays) == int:
        if rays == 6:
            rays = np.vstack([np.eye(3), -np.eye(3)])  # 6-ray grid
        else:
            # generate a random grid of ray directions
            rays = np.random.normal(size=(rays, 3))  # normalize later
    elif type(rays) == np.ndarray:
        # check that the shape is correct
        if not len(rays.shape) == 2:
            raise Exception("rays array argument must be 2D.")
        elif rays.shape[1] != 3:
            raise Exception("rays array argument is not an array of 3D vectors.")
        rays = np.copy(rays)  # so that we don't overwrite the argument
    elif rays is not None:
        raise Exception("rays argument type is not supported")

    if healpix:
        # Imported here, not at module scope: healpy is not in install_requires (nothing else in the
        # package needs it), and `hp` was simply never bound, so this path raised NameError.
        try:
            import healpy as hp
        except ImportError as e:
            raise ImportError("healpix=... needs healpy: pip install healpy") from e
        nside = healpix
        npix = hp.nside2npix(nside)
        rays = np.array(hp.pix2vec(nside, np.arange(npix))).T

    if rays is not None:
        rays /= np.sqrt((rays * rays).sum(1))[:, None]  # normalize the ray vectors

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda":
        if rays is None or randomize_rays:
            raise ValueError("device='cuda' supports only the ray-traced path: pass rays, and not randomize_rays")
        from .cuda import CudaColumnDensity  # imported lazily; numba-cuda is an optional extra

        columns = CudaColumnDensity(tree)(pos_sorted, rays)
    # Grouping needs every target in a group to share the same ray directions, so randomize_rays --
    # which rotates the grid per target -- keeps the per-target walks.  Dispatched here rather than
    # inside ColumnDensity_tree: calling one parallel=True kernel from another serializes its prange.
    elif rays is not None and not randomize_rays and len(pos_sorted) >= COLUMN_GROUP_MIN_TARGETS:
        walk = ColumnDensity_grouped_parallel if parallel else ColumnDensity_grouped
        columns = walk(pos_sorted, rays, tree, group_size)
    elif rays is None and len(pos_sorted) >= COLUMN_GROUP_MIN_TARGETS:
        # Not merely faster: opening a superset of nodes splits each node's mass across the sky bins
        # more finely, roughly halving the bin-to-bin scatter. group_size=1 recovers the old result.
        walk = ColumnDensityBinned_grouped_parallel if parallel else ColumnDensityBinned_grouped
        columns = walk(pos_sorted, tree, theta, group_size)
    elif parallel:
        columns = ColumnDensity_tree_parallel(pos_sorted, tree, rays, randomize_rays=randomize_rays, theta=theta)
    else:
        columns = ColumnDensity_tree(pos_sorted, tree, rays, randomize_rays=randomize_rays, theta=theta)
    if np.any(np.isnan(columns)):
        print("WARNING some column densities are NaN!")
    unsorted = np.empty_like(columns)
    unsorted[idx] = columns  # undo the permutation
    columns = unsorted

    if return_tree:
        return columns, tree
    else:
        return columns
