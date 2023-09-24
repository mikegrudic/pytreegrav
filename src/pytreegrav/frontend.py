import numpy as np
from numba import njit, prange
from numpy import zeros_like, sqrt
from .kernel import *
from .octree import *
from .dynamic_tree import *
from .treewalk import *
from .bruteforce import *


def valueTestMethod(method):
    methods = ["adaptive", "bruteforce", "tree"]

    ## check if method is a str
    if type(method) != str:
        raise TypeError("Invalid method type %s, must be str" % type(method))

    ## check if method is a valid method
    if method not in methods:
        raise ValueError(
            "Invalid method %s. Must be one of: %s" % (method, str(methods))
        )


def ConstructTree(pos, m, softening=None, quadrupole=False, vel=None):
    """Builds a tree containing particle data, for subsequent potential/field evaluation

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    softening: array_like or None, optional
        shape (N,) array of particle softening lengths - these give the radius of compact support of the M4 cubic spline mass distribution of each particle
    quadrupole: bool, optional
        Whether to store quadrupole moments (default False)
    vel: bool, optional
        Whether to store node velocities in the tree (default False)

    Returns
    -------
    tree: octree
        Octree instance built from particle data
    """
    if softening is None:
        softening = zeros_like(m)
    if not (
        np.all(np.isfinite(pos))
        and np.all(np.isfinite(m))
        and np.all(np.isfinite(softening))
    ):
        print(
            "Invalid input detected - aborting treebuild to avoid going into an infinite loop!"
        )
        raise
    if vel is None:
        return Octree(pos, m, softening, quadrupole=quadrupole)
    else:
        return DynamicOctree(pos, m, softening, vel, quadrupole=quadrupole)


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
        shape (N,) array containing kernel support radii for gravitational softening -  - these give the radius of compact support of the M4 cubic spline mass distribution - set to 0 by default
    theta: float, optional
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
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

    Returns
    -------
    phi: array_like
        shape (N,) array of potentials at the particle positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    if softening is None:
        softening = np.zeros_like(m)

    # figure out which method to use
    if method == "adaptive":
        if len(pos) > 1000:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            phi = Potential_bruteforce_parallel(pos, m, softening, G=G)
        else:
            phi = Potential_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                np.float64(pos),
                np.float64(m),
                np.float64(softening),
                quadrupole=quadrupole,
            )  # build the tree if needed
        idx = tree.TreewalkIndices

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos, idx, axis=0)
        h_sorted = np.take(softening, idx)

        if parallel:
            phi = PotentialTarget_tree_parallel(
                pos_sorted, h_sorted, tree, theta=theta, G=G, quadrupole=quadrupole
            )
        else:
            phi = PotentialTarget_tree(
                pos_sorted, h_sorted, tree, theta=theta, G=G, quadrupole=quadrupole
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
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
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
        softening_target = np.zeros(len(pos_target))
    if softening_source is None and pos_source is not None:
        softening_source = np.zeros(len(pos_source))

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
                np.float64(pos_source),
                np.float64(m_source),
                np.float64(softening_source),
                quadrupole=quadrupole,
            )  # build the tree if needed
        if parallel:
            phi = PotentialTarget_tree_parallel(
                pos_target,
                softening_target,
                tree,
                theta=theta,
                G=G,
                quadrupole=quadrupole,
            )
        else:
            phi = PotentialTarget_tree(
                pos_target,
                softening_target,
                tree,
                theta=theta,
                G=G,
                quadrupole=quadrupole,
            )

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
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
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

    Returns
    -------
    g: array_like
        shape (N,3) array of acceleration vectors at the particle positions
    """

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

    if softening is None:
        softening = np.zeros_like(m)

    # figure out which method to use
    if method == "adaptive":
        if len(pos) > 1000:
            method = "tree"
        else:
            method = "bruteforce"

    if method == "bruteforce":  # we're using brute force
        if parallel:
            g = Accel_bruteforce_parallel(pos, m, softening, G=G)
        else:
            g = Accel_bruteforce(pos, m, softening, G=G)
        if return_tree:
            tree = None
    else:  # we're using the tree algorithm
        if tree is None:
            tree = ConstructTree(
                np.float64(pos),
                np.float64(m),
                np.float64(softening),
                quadrupole=quadrupole,
            )  # build the tree if needed
        idx = tree.TreewalkIndices

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos, idx, axis=0)
        h_sorted = np.take(softening, idx)

        if parallel:
            g = AccelTarget_tree_parallel(
                pos_sorted, h_sorted, tree, theta=theta, G=G, quadrupole=quadrupole
            )
        else:
            g = AccelTarget_tree(
                pos_sorted, h_sorted, tree, theta=theta, G=G, quadrupole=quadrupole
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
        cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
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
        softening_target = np.zeros(len(pos_target))
    if softening_source is None and pos_source is not None:
        softening_source = np.zeros(len(pos_source))

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
                np.float64(pos_source),
                np.float64(m_source),
                np.float64(softening_source),
                quadrupole=quadrupole,
            )  # build the tree if needed
        if parallel:
            g = AccelTarget_tree_parallel(
                pos_target,
                softening_target,
                tree,
                theta=theta,
                G=G,
                quadrupole=quadrupole,
            )
        else:
            g = AccelTarget_tree(
                pos_target,
                softening_target,
                tree,
                theta=theta,
                G=G,
                quadrupole=quadrupole,
            )

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
        rbins = 10 ** np.linspace(
            np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3))
        )

    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(
            np.float64(pos), np.float64(m), np.float64(softening)
        )  # build the tree if needed
    idx = tree.TreewalkIndices

    # sort by the order they appear in the treewalk to improve access pattern efficiency
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
        rbins = 10 ** np.linspace(
            np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3))
        )

    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(
            np.float64(pos), np.float64(m), np.float64(softening), vel=v
        )  # build the tree if needed
    idx = tree.TreewalkIndices

    # sort by the order they appear in the treewalk to improve access pattern efficiency
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
    """Computes the structure function for a vector field: the average value of |v(x)-v(x+r)|^2, in radial bins for r

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
        rbins = 10 ** np.linspace(
            np.log10(r[10]), np.log10(r[-1]), int(len(r) ** (1.0 / 3))
        )

    if tree is None:
        softening = np.zeros_like(m)
        tree = ConstructTree(
            np.float64(pos), np.float64(m), np.float64(softening), vel=v
        )  # build the tree if needed
    idx = tree.TreewalkIndices

    # sort by the order they appear in the treewalk to improve access pattern efficiency
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
    pos, m, radii, rays=None, tree=None, return_tree=False, parallel=False
):
    """Ray-traced column density calculation.

    Returns the column density from the position of each particle integrated to
    infinity, assuming the particles are represented by uniform spheres. Note
    that optical depth can be obtained by supplying "σ = opacity * mass" in
    place of mass, useful in situations where opacity is highly variable.

    Parameters
    ----------
    pos: array_like
        shape (N,3) array of particle positions
    m: array_like
        shape (N,) array of particle masses
    radii: array_like
        shape (N,) array containing particle radii of the uniform spheres that
        we use to model the particles' mass distribution
    rays: optional
        Which ray directions to raytrace the columns. DEFAULT: The simple
        6-ray grid.
        OPTION 2: Give a (N_rays,3) array of vectors specifying the
        directions, which will be automatically normalized.
        OPTION 3: Give an integer number N to generate a raygrid of N random
        directions.
    parallel: bool, optional
        If True, will parallelize the column density over all available cores.
        (default False)
    tree: Octree, optional
        optional pre-generated Octree: this can contain any set of particles,
        not necessarily the target particles at pos (default None)
    return_tree: bool, optional
        return the tree used for future use (default False)

    Returns
    -------
    columns: array_like
        shape (N,N_rays) float array of column densities from particle
        centers integrated along the rays
    """

    if tree is None:
        tree = ConstructTree(
            np.float64(pos),
            np.float64(m),
            np.float64(radii),
        )  # build the tree if needed
    idx = tree.TreewalkIndices

    # generate the array of rays
    if rays is None:
        rays = np.vstack([np.eye(3), -np.eye(3)])  # 6-ray grid
    elif type(rays) == int:
        # generate a random grid of ray directions
        rays = np.random.normal(size=(rays, 3))  # normalize later
    elif type(rays) == np.ndarray:
        # check that the shape is correct
        if not len(rays.shape) == 2:
            raise Exception("rays array argument must be 2D.")
        elif rays.shape[1] != 3:
            raise Exception("rays array argument is not an array of 3D vectors.")
        rays = np.copy(rays)  # so that we don't overwrite the argument
    else:
        raise Exception("rays argument type is not supported")

    rays /= np.sqrt((rays * rays).sum(1))[:, None]  # normalize the ray vectors

    pos_sorted = np.take(pos, idx, axis=0)

    if parallel:
        columns = ColumnDensity_tree_parallel(pos_sorted, rays, tree)
    else:
        columns = ColumnDensity_tree(pos_sorted, rays, tree)

    columns = np.take(columns, idx.argsort(), axis=0)

    if return_tree:
        return columns, tree
    else:
        return columns
