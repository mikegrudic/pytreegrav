from numpy import sqrt, empty, zeros, empty_like, zeros_like, dot, fabs
from numba import njit, prange, get_num_threads
from math import copysign
from .kernel import *
import numpy as np


@njit(fastmath=True)
def NearestImage(x, boxsize):
    if abs(x) > boxsize / 2:
        return -copysign(boxsize - abs(x), x)
    else:
        return x
    # define TMP_WRAP_X_S(x,y,z,sign) (x=((x)>boxHalf_X)?((x)-boxSize_X):(((x)<-boxHalf_X)?((x)+boxSize_X):(x)))


@njit(fastmath=True)
def PotentialWalk(pos, tree, softening=0, no=-1, theta=0.7):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    phi = 0
    dx = np.empty(3, dtype=np.float64)

    while no > -1:
        r = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r += dx[k] * dx[k]
        r = sqrt(r)
        h = max(tree.Softenings[no], softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            if r > 0:  # by default we neglect the self-potential
                if r < h:
                    phi += tree.Masses[no] * PotentialKernel(r, h)
                else:
                    phi -= tree.Masses[no] / r
            no = tree.NextBranch[no]
        elif r > max(
            tree.Sizes[no] / theta + tree.Deltas[no],
            h + tree.Sizes[no] * 0.6 + tree.Deltas[no],
        ):  # if we satisfy the criteria for accepting the monopole
            phi -= tree.Masses[no] / r
            no = tree.NextBranch[no]
        else:  # open the node
            no = tree.FirstSubnode[no]

    return phi


@njit(fastmath=True)
def PotentialWalk_quad(pos, tree, softening=0, no=-1, theta=0.7):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance. Uses the quadrupole expansion.
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    phi = 0
    dx = np.empty(3, dtype=np.float64)

    while no > -1:
        r = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r += dx[k] * dx[k]
        r = sqrt(r)
        h = max(tree.Softenings[no], softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            if r > 0:  # by default we neglect the self-potential
                if r < h:
                    phi += tree.Masses[no] * PotentialKernel(r, h)
                else:
                    phi -= tree.Masses[no] / r
            no = tree.NextBranch[no]
        elif r > max(
            tree.Sizes[no] / theta + tree.Deltas[no],
            h + tree.Sizes[no] * 0.6 + tree.Deltas[no],
        ):  # if we satisfy the criteria for accepting the monopole
            phi -= tree.Masses[no] / r
            # phi -= 0.5 * np.dot(np.dot(dx,tree.Quadrupoles[no]),dx)/(r*r*r*r*r) # Potential from the quadrupole moment
            quad = tree.Quadrupoles[no]
            r5inv = 1 / (r * r * r * r * r)
            for k in range(3):
                for l in range(3):
                    phi -= 0.5 * dx[k] * quad[k, l] * dx[l] * r5inv
            no = tree.NextBranch[no]
        else:  # open the node
            no = tree.FirstSubnode[no]

    return phi


@njit(fastmath=True)
def AccelWalk(
    pos, tree, softening=0, no=-1, theta=0.7
):  # ,include_self_potential=False):
    """Returns the gravitational acceleration field at position x by performing the Barnes-Hut treewalk using the provided octree instance
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree instance storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    g = zeros(3, dtype=np.float64)
    dx = np.empty(3, dtype=np.float64)

    while no > -1:  # loop until we get to the end of the tree
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        h = max(tree.Softenings[no], softening)

        sum_field = False

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            if r > 0:  # no self-force
                if r < h:  # within the softening radius
                    fac = tree.Masses[no] * ForceKernel(
                        r, h
                    )  # fac stores the quantity M(<R)/R^3 to be used later for force computation
                else:  # use point mass force
                    fac = tree.Masses[no] / (r * r2)
                sum_field = True
            no = tree.NextBranch[no]
        elif r > max(
            tree.Sizes[no] / theta + tree.Deltas[no],
            h + tree.Sizes[no] * 0.6 + tree.Deltas[no],
        ):  # if we satisfy the criteria for accepting the monopole
            fac = tree.Masses[no] / (r * r2)
            sum_field = True
            no = tree.NextBranch[no]  # go to the next branch in the tree
        else:  # open the node
            no = tree.FirstSubnode[no]
            continue

        if sum_field:  # OK, we have fac for this element and can now sum the force
            for k in range(3):
                g[k] += fac * dx[k]

    return g


@njit(fastmath=True)
def AccelWalk_quad(
    pos, tree, softening=0, no=-1, theta=0.7
):  # ,include_self_potential=False):
    """Returns the gravitational acceleration field at position x by performing the Barnes-Hut treewalk using the provided octree instance. Uses the quadrupole expansion.
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree instance storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    g = zeros(3, dtype=np.float64)
    dx = np.empty(3, dtype=np.float64)

    while no > -1:  # loop until we get to the end of the tree
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        h = max(tree.Softenings[no], softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            if r > 0:  # no self-force
                if r < h:  # within the softening radius
                    fac = tree.Masses[no] * ForceKernel(
                        r, h
                    )  # fac stores the quantity M(<R)/R^3 to be used later for force computation
                else:  # use point mass force
                    fac = tree.Masses[no] / (r * r2)
            for k in range(3):
                g[k] += fac * dx[k]  # monopole
            no = tree.NextBranch[no]
            continue
        elif r > max(
            tree.Sizes[no] / theta + tree.Deltas[no],
            h + tree.Sizes[no] * 0.6 + tree.Deltas[no],
        ):  # if we satisfy the criteria for accepting the multipole expansion
            fac = tree.Masses[no] / (r * r2)
            quad = tree.Quadrupoles[no]
            #            g -= dot(tree.Quadrupoles[no], dx/r)/(r*r*r*r) - 2.5*dot(dx/r, dot(tree.Quadrupoles[no], dx/r))*dx/(r*r*r*r*r)
            r5inv = 1 / (r2 * r2 * r)
            quad_fac = 0
            for k in range(3):
                g[k] += fac * dx[k]  # monopole
                for l in range(3):  # prepass to compute contraction of quad with dx
                    quad_fac += quad[k, l] * dx[k] * dx[l]
            quad_fac *= r5inv / r2
            for k in range(3):
                g[k] += 2.5 * quad_fac * dx[k]
                for l in range(3):
                    g[k] -= quad[k, l] * dx[l] * r5inv

            no = tree.NextBranch[no]  # go to the next branch in the tree
        else:  # open the node
            no = tree.FirstSubnode[no]
            continue

    return g


def PotentialTarget_tree(
    pos_target, softening_target, tree, G=1.0, theta=0.7, quadrupole=False
):
    """Returns the gravitational potential at the specified points, given a tree containing the mass distribution
    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the potential
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all potential computations
    tree -- Octree instance containing the positions, masses, and softenings of the source particles
    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)
    Returns:
    shape (N,) array of potential values at each point in pos
    """
    result = empty(pos_target.shape[0])
    if quadrupole:
        for i in prange(pos_target.shape[0]):
            result[i] = G * PotentialWalk_quad(
                pos_target[i], tree, softening=softening_target[i], theta=theta
            )
    else:
        for i in prange(pos_target.shape[0]):
            result[i] = G * PotentialWalk(
                pos_target[i], tree, softening=softening_target[i], theta=theta
            )
    return result


# JIT this function and its parallel version
PotentialTarget_tree_parallel = njit(PotentialTarget_tree, fastmath=True, parallel=True)
PotentialTarget_tree = njit(PotentialTarget_tree, fastmath=True)


def AccelTarget_tree(
    pos_target, softening_target, tree, G=1.0, theta=0.7, quadrupole=False
):
    """Returns the gravitational acceleration at the specified points, given a tree containing the mass distribution
    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the field
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all accel computations
    tree -- Octree instance containing the positions, masses, and softenings of the source particles
    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)
    Returns:
    shape (N,3) array of acceleration values at each point in pos_target
    """
    if softening_target is None:
        softening_target = zeros(pos_target.shape[0])
    result = empty(pos_target.shape)
    if quadrupole:
        for i in prange(pos_target.shape[0]):
            result[i] = G * AccelWalk_quad(
                pos_target[i], tree, softening=softening_target[i], theta=theta
            )
    else:
        for i in prange(pos_target.shape[0]):
            result[i] = G * AccelWalk(
                pos_target[i], tree, softening=softening_target[i], theta=theta
            )
    return result


# JIT this function and its parallel version
AccelTarget_tree_parallel = njit(AccelTarget_tree, fastmath=True, parallel=True)
AccelTarget_tree = njit(AccelTarget_tree, fastmath=True)


@njit(fastmath=True)
def do_weighted_binning(tree, no, rbins, mbin, r, r_idx, quantity):
    h = 0.5 * tree.Sizes[no]
    Nbins = rbins.shape[0] - 1
    if (r + h < rbins[r_idx + 1]) and (r - h > rbins[r_idx]):
        mbin[r_idx] += tree.Masses[no] * quantity
    else:
        min_bin = int((np.log10((r - h) / rbins[0]) / np.log10(rbins[1] / rbins[0])))
        max_bin = min(
            int(np.log10((r + h) / rbins[0]) / np.log10(rbins[1] / rbins[0]) + 1), Nbins
        )
        total_wt = 0
        for i in range(
            min_bin, max_bin
        ):  # range(min_bin,max_bin): # first the prepass to get the total weight
            # (r > rbins[i] and r < rbins[i+1]) or dr < 0.5*tree.Sizes[no]:
            i1, i2 = max(r - h, rbins[i]), min(r + h, rbins[i + 1])
            overlap = i2 - i1
            if overlap > 0:  # if there's overlap
                reff = 0.5 * (i1 + i2)  # sqrt(rbins[i]*rbins[i+1])
                dr = fabs(r - reff)
                wt = max(0, 1 - dr * dr / (h * h)) * overlap
                total_wt += wt

        for i in range(
            min_bin, max_bin
        ):  # range(min_bin,max_bin): # then distribute according to the normalized weighting
            i1, i2 = max(r - h, rbins[i]), min(r + h, rbins[i + 1])
            overlap = i2 - i1
            if overlap > 0:  # if there's overlap
                reff = 0.5 * (i1 + i2)  # sqrt(rbins[i]*rbins[i+1])
                dr = fabs(r - reff)
                wt = max(0, 1 - dr * dr / (h * h)) * overlap / total_wt
                mbin[i] += wt * tree.Masses[no] * quantity


@njit(fastmath=True)
def DensityCorrWalk(
    pos,
    tree,
    rbins,
    max_bin_size_ratio=100,
    theta=0.7,
    no=-1,
    boxsize=0,
    weighted_binning=False,
):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure

    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index

    Nbins = rbins.shape[0] - 1
    mbin = zeros(Nbins)
    counts = zeros(Nbins)
    rmin = rbins[0]
    rmax = rbins[-1]
    dx = np.empty(3, dtype=np.float64)

    logr_min = np.log10(rmin)
    logr_max = np.log10(rmax)
    dlogr = logr_max - logr_min

    while no > -1:
        r = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            if boxsize > 0:
                dx[k] = NearestImage(dx[k], boxsize)
            r += dx[k] * dx[k]

        r = sqrt(r)
        #        theta = min(1,theta * np.exp(0.5*np.random.normal())) # if we randomize the opening criteria a bit we'll get fewer binning artifacts
        within_bounds = (r > rmin) and (r < rmax)
        if within_bounds:
            logr = np.log10(r)
            r_idx = int(Nbins * (logr - logr_min) / dlogr)
            if no < tree.NumParticles:
                mbin[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            elif (
                r
                > max(
                    tree.Sizes[no] / theta + tree.Deltas[no],
                    tree.Sizes[no] * 0.6 + tree.Deltas[no],
                )
            ) and (
                tree.Sizes[no] < max_bin_size_ratio * (rbins[r_idx + 1] - rbins[r_idx])
            ):
                if weighted_binning:
                    do_weighted_binning(tree, no, rbins, mbin, r, r_idx, 1)
                else:
                    rnew = r + (np.random.rand() - 0.5) * tree.Sizes[no]
                    r_idx = int(Nbins * (np.log10(rnew) - logr_min) / dlogr)
                    mbin[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
        else:
            if no < tree.NumParticles:
                no = tree.NextBranch[no]
            elif r > max(
                tree.Sizes[no] / theta + tree.Deltas[no],
                tree.Sizes[no] * 0.6 + tree.Deltas[no],
            ):
                no = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
    return mbin


def DensityCorrFunc_tree(
    pos,
    tree,
    rbins,
    max_bin_size_ratio=100,
    theta=0.7,
    boxsize=0,
    weighted_binning=False,
):
    """Returns the average mass in radial bins surrounding a point

    Arguments:
    pos -- shape (N,3) array of particle positions
    tree -- Octree instance containing the positions, masses, and softenings of the source particles

    Optional arguments:
    rbins -- 1D array of radial bin edges - if None will use heuristics to determine sensible bins
    max_bin_size_ratio -- controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width

    Returns:
    mbins -- arrays containing total mass in each bin
    """
    Nthreads = get_num_threads()
    mbin = zeros((Nthreads, rbins.shape[0] - 1))
    # break into chunks for parallelization
    for chunk in prange(Nthreads):
        for i in range(chunk, pos.shape[0], Nthreads):
            dmbin = DensityCorrWalk(
                pos[i],
                tree,
                rbins,
                max_bin_size_ratio=max_bin_size_ratio,
                theta=theta,
                boxsize=boxsize,
                weighted_binning=weighted_binning,
            )
            for j in range(mbin.shape[1]):
                mbin[chunk, j] += dmbin[j]
    return mbin.sum(0) / pos.shape[0]


# JIT this function and its parallel version
DensityCorrFunc_tree_parallel = njit(DensityCorrFunc_tree, fastmath=True, parallel=True)
DensityCorrFunc_tree = njit(DensityCorrFunc_tree, fastmath=True)


@njit(fastmath=True)
def VelocityCorrWalk(
    pos,
    vel,
    tree,
    rbins,
    max_bin_size_ratio=100,
    theta=0.7,
    no=-1,
    boxsize=0,
    weighted_binning=False,
):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance

    Arguments:
    pos - (3,) array containing position of interest
    vel - (3,) array containing velocity of point of interest
    tree - octree object storing the tree structure

    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index

    Nbins = rbins.shape[0] - 1
    binsums = zeros(Nbins)
    wtsums = zeros(Nbins)
    #    counts = zeros(Nbins)
    rmin = rbins[0]
    rmax = rbins[-1]
    dx = np.empty(3, dtype=np.float64)

    logr_min = np.log10(rmin)
    logr_max = np.log10(rmax)
    dlogr = logr_max - logr_min

    while no > -1:
        r = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            if boxsize > 0:
                dx[k] = NearestImage(dx[k], boxsize)
            r += dx[k] * dx[k]
        r = sqrt(r)
        #        theta = min(1,theta * np.exp(0.5*np.random.normal())) # if we randomize the opening criteria a bit we'll get fewer binning artifacts
        within_bounds = (r > rmin) and (r < rmax)
        if within_bounds:
            logr = np.log10(r)
            r_idx = int(Nbins * (logr - logr_min) / dlogr)
            if no < tree.NumParticles:
                vprod = 0
                for k in range(3):
                    vprod += vel[k] * tree.Velocities[no][k] * tree.Masses[no]
                binsums[r_idx] += vprod
                wtsums[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            elif r > max(
                tree.Sizes[no] / theta + tree.Deltas[no],
                tree.Sizes[no] * 0.6 + tree.Deltas[no],
            ) and tree.Sizes[no] < max_bin_size_ratio * (
                rbins[r_idx + 1] - rbins[r_idx]
            ):
                vprod = 0
                for k in range(3):
                    vprod += vel[k] * tree.Velocities[no][k]
                if weighted_binning:
                    do_weighted_binning(tree, no, rbins, binsums, r, r_idx, vprod)
                    do_weighted_binning(tree, no, rbins, wtsums, r, r_idx, 1)
                else:
                    rnew = r + (np.random.rand() - 0.5) * tree.Sizes[no]
                    r_idx = int(Nbins * (np.log10(rnew) - logr_min) / dlogr)
                    binsums[r_idx] += vprod * tree.Masses[no]
                    wtsums[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
        else:
            if no < tree.NumParticles:
                no = tree.NextBranch[no]
            elif r > max(
                tree.Sizes[no] / theta + tree.Deltas[no],
                tree.Sizes[no] * 0.6 + tree.Deltas[no],
            ):
                no = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
    return wtsums, binsums


def VelocityCorrFunc_tree(
    pos,
    vel,
    weight,
    tree,
    rbins,
    max_bin_size_ratio=100,
    theta=0.7,
    boxsize=0,
    weighted_binning=False,
):
    """Returns the average mass in radial bins surrounding a point

    Arguments:
    pos -- shape (N,3) array of particle positions
    tree -- Octree instance containing the positions, masses, and softenings of the source particles

    Optional arguments:
    rbins -- 1D array of radial bin edges - if None will use heuristics to determine sensible bins
    max_bin_size_ratio -- controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 0.5)

    Returns:
    mbins -- arrays containing total mass in each bin
    """
    Nthreads = get_num_threads()
    mbin = zeros((Nthreads, rbins.shape[0] - 1))
    wtsum = zeros_like(mbin)
    # break into chunks for parallelization
    for chunk in prange(Nthreads):
        for i in range(chunk, pos.shape[0], Nthreads):
            dwtsum, dmbin = VelocityCorrWalk(
                pos[i],
                vel[i],
                tree,
                rbins,
                max_bin_size_ratio=max_bin_size_ratio,
                theta=theta,
                boxsize=boxsize,
                weighted_binning=weighted_binning,
            )
            for j in range(mbin.shape[1]):
                mbin[chunk, j] += dmbin[j] * weight[i]
                wtsum[chunk, j] += weight[i] * dwtsum[j]
    return mbin.sum(0) / wtsum.sum(0)


# JIT this function and its parallel version
VelocityCorrFunc_tree_parallel = njit(
    VelocityCorrFunc_tree, fastmath=True, parallel=True
)
VelocityCorrFunc_tree = njit(VelocityCorrFunc_tree, fastmath=True)


@njit(fastmath=True)
def VelocityStructWalk(
    pos,
    vel,
    tree,
    rbins,
    max_bin_size_ratio=100,
    theta=0.7,
    no=-1,
    boxsize=0,
    weighted_binning=False,
):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance

    Arguments:
    pos - (3,) array containing position of interest
    vel - (3,) array containing velocity of point of interest
    tree - octree object storing the tree structure

    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index

    Nbins = rbins.shape[0] - 1
    binsums = zeros(Nbins)
    wtsums = zeros(Nbins)
    rmin = rbins[0]
    rmax = rbins[-1]
    dx = np.empty(3, dtype=np.float64)
    logr_min = np.log10(rmin)
    logr_max = np.log10(rmax)
    dlogr = logr_max - logr_min

    while no > -1:
        r = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            if boxsize > 0:
                dx[k] = NearestImage(dx[k], boxsize)
            r += dx[k] * dx[k]
        r = sqrt(r)

        #        theta = min(1,theta * np.exp(0.5*np.random.normal())) # if we randomize the opening criteria a bit we'll get fewer binning artifacts
        within_bounds = (r > rmin) and (r < rmax)
        if within_bounds:
            logr = np.log10(r)
            r_idx = int(Nbins * (logr - logr_min) / dlogr)
            if no < tree.NumParticles:
                vprod = 0
                for k in range(3):
                    vprod += (
                        (vel[k] - tree.Velocities[no][k])
                        * (vel[k] - tree.Velocities[no][k])
                        * tree.Masses[no]
                    )
                binsums[r_idx] += vprod
                wtsums[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            elif r > max(
                tree.Sizes[no] / theta + tree.Deltas[no],
                tree.Sizes[no] * 0.6 + tree.Deltas[no],
            ) and (
                tree.Sizes[no] < max_bin_size_ratio * (rbins[r_idx + 1] - rbins[r_idx])
            ):
                vprod = 0
                for k in range(3):
                    vprod += (vel[k] - tree.Velocities[no][k]) * (
                        vel[k] - tree.Velocities[no][k]
                    )
                vprod += tree.VelocityDisp[no]
                if weighted_binning:
                    do_weighted_binning(tree, no, rbins, binsums, r, r_idx, vprod)
                    do_weighted_binning(tree, no, rbins, wtsums, r, r_idx, 1)
                else:
                    rnew = r + (np.random.rand() - 0.5) * tree.Sizes[no]
                    r_idx = int(Nbins * (np.log10(rnew) - logr_min) / dlogr)
                    binsums[r_idx] += vprod * tree.Masses[no]
                    wtsums[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
        else:
            if no < tree.NumParticles:
                no = tree.NextBranch[no]
            elif r > max(
                tree.Sizes[no] / theta + tree.Deltas[no],
                tree.Sizes[no] * 0.6 + tree.Deltas[no],
            ):
                no = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
    return wtsums, binsums


def VelocityStructFunc_tree(
    pos,
    vel,
    weight,
    tree,
    rbins,
    max_bin_size_ratio=100,
    theta=0.7,
    boxsize=0,
    weighted_binning=False,
):
    """Returns the average mass in radial bins surrounding a point

    Arguments:
    pos -- shape (N,3) array of particle positions
    tree -- Octree instance containing the positions, masses, and softenings of the source particles

    Optional arguments:
    rbins -- 1D array of radial bin edges - if None will use heuristics to determine sensible bins
    max_bin_size_ratio -- controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 0.5)

    Returns:
    mbins -- arrays containing total mass in each bin
    """

    Nthreads = get_num_threads()
    mbin = zeros((Nthreads, rbins.shape[0] - 1))
    wtsum = zeros_like(mbin)
    # break into chunks for parallelization
    for chunk in prange(Nthreads):
        for i in range(chunk, pos.shape[0], Nthreads):
            dwtsum, dmbin = VelocityStructWalk(
                pos[i],
                vel[i],
                tree,
                rbins,
                max_bin_size_ratio=max_bin_size_ratio,
                theta=theta,
                boxsize=boxsize,
                weighted_binning=weighted_binning,
            )
            for j in range(mbin.shape[1]):
                mbin[chunk, j] += dmbin[j] * weight[i]
                wtsum[chunk, j] += weight[i] * dwtsum[j]
    return mbin.sum(0) / wtsum.sum(0)


# JIT this function and its parallel version
VelocityStructFunc_tree_parallel = njit(
    VelocityStructFunc_tree, fastmath=True, parallel=True
)
VelocityStructFunc_tree = njit(VelocityStructFunc_tree, fastmath=True)


@njit(fastmath=True)
def ColumnDensityWalk_multiray(pos, rays, tree, no=-1):
    """Returns the integrated column density to infinity from pos, in the directions given by the rays argument

    Arguments:
    pos - (3,) array containing position of interest
    rays - (N_rays, 3) array of unit vectors
    tree - octree object storing the tree structure

    Returns:
    columns - (N_rays,) array of column densities along directions given by rays

    Keyword arguments:
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index

    N_rays = rays.shape[0]
    columns = np.zeros(N_rays)
    dx = np.empty(3, dtype=np.float64)
    z_ray = np.zeros(
        N_rays
    )  # perpendicular distances of elements to nearest point on rays

    fac_density = 3 / (4 * np.pi)

    while no > -1:
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        for i in range(N_rays):
            z_ray[i] = rays[i, 0] * dx[0] + rays[i, 1] * dx[1] + rays[i, 2] * dx[2]
        #        print(f"r={r}, z_ray={z_ray}")
        h_no = tree.Softenings[no]
        h_no_inv = 1.0 / h_no
        h = h_no  # max(h_no,softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            # add the particle's column if it's in the right direction
            fac = (
                fac_density * tree.Masses[no] * h_no_inv * h_no_inv
            )  # assumes uniform sphere geometry
            for i in range(N_rays):
                r_proj = sqrt(r2 - z_ray[i] * z_ray[i])
                q = r_proj * h_no_inv
                if r_proj < h_no:
                    if (
                        r > h_no
                    ):  # not overlapping the target point - integrate the whole cell
                        if z_ray[i] < 0:
                            continue  # not on the ray
                        columns[i] += fac * 2 * sqrt(1 - q * q)
                    else:  # overlapping, so need to integrate only a portion of the cell - this case includes the self-shielding if the point is in the tree!
                        dz = z_ray[i] * h_no_inv
                        columns[i] += fac * (dz + sqrt(1 - q * q))

            no = tree.NextBranch[no]

        else:  # we have a node, need to check if it intersects a ray
            node_intersects_ray = False
            R_eff = (
                tree.Sizes[no] * 0.8660254037844386 + tree.Deltas[no]
            )  # effective search radius from center of mass
            for i in range(N_rays):
                if (
                    r < h + R_eff
                ):  # if node contains the origin then it must intersect all rays
                    node_intersects_ray = True
                    break
                elif (z_ray[i] > 0) and (
                    (r2 - z_ray[i] * z_ray[i])
                    < (tree.Softenings[no] + R_eff) * (tree.Softenings[no] + R_eff)
                ):  # if perpendicular distance is less than node effective size
                    node_intersects_ray = True
                    break

            if node_intersects_ray:
                no = tree.FirstSubnode[no]  # open the node
            else:
                no = tree.NextBranch[
                    no
                ]  # no intersection with any way, so go to next node

    return columns


@njit(fastmath=True)
def ColumnDensityWalk(pos, ray, tree, no=-1):
    """Returns the integrated column density to infinity from pos, in the directions given by the rays argument

    Arguments:
    pos - (3,) array containing position of interest
    ray - (3,) array with the unit vector of the ray
    tree - octree object storing the tree structure

    Returns:
    columns - (N_rays,) array of column densities along directions given by rays

    Keyword arguments:
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index

    column = 0
    dx = np.empty(3, dtype=np.float64)
    z_ray = 0  # perpendicular distances of elements to nearest point on rays
    fac_density = 3 / (4 * np.pi)

    while no > -1:
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        #        for i in range(N_rays):
        z_ray = ray[0] * dx[0] + ray[1] * dx[1] + ray[2] * dx[2]
        #        print(f"r={r}, z_ray={z_ray}")
        h_no = tree.Softenings[no]
        h_no_inv = 1.0 / h_no
        h = h_no  # max(h_no,softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            # add the particle's column if it's in the right direction
            fac = fac_density * tree.Masses[no] * h_no_inv * h_no_inv
            # assumes uniform sphere geometry
            #            for i in range(N_rays):
            r_proj = sqrt(r2 - z_ray * z_ray)
            q = r_proj * h_no_inv
            if r_proj < h_no:
                if (
                    r > h_no
                ):  # not overlapping the target point - integrate the whole cell
                    if z_ray > 0:
                        column += fac * 2 * sqrt(1 - q * q)
                else:  # overlapping, so need to integrate only a portion of the cell - this case includes the self-shielding if the point is in the tree!
                    dz = z_ray * h_no_inv
                    column += fac * (dz + sqrt(1 - q * q))
            no = tree.NextBranch[no]

        else:  # we have a node, need to check if it intersects a ray
            node_intersects_ray = False
            R_eff = tree.Sizes[no] * 0.8660254037844386 + tree.Deltas[no]
            # effective search radius from center of mass
            if r < h + R_eff:
                # if node contains the origin then it must intersect all rays
                node_intersects_ray = True
            elif (z_ray > 0) and (
                (r2 - z_ray * z_ray)
                < (tree.Softenings[no] + R_eff) * (tree.Softenings[no] + R_eff)
            ):  # if perpendicular distance is less than node effective size
                node_intersects_ray = True

            if node_intersects_ray:
                no = tree.FirstSubnode[no]  # open the node
            else:  # no intersection with any way, so go to next node
                no = tree.NextBranch[no]
    return column


def ColumnDensity_tree(pos_target, rays, tree):
    """Returns the gravitational potential at the specified points, given a
    tree containing the mass distribution
    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the
    potential
    tree -- Octree instance containing the positions, masses, and softenings of
    the source particles
    """
    result = empty((pos_target.shape[0], rays.shape[0]))
    for i in range(rays.shape[0]):
        # outer loop over rays - empirically better access pattern
        for j in prange(pos_target.shape[0]):
            result[j, i] = ColumnDensityWalk(pos_target[j], rays[i], tree)
    return result


ColumnDensity_tree_parallel = njit(ColumnDensity_tree, fastmath=True, parallel=True)
ColumnDensity_tree = njit(ColumnDensity_tree, fastmath=True)
