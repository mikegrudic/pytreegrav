from numpy import sqrt, empty, zeros, empty_like, zeros_like, dot, fabs
from numba import njit, prange, get_num_threads, parallel_chunksize, int64, float64
from math import copysign
from .kernel import *
from .misc import *
import numpy as np
from scipy.spatial.transform import Rotation as R

# Above this ray count ColumnDensity_tree drops the bundled multi-ray walk for one single-ray walk
# per direction.  ColumnDensityWalk_multiray traverses once but opens the union of every ray's node
# set and tests each accepted element against all rays, so it costs O(N_rays^2) vs O(N_rays).  Both
# are exact, so this only trades speed.  Measured multiray/(N_rays x singleray) at N=30000:
# 0.73x at 12 rays, 1.21x at 48, 2.07x at 192 -- crossover near 24.
MULTIRAY_MAX_RAYS = 24

# Below this target count ColumnDensity keeps the per-target walks.  Grouping pads the acceptance
# reach by the group's spatial extent, so it needs Morton-adjacent targets to actually be close; a
# sparse target set gives group extents approaching the cloud size and a beam that opens everything.
COLUMN_GROUP_MIN_TARGETS = 1000


@njit(fastmath=True)
def acceptance_criterion(r: float, h: float, size: float, delta: float, theta: float) -> bool:
    """Criterion for accepting the multipole approximation for summing the contribution of a node"""
    return r > max(size / theta + delta, h + size * 0.6 + delta)


@njit(inline="always")
def subtree_end(tree, no):
    """Index at which a walk started at ``no`` must stop to stay inside no's subtree.

    Depth-first order resumes at ``NextBranch[no]`` once the subtree is done; without this bound the
    chain leads back out and keeps going.  -1 at the root, so the default whole-tree walk is unaffected.
    """
    return tree.NextBranch[no]


@njit(int64(float64, float64, float64), inline="always", fastmath=True)
def angular_bin_scalar(dx0, dx1, dx2):
    """Angular bin of a separation, taken as scalars so grouped kernels can call it in their target
    loop without materializing a length-3 array -- the allocation that made grouped_treewalk's
    quadrupole kernel anti-scale."""
    if fabs(dx0) > fabs(dx1) and fabs(dx0) > fabs(dx2):
        return 0 if dx0 > 0 else 1
    elif fabs(dx1) > fabs(dx2):
        return 2 if dx1 > 0 else 3
    else:
        return 4 if dx2 > 0 else 5


@njit([int64(float64[:])], fastmath=True)
def angular_bin(dx):
    """Angular bin for binned column density estimator"""
    if fabs(dx[0]) > fabs(dx[1]) and fabs(dx[0]) > fabs(dx[2]):
        if dx[0] > 0:
            bin = 0
        else:
            bin = 1
    elif fabs(dx[1]) > fabs(dx[2]):
        if dx[1] > 0:
            bin = 2
        else:
            bin = 3
    else:
        if dx[2] > 0:
            bin = 4
        else:
            bin = 5
    return bin


@njit(fastmath=True)
def NearestImage(x, boxsize):
    """Nearest periodic image of a 1D separation x in a box of side boxsize."""
    if abs(x) > boxsize / 2:
        return -copysign(boxsize - abs(x), x)
    else:
        return x


@njit(fastmath=True)
def PotentialWalk(pos, tree, softening=0, no=-1, theta=0.7):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~0.2% RMS acceleration error on a Plummer sphere)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)
    phi = 0
    dx = np.empty(3, dtype=np.float64)

    while no > -1 and no != stop:
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
        elif acceptance_criterion(
            r, h, tree.Sizes[no], tree.Deltas[no], theta
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
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~0.2% RMS acceleration error on a Plummer sphere)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)
    phi = 0
    dx = np.empty(3, dtype=np.float64)

    while no > -1 and no != stop:
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
        elif acceptance_criterion(r, h, tree.Sizes[no], tree.Deltas[no], theta):
            # if we satisfy the criteria for accepting the monopole
            phi -= tree.Masses[no] / r
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
def AccelWalk(pos, tree, softening=0, no=-1, theta=0.7):
    """Returns the gravitational acceleration field at position x by performing the Barnes-Hut treewalk using the provided octree instance
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree instance storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~0.2% RMS acceleration error on a Plummer sphere)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)
    g = zeros(3, dtype=np.float64)
    dx = np.empty(3, dtype=np.float64)

    while no > -1 and no != stop:  # loop until we get to the end of the tree
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
                    # fac stores the quantity M(<R)/R^3 to be used later for force computation
                    fac = tree.Masses[no] * ForceKernel(r, h)
                else:  # use point mass force
                    fac = tree.Masses[no] / (r * r2)
                sum_field = True
            no = tree.NextBranch[no]
        elif acceptance_criterion(r, h, tree.Sizes[no], tree.Deltas[no], theta):
            # if we satisfy the criteria for accepting the monopole
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
def AccelWalk_quad(pos, tree, softening=0, no=-1, theta=0.7):  # ,include_self_potential=False):
    """Returns the gravitational acceleration field at position x by performing the Barnes-Hut treewalk using the provided octree instance. Uses the quadrupole expansion.
    Arguments:
    pos - (3,) array containing position of interest
    tree - octree instance storing the tree structure
    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~0.2% RMS acceleration error on a Plummer sphere)
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)
    g = zeros(3, dtype=np.float64)
    dx = np.empty(3, dtype=np.float64)

    while no > -1 and no != stop:  # loop until we get to the end of the tree
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        h = max(tree.Softenings[no], softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            if r > 0:  # no self-force
                if r < h:  # within the softening radius
                    # fac stores the quantity M(<R)/R^3 to be used later for force computation
                    fac = tree.Masses[no] * ForceKernel(r, h)
                else:  # use point mass force
                    fac = tree.Masses[no] / (r * r2)
            for k in range(3):
                g[k] += fac * dx[k]  # monopole
            no = tree.NextBranch[no]
            continue
        elif acceptance_criterion(r, h, tree.Sizes[no], tree.Deltas[no], theta):
            # if we satisfy the criteria for accepting the multipole expansion
            fac = tree.Masses[no] / (r * r2)
            quad = tree.Quadrupoles[no]
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


def PotentialTarget_tree(pos_target, softening_target, tree, G=1.0, theta=0.7, quadrupole=False):
    """Returns the gravitational potential at the specified points, given a tree containing the mass distribution
    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the potential
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all potential computations
    tree -- Octree instance containing the positions, masses, and softenings of the source particles
    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)
    quadrupole -- if True, include node quadrupole moments in the multipole expansion; requires a tree built with quadrupole=True (default False)
    Returns:
    shape (N,) array of potential values at each point in pos
    """
    result = empty(pos_target.shape[0])
    # scope the chunksize instead of setting it globally: the global form leaks process-wide
    # and halves the throughput of every unrelated prange afterwards, including in user code
    with parallel_chunksize(10000):
        if quadrupole:
            for i in prange(pos_target.shape[0]):
                result[i] = G * PotentialWalk_quad(pos_target[i], tree, softening=softening_target[i], theta=theta)
        else:
            for i in prange(pos_target.shape[0]):
                result[i] = G * PotentialWalk(pos_target[i], tree, softening=softening_target[i], theta=theta)
    return result


# JIT this function and its parallel version
PotentialTarget_tree_parallel = njit(PotentialTarget_tree, fastmath=True, parallel=True)
PotentialTarget_tree = njit(PotentialTarget_tree, fastmath=True)


def AccelTarget_tree(pos_target, softening_target, tree, G=1.0, theta=0.7, quadrupole=False):
    """Returns the gravitational acceleration at the specified points, given a tree containing the mass distribution
    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the field
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all accel computations
    tree -- Octree instance containing the positions, masses, and softenings of the source particles
    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)
    quadrupole -- if True, include node quadrupole moments in the multipole expansion; requires a tree built with quadrupole=True (default False)
    Returns:
    shape (N,3) array of acceleration values at each point in pos_target
    """
    if softening_target is None:
        softening_target = zeros(pos_target.shape[0])
    result = empty(pos_target.shape)
    # scope the chunksize instead of setting it globally: the global form leaks process-wide
    # and halves the throughput of every unrelated prange afterwards, including in user code
    with parallel_chunksize(10000):
        if quadrupole:
            for i in prange(pos_target.shape[0]):
                result[i] = G * AccelWalk_quad(pos_target[i], tree, softening=softening_target[i], theta=theta)
        else:
            for i in prange(pos_target.shape[0]):
                result[i] = G * AccelWalk(pos_target[i], tree, softening=softening_target[i], theta=theta)
    return result


# JIT this function and its parallel version
AccelTarget_tree_parallel = njit(AccelTarget_tree, fastmath=True, parallel=True)
AccelTarget_tree = njit(AccelTarget_tree, fastmath=True)


@njit(fastmath=True)
def do_weighted_binning(tree, no, rbins, mbin, r, r_idx, quantity):
    """(experimental) Spread a node's contribution across the radial bins its extent overlaps,
    weighted by the overlap fraction, instead of assigning it all to the bin containing r."""
    h = 0.5 * tree.Sizes[no]
    Nbins = rbins.shape[0] - 1
    if (r + h < rbins[r_idx + 1]) and (r - h > rbins[r_idx]):
        mbin[r_idx] += tree.Masses[no] * quantity
    else:
        min_bin = int((np.log10((r - h) / rbins[0]) / np.log10(rbins[1] / rbins[0])))
        max_bin = min(int(np.log10((r + h) / rbins[0]) / np.log10(rbins[1] / rbins[0]) + 1), Nbins)
        total_wt = 0
        for i in range(min_bin, max_bin):  # range(min_bin,max_bin): # first the prepass to get the total weight
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
    """Total mass in each radial bin around ``pos``, by Barnes-Hut treewalk.

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure
    rbins - 1D array of radial bin edges; must be logarithmically spaced

    Keyword arguments:
    max_bin_size_ratio - controls binning accuracy: nodes are opened until their side length is at most this factor times the radial bin width (default 100)
    no - index of the top-level node to start the walk from; defaults to the global root, can be a subnode in principle for e.g. parallelization
    theta - cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    boxsize - finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning - (experimental) if True, distribute mass among radial bins with a weighted kernel instead of assigning each node to a single bin (default False)

    Returns:
    shape (len(rbins)-1,) array of total mass in each radial bin
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)

    Nbins = rbins.shape[0] - 1
    mbin = zeros(Nbins)
    counts = zeros(Nbins)
    rmin = rbins[0]
    rmax = rbins[-1]
    dx = np.empty(3, dtype=np.float64)

    logr_min = np.log10(rmin)
    logr_max = np.log10(rmax)
    dlogr = logr_max - logr_min

    while no > -1 and no != stop:
        r = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            if boxsize > 0:
                dx[k] = NearestImage(dx[k], boxsize)
            r += dx[k] * dx[k]

        r = sqrt(r)
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
            ) and (tree.Sizes[no] < max_bin_size_ratio * (rbins[r_idx + 1] - rbins[r_idx])):
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
    """Average mass in each radial bin around a point, averaged over all points.

    Arguments:
    pos -- shape (N,3) array of particle positions
    tree -- Octree instance containing the positions, masses and softenings of the sources
    rbins -- 1D array of radial bin edges; must be logarithmically spaced

    Optional arguments:
    max_bin_size_ratio -- controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta -- cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    boxsize -- finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning -- (experimental) if True, distribute mass among radial bins with a weighted kernel instead of assigning each node to a single bin (default False)

    Returns:
    shape (len(rbins)-1,) array of mean mass per radial bin, averaged over all points
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
    """Weight-summed velocity correlation in each radial bin around ``pos``, by Barnes-Hut treewalk.

    Arguments:
    pos - (3,) array containing position of interest
    vel - (3,) array containing velocity of the point of interest
    tree - octree object storing the tree structure
    rbins - 1D array of radial bin edges; must be logarithmically spaced

    Keyword arguments:
    max_bin_size_ratio - controls binning accuracy: nodes are opened until their side length is at most this factor times the radial bin width (default 100)
    no - index of the top-level node to start the walk from; defaults to the global root, can be a subnode in principle for e.g. parallelization
    theta - cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    boxsize - finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning - (experimental) if True, distribute mass among radial bins with a weighted kernel instead of assigning each node to a single bin (default False)

    Returns:
    (wtsums, binsums) - per-bin weight sums and weighted v . v' sums; the caller divides one by the
    other to form the correlation function
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)

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

    while no > -1 and no != stop:
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
            ) and tree.Sizes[no] < max_bin_size_ratio * (rbins[r_idx + 1] - rbins[r_idx]):
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
    """Mass-weighted velocity correlation function of v . v' in radial bins, averaged over all points.

    Arguments:
    pos -- shape (N,3) array of particle positions
    vel -- shape (N,3) array of particle velocities
    weight -- shape (N,) array of per-particle weights (e.g. masses)
    tree -- DynamicOctree instance containing the source positions, weights and velocities
    rbins -- 1D array of radial bin edges; must be logarithmically spaced

    Optional arguments:
    max_bin_size_ratio -- controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta -- cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    boxsize -- finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning -- (experimental) if True, distribute mass among radial bins with a weighted kernel instead of assigning each node to a single bin (default False)

    Returns:
    shape (len(rbins)-1,) array of the weighted mean of v.v' in each radial bin
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
VelocityCorrFunc_tree_parallel = njit(VelocityCorrFunc_tree, fastmath=True, parallel=True)
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
    """Weight-summed velocity structure function in each radial bin around ``pos``.

    Arguments:
    pos - (3,) array containing position of interest
    vel - (3,) array containing velocity of the point of interest
    tree - octree object storing the tree structure
    rbins - 1D array of radial bin edges; must be logarithmically spaced

    Keyword arguments:
    max_bin_size_ratio - controls binning accuracy: nodes are opened until their side length is at most this factor times the radial bin width (default 100)
    no - index of the top-level node to start the walk from; defaults to the global root, can be a subnode in principle for e.g. parallelization
    theta - cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    boxsize - finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning - (experimental) if True, distribute mass among radial bins with a weighted kernel instead of assigning each node to a single bin (default False)

    Returns:
    (wtsums, binsums) - per-bin weight sums and weighted (v - v')^2 sums; the caller divides one by
    the other to form the structure function
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)

    Nbins = rbins.shape[0] - 1
    binsums = zeros(Nbins)
    wtsums = zeros(Nbins)
    rmin = rbins[0]
    rmax = rbins[-1]
    dx = np.empty(3, dtype=np.float64)
    logr_min = np.log10(rmin)
    logr_max = np.log10(rmax)
    dlogr = logr_max - logr_min

    while no > -1 and no != stop:
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
                    vprod += (vel[k] - tree.Velocities[no][k]) * (vel[k] - tree.Velocities[no][k]) * tree.Masses[no]
                binsums[r_idx] += vprod
                wtsums[r_idx] += tree.Masses[no]
                no = tree.NextBranch[no]
            elif r > max(
                tree.Sizes[no] / theta + tree.Deltas[no],
                tree.Sizes[no] * 0.6 + tree.Deltas[no],
            ) and (tree.Sizes[no] < max_bin_size_ratio * (rbins[r_idx + 1] - rbins[r_idx])):
                vprod = 0
                for k in range(3):
                    vprod += (vel[k] - tree.Velocities[no][k]) * (vel[k] - tree.Velocities[no][k])
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
    """Mass-weighted velocity structure function of (v - v')^2 in radial bins, averaged over all points.

    Arguments:
    pos -- shape (N,3) array of particle positions
    vel -- shape (N,3) array of particle velocities
    weight -- shape (N,) array of per-particle weights (e.g. masses)
    tree -- DynamicOctree instance containing the source positions, weights and velocities
    rbins -- 1D array of radial bin edges; must be logarithmically spaced

    Optional arguments:
    max_bin_size_ratio -- controls the accuracy of the binning - tree nodes are subdivided until their side length is at most this factor * the radial bin width (default 100)
    theta -- cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    boxsize -- finite periodic box size, if periodic boundary conditions are to be used (default 0)
    weighted_binning -- (experimental) if True, distribute mass among radial bins with a weighted kernel instead of assigning each node to a single bin (default False)

    Returns:
    shape (len(rbins)-1,) array of the weighted mean of (v - v')^2 in each radial bin
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
VelocityStructFunc_tree_parallel = njit(VelocityStructFunc_tree, fastmath=True, parallel=True)
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
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)

    N_rays = rays.shape[0]
    columns = np.zeros(N_rays)
    dx = np.empty(3, dtype=np.float64)
    z_ray = np.zeros(N_rays)  # perpendicular distances of elements to nearest point on rays

    fac_density = 3 / (4 * np.pi)

    while no > -1 and no != stop:
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        for i in range(N_rays):
            z_ray[i] = rays[i, 0] * dx[0] + rays[i, 1] * dx[1] + rays[i, 2] * dx[2]
        h_no = tree.Softenings[no]
        h = h_no  # max(h_no,softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            # A zero-radius particle has no cross-section, so it obscures nothing.  Guard the
            # reciprocal here, not above the branch: taking it for every visited element made a
            # single zero radius raise ZeroDivisionError serially and yield silent NaNs in parallel.
            if h_no > 0:
                h_no_inv = 1.0 / h_no
                fac = fac_density * tree.Masses[no] * h_no_inv * h_no_inv  # assumes uniform sphere geometry
                for i in range(N_rays):
                    r_proj = r2 - z_ray[i] * z_ray[i]
                    if r_proj < 0:
                        continue
                    r_proj = sqrt(r2 - z_ray[i] * z_ray[i])
                    q = r_proj * h_no_inv
                    if r_proj < h_no:
                        if r > h_no:  # not overlapping the target point - integrate the whole cell
                            if z_ray[i] < 0:
                                continue  # not on the ray
                            columns[i] += fac * 2 * sqrt(1 - q * q)
                        else:  # overlapping, so need to integrate only a portion of the cell - this case includes the self-shielding if the point is in the tree!
                            dz = z_ray[i] * h_no_inv
                            columns[i] += fac * (dz + sqrt(1 - q * q))

            no = tree.NextBranch[no]

        else:  # we have a node, need to check if it intersects a ray
            node_intersects_ray = False
            R_eff = tree.Sizes[no] * 0.8660254037844386 + tree.Deltas[no]  # effective search radius from center of mass
            for i in range(N_rays):
                if r < h + R_eff:  # if node contains the origin then it must intersect all rays
                    node_intersects_ray = True
                    break
                elif (z_ray[i] > 0) and (
                    (r2 - z_ray[i] * z_ray[i]) < (tree.Softenings[no] + R_eff) * (tree.Softenings[no] + R_eff)
                ):  # if perpendicular distance is less than node effective size
                    node_intersects_ray = True
                    break

            if node_intersects_ray:
                no = tree.FirstSubnode[no]  # open the node
            else:
                no = tree.NextBranch[no]  # no intersection with any way, so go to next node

    return columns


@njit(fastmath=True)
def ColumnDensityWalk_singleray(pos, ray, tree, no=-1):
    """Returns the integrated column density to infinity from pos, in the directions given by the rays argument

    Arguments:
    pos - (3,) array containing position of interest
    ray - (3,) array with the unit vector of the ray
    tree - octree object storing the tree structure

    Returns:
    columns - (N_rays,) array of column densities along directions given by rays

    Keyword arguments:
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)

    column = 0
    dx = np.empty(3, dtype=np.float64)
    z_ray = 0  # perpendicular distances of elements to nearest point on rays
    fac_density = 3 / (4 * np.pi)

    while no > -1 and no != stop:
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]
        r = sqrt(r2)
        z_ray = ray[0] * dx[0] + ray[1] * dx[1] + ray[2] * dx[2]
        if r2 - z_ray * z_ray < 0:
            no = tree.NextBranch[no]
            continue
        h_no = tree.Softenings[no]
        h = h_no  # max(h_no,softening)

        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            if h_no > 0:  # zero radius -> no cross-section; see ColumnDensityWalk_multiray
                h_no_inv = 1.0 / h_no
                fac = fac_density * tree.Masses[no] * h_no_inv * h_no_inv
                # assumes uniform sphere geometry
                r_proj = sqrt(r2 - z_ray * z_ray)
                q = r_proj * h_no_inv
                if r_proj < h_no:
                    if r > h_no:  # not overlapping the target point - integrate the whole cell
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
                (r2 - z_ray * z_ray) < (tree.Softenings[no] + R_eff) * (tree.Softenings[no] + R_eff)
            ):  # if perpendicular distance is less than node effective size
                node_intersects_ray = True

            if node_intersects_ray:
                no = tree.FirstSubnode[no]  # open the node
            else:  # no intersection with any way, so go to next node
                no = tree.NextBranch[no]
    return column


@njit(fastmath=True)
def ColumnDensityWalk_binned(pos, tree, theta=0.5, no=-1):
    """Returns the integrated column density to infinity from pos, in the directions given by the rays argument

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure

    Returns:
    columns - shape (6,) array of average column densities in the 6 equal bins on the sphere

    Keyword arguments:
    no - root of the subtree to sum over; defaults to the whole tree. Summing over a node's children reproduces that node's own result, e.g. for parallelization
    """
    if no < 0:
        no = tree.NumParticles  # we default to the top-level node index
    stop = subtree_end(tree, no)

    n_bins = 6
    column = np.zeros(n_bins)
    dx = np.empty(3, dtype=np.float64)
    angular_bin_size = (4 * np.pi) / n_bins

    while no > -1 and no != stop:
        r2 = 0
        for k in range(3):
            dx[k] = tree.Coordinates[no, k] - pos[k]
            r2 += dx[k] * dx[k]

        h_no = tree.Softenings[no]
        h = h_no
        if no < tree.NumParticles:  # if we're looking at a leaf/particle
            # add the particle's column if it's in the right direction
            bin = angular_bin(dx)
            if r2 > h * h:  # no overlap: a point mass at distance sqrt(r2)
                column[bin] += tree.Masses[no] / r2 / angular_bin_size
            elif h > 0:  # interpolate between full overlap case and no overlap
                col0 = tree.Masses[no] * (3 / (4 * np.pi * h * h))
                fac = sqrt(r2) / h  # 0 to 1 when there is overlap
                col_isotropic = (1 - fac) * col0
                for k in range(n_bins):
                    column[k] += col_isotropic
                column[bin] += col0 * fac * 2
            # else h == 0 and r2 == 0: a point mass exactly at the target has no finite column, and
            # the interpolation below would divide by zero.  Contribute nothing.
            no = tree.NextBranch[no]
        elif acceptance_criterion(
            sqrt(r2), h, tree.Sizes[no], tree.Deltas[no], theta
        ):  # we can put the whole node in a bin
            column[angular_bin(dx)] += tree.Masses[no] / r2 / angular_bin_size
            no = tree.NextBranch[no]
        else:
            no = tree.FirstSubnode[no]
    return column


def ColumnDensity_tree(pos_target, tree, rays=None, randomize_rays=False, theta=0.7):
    """Returns the column density integrated to infinity from pos_target along rays, given the mass distribution in an Octree

    Parameters
    ----------
    pos_target: array_like
        shape (N,3) array of target particle positions where you want to know the potential.
    tree: Octree
        Octree instance initialized with the positions, masses, and softenings of the source particles.
    rays: array_like
        Shape (N_rays,3) array of ray direction unit vectors. If None then we instead compute average column densities in a 6-bin tesselation of the sphere.
    randomize_rays: bool, optional
        Randomly orients the raygrid for each particle.

    theta - cell opening angle; smaller is slower (runtime ~ theta^-3) but more accurate (default 0.7)
    """
    # Scoped, not global -- see the note in PotentialTarget_tree.  64 rather than 10000: a coarse
    # chunk collapses concurrency.  At N=3e5 with 12 rays both spend ~100 s of CPU, but chunksize
    # 10000 takes 43.6 s of wall against 10.5 s (cpu/wall 2.10 vs 9.05 on 16 threads).  Anything from
    # 8 to 2048 is equally good and bit-identical, so this only ever costs scheduling.
    with parallel_chunksize(64):
        if rays is None:  # do angular-binned column density
            result = empty((pos_target.shape[0], 6))
            for i in prange(pos_target.shape[0]):
                result[i] = ColumnDensityWalk_binned(pos_target[i], tree, theta)
        elif randomize_rays and rays.shape[0] <= MULTIRAY_MAX_RAYS:
            # Small bundle: one traversal for the whole grid amortizes the descent across the rays.
            result = empty((pos_target.shape[0], len(rays)))
            for i in prange(pos_target.shape[0]):
                rays_random = rays @ random_rotation(i)
                result[i] = ColumnDensityWalk_multiray(pos_target[i], rays_random, tree)
        elif randomize_rays:
            # Large bundle: the multiray walk opens the *union* of every ray's node set and then
            # evaluates each element against all N_rays rays, so it costs O(N_rays^2 * N^(1/3)),
            # while independent single-ray walks cost O(N_rays * N^(1/3)).  Rays sharing an origin
            # diverge past the top few levels, so the union really does grow with N_rays.  Both
            # walks are conservative and test each leaf per-ray, so they agree exactly (measured:
            # zero disagreement) -- this is purely a cost decision.  See MULTIRAY_MAX_RAYS.
            result = empty((pos_target.shape[0], len(rays)))
            for i in prange(pos_target.shape[0]):
                rays_random = rays @ random_rotation(i)
                for j in range(rays_random.shape[0]):
                    result[i, j] = ColumnDensityWalk_singleray(pos_target[i], rays_random[j], tree)
        else:
            result = empty((pos_target.shape[0], len(rays)))
            for i in range(rays.shape[0]):
                # outer loop over rays - empirically better access pattern
                for j in prange(pos_target.shape[0]):
                    result[j, i] = ColumnDensityWalk_singleray(pos_target[j], rays[i], tree)
    return result


ColumnDensity_tree_parallel = njit(ColumnDensity_tree, fastmath=True, parallel=True)
ColumnDensity_tree = njit(ColumnDensity_tree, fastmath=True)


def ColumnDensity_grouped(pos_target, rays, tree, group_size=16):
    """Ray-traced column density, traversing the tree once per (group of targets, ray direction).

    The amortization grouped_treewalk applies to gravity.  Morton-order targets make each group
    spatially compact, so for a fixed direction their rays are parallel lines separated by at most the
    group's extent; inflating the acceptance reach by the group bbox half-diagonal ``E`` therefore
    opens a superset of what any individual ray would.  Accepted leaves then loop over the group with
    the leaf's mass and reciprocal radius hoisted into registers, trading G gather-driven traversals
    for one traversal plus G register-only tests.

    Output is bit-identical at every group_size -- the superset only adds leaves contributing nothing,
    and each target accumulates in the same depth-first order.  Measured against group_size=1 at
    12 rays: 2.80x at N=3e5, 2.5x at N=3e4; group_size=16 best at both.

    Not usable with randomize_rays, which gives each target its own rotated grid; grouping needs a
    shared one.

    Arguments:
    pos_target -- shape (N,3) target positions, in Morton order for grouping to be effective
    rays -- shape (N_rays,3) array of unit ray direction vectors
    tree -- Octree holding the source mass distribution
    group_size -- targets per traversal (default 16); 1 reproduces the per-target walk

    Returns:
    shape (N, N_rays) array of column densities, in the same order as pos_target
    """
    N = pos_target.shape[0]
    n_rays = rays.shape[0]
    out = np.zeros((N, n_rays))
    ngroups = (N + group_size - 1) // group_size
    fac_density = 3 / (4 * np.pi)
    # 4, not ColumnDensity_tree's 64: there are group_size times fewer groups than targets, so keeping
    # chunks-per-thread up needs a smaller chunk.  At N=3e4, group_size=16, chunksize 64 leaves 29
    # chunks for 16 threads and hits the same concurrency cliff -- 758 ms against 144 ms at 4.
    with parallel_chunksize(4):
        for gi in prange(ngroups):
            a = gi * group_size
            b = min(a + group_size, N)
            # group bounding box -> centre, and half-diagonal as a conservative angular padding
            bmin0 = bmax0 = pos_target[a, 0]
            bmin1 = bmax1 = pos_target[a, 1]
            bmin2 = bmax2 = pos_target[a, 2]
            for t in range(a + 1, b):
                if pos_target[t, 0] < bmin0:
                    bmin0 = pos_target[t, 0]
                elif pos_target[t, 0] > bmax0:
                    bmax0 = pos_target[t, 0]
                if pos_target[t, 1] < bmin1:
                    bmin1 = pos_target[t, 1]
                elif pos_target[t, 1] > bmax1:
                    bmax1 = pos_target[t, 1]
                if pos_target[t, 2] < bmin2:
                    bmin2 = pos_target[t, 2]
                elif pos_target[t, 2] > bmax2:
                    bmax2 = pos_target[t, 2]
            cx = 0.5 * (bmin0 + bmax0)
            cy = 0.5 * (bmin1 + bmax1)
            cz = 0.5 * (bmin2 + bmax2)
            ex = 0.5 * (bmax0 - bmin0)
            ey = 0.5 * (bmax1 - bmin1)
            ez = 0.5 * (bmax2 - bmin2)
            E = sqrt(ex * ex + ey * ey + ez * ez)

            for i in range(n_rays):
                rx = rays[i, 0]
                ry = rays[i, 1]
                rz = rays[i, 2]
                no = tree.NumParticles
                stop = tree.NextBranch[no]
                while no > -1 and no != stop:
                    gx = tree.Coordinates[no, 0]
                    gy = tree.Coordinates[no, 1]
                    gz = tree.Coordinates[no, 2]
                    dx = gx - cx
                    dy = gy - cy
                    dz = gz - cz
                    r2 = dx * dx + dy * dy + dz * dz
                    z = rx * dx + ry * dy + rz * dz
                    rp2 = r2 - z * z
                    h_no = tree.Softenings[no]
                    if no < tree.NumParticles:  # leaf/particle
                        reach = h_no + E
                        # h_no > 0 first: cheapest reject, and zero radius contributes nothing
                        if h_no > 0 and (r2 < reach * reach or (z + E >= 0.0 and rp2 < reach * reach)):
                            h_inv = 1.0 / h_no  # hoisted: keeps the target loop allocation-free
                            fac = fac_density * tree.Masses[no] * h_inv * h_inv
                            hh = h_no * h_no
                            for t in range(a, b):
                                ddx = gx - pos_target[t, 0]
                                ddy = gy - pos_target[t, 1]
                                ddz = gz - pos_target[t, 2]
                                rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                                zz = rx * ddx + ry * ddy + rz * ddz
                                pp2 = rr2 - zz * zz
                                if pp2 < 0:
                                    continue
                                if pp2 < hh:
                                    q = sqrt(pp2) * h_inv
                                    chord = sqrt(1 - q * q)
                                    if rr2 > hh:  # target outside the sphere: whole chord
                                        if zz > 0:  # and the sphere is ahead of the target
                                            out[t, i] += fac * 2 * chord
                                    else:  # target inside: only the forward half-chord
                                        out[t, i] += fac * (zz * h_inv + chord)
                        no = tree.NextBranch[no]
                    else:  # node: open it if it could intersect any of the group's rays
                        reach = h_no + tree.Sizes[no] * 0.8660254037844386 + tree.Deltas[no] + E
                        if r2 < reach * reach or (z + E >= 0.0 and rp2 < reach * reach):
                            no = tree.FirstSubnode[no]
                        else:
                            no = tree.NextBranch[no]
    return out


ColumnDensity_grouped_parallel = njit(ColumnDensity_grouped, fastmath=True, parallel=True)
ColumnDensity_grouped = njit(ColumnDensity_grouped, fastmath=True)


def ColumnDensityBinned_grouped(pos_target, tree, theta=0.5, group_size=16):
    """Angular-binned (rays=None) column density, one traversal per group of Morton-adjacent targets.

    Structurally grouped_treewalk's gravity walk: acceptance uses ``r_min``, the nearest distance from
    the node to the group's bounding box (0 if inside).  That is at most any member's own distance, so
    the group opens a superset of what any individual target would -- conservative.

    Unlike the ray-traced grouped walk this is **not** bit-identical, because nodes themselves
    contribute: a finer node set splits each node's mass across the sky bins differently.  That is an
    improvement.  On the glass sphere, whose central column is exactly rho*R in every direction, the
    mean/truth goes 0.9401 -> 0.9426 and the bin spread 0.792-1.095 -> 0.870-1.046 from per-target to
    group_size=16 -- the dominant error nearly halved, alongside a 2.3x speedup.  group_size=1 makes
    r_min exact and recovers the per-target result.

    Arguments:
    pos_target -- shape (N,3) target positions, in Morton order for grouping to be effective
    tree -- Octree holding the source mass distribution
    theta -- opening angle (default 0.5)
    group_size -- targets per traversal (default 16); 1 reproduces the per-target walk

    Returns:
    shape (N, 6) array of mean column density per equal-area sky bin, in the same order as pos_target
    """
    N = pos_target.shape[0]
    n_bins = 6
    out = np.zeros((N, n_bins))
    angular_bin_size = (4 * np.pi) / n_bins
    ngroups = (N + group_size - 1) // group_size
    with parallel_chunksize(4):  # see the chunksize note in ColumnDensity_grouped
        for gi in prange(ngroups):
            a = gi * group_size
            b = min(a + group_size, N)
            bmin0 = bmax0 = pos_target[a, 0]
            bmin1 = bmax1 = pos_target[a, 1]
            bmin2 = bmax2 = pos_target[a, 2]
            for t in range(a + 1, b):
                if pos_target[t, 0] < bmin0:
                    bmin0 = pos_target[t, 0]
                elif pos_target[t, 0] > bmax0:
                    bmax0 = pos_target[t, 0]
                if pos_target[t, 1] < bmin1:
                    bmin1 = pos_target[t, 1]
                elif pos_target[t, 1] > bmax1:
                    bmax1 = pos_target[t, 1]
                if pos_target[t, 2] < bmin2:
                    bmin2 = pos_target[t, 2]
                elif pos_target[t, 2] > bmax2:
                    bmax2 = pos_target[t, 2]

            no = tree.NumParticles
            stop = tree.NextBranch[no]
            while no > -1 and no != stop:
                cx = tree.Coordinates[no, 0]
                cy = tree.Coordinates[no, 1]
                cz = tree.Coordinates[no, 2]
                h_no = tree.Softenings[no]
                M = tree.Masses[no]
                if no < tree.NumParticles:  # leaf: contributes to every target exactly, no criterion
                    hh = h_no * h_no
                    for t in range(a, b):
                        d0 = cx - pos_target[t, 0]
                        d1 = cy - pos_target[t, 1]
                        d2 = cz - pos_target[t, 2]
                        r2 = d0 * d0 + d1 * d1 + d2 * d2
                        bin = angular_bin_scalar(d0, d1, d2)
                        if r2 > hh:  # no overlap: a point mass at distance sqrt(r2)
                            out[t, bin] += M / r2 / angular_bin_size
                        elif h_no > 0:  # overlapping: blend the isotropic and directed limits
                            col0 = M * (3 / (4 * np.pi * hh))
                            fac = sqrt(r2) / h_no
                            col_isotropic = (1 - fac) * col0
                            for k in range(n_bins):
                                out[t, k] += col_isotropic
                            out[t, bin] += col0 * fac * 2
                        # else h_no == 0 and r2 == 0: point mass at the target, no finite column
                    no = tree.NextBranch[no]
                else:
                    # nearest distance from the node to the group's bounding box, 0 if inside
                    dxm = 0.0
                    if cx < bmin0:
                        dxm = bmin0 - cx
                    elif cx > bmax0:
                        dxm = cx - bmax0
                    dym = 0.0
                    if cy < bmin1:
                        dym = bmin1 - cy
                    elif cy > bmax1:
                        dym = cy - bmax1
                    dzm = 0.0
                    if cz < bmin2:
                        dzm = bmin2 - cz
                    elif cz > bmax2:
                        dzm = cz - bmax2
                    r_min = sqrt(dxm * dxm + dym * dym + dzm * dzm)
                    if acceptance_criterion(r_min, h_no, tree.Sizes[no], tree.Deltas[no], theta):
                        for t in range(a, b):
                            d0 = cx - pos_target[t, 0]
                            d1 = cy - pos_target[t, 1]
                            d2 = cz - pos_target[t, 2]
                            r2 = d0 * d0 + d1 * d1 + d2 * d2
                            out[t, angular_bin_scalar(d0, d1, d2)] += M / r2 / angular_bin_size
                        no = tree.NextBranch[no]
                    else:
                        no = tree.FirstSubnode[no]
    return out


ColumnDensityBinned_grouped_parallel = njit(ColumnDensityBinned_grouped, fastmath=True, parallel=True)
ColumnDensityBinned_grouped = njit(ColumnDensityBinned_grouped, fastmath=True)
