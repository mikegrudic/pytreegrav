"""Grouped/vectorized Barnes-Hut treewalk: one traversal per spatially-compact group of targets.

One traversal per *group* of consecutive (Morton-sorted, hence spatially compact) targets rather than
per target; at every accepted element the kernel inner-loops over the group.  The win is *amortizing
the traversal*: the branchy pointer-chasing descent (a step costs ~3x a force-pair evaluation) runs
~group_size times fewer, which more than pays for the extra force-pairs grouping incurs.  It is NOT
SIMD -- the inner loop's branches and ForceKernel call keep it scalar, and fastmath moves runtime ~2%.

Acceptance uses the group bbox's nearest distance (r_min) and max softening, strictly more
conservative than per-particle, so a superset of nodes opens: accuracy is equal-or-better for a given
theta, at the cost of more force-pairs.  group_size=1 reproduces the per-particle walk; the optimum
(~8) is where amortization saturates before interaction-list bloat dominates.

The traversal core is shared; only the small ``inline='always'`` per-interaction kernel varies
(monopole/quadrupole x potential/acceleration), and numba inlines it, so sharing costs a few percent.
"""

import numpy as np
from math import sqrt
from numba import njit, prange, parallel_chunksize

from .kernel import ForceKernel, PotentialKernel
from .treewalk import acceptance_criterion
from .octree import _morton_keys, _radix_argsort


# --------------------------------------------------------------------------------------------------
# Per-interaction kernels.  Signature: (no, a, b, pos, soft, tree, acc) where [a, b) is the group's
# range in the Morton-sorted target arrays and acc is the group's (m, W) accumulator.  Each adds the
# contribution of source element ``no`` (a particle if no < NumParticles, else a node) to every
# target in the group.  Marked inline='always' so the shared core folds them in.
# --------------------------------------------------------------------------------------------------


@njit(inline="always", fastmath=True)
def _accel_mono(no, a, b, pos, soft, tree, acc):
    """Add source ``no``'s monopole (point-mass, softened) acceleration to every target in [a, b)."""
    cx = tree.Coordinates[no, 0]
    cy = tree.Coordinates[no, 1]
    cz = tree.Coordinates[no, 2]
    M = tree.Masses[no]
    hs = tree.Softenings[no]
    for t in range(a, b):
        dx = cx - pos[t, 0]
        dy = cy - pos[t, 1]
        dz = cz - pos[t, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > 0:
            r = sqrt(r2)
            ht = max(hs, soft[t])
            if r < ht:
                fac = M * ForceKernel(r, ht)
            else:
                fac = M / (r * r2)
            acc[t - a, 0] += fac * dx
            acc[t - a, 1] += fac * dy
            acc[t - a, 2] += fac * dz


@njit(inline="always", fastmath=True)
def _pot_mono(no, a, b, pos, soft, tree, acc):
    """Add source ``no``'s monopole (point-mass, softened) potential to every target in [a, b)."""
    cx = tree.Coordinates[no, 0]
    cy = tree.Coordinates[no, 1]
    cz = tree.Coordinates[no, 2]
    M = tree.Masses[no]
    hs = tree.Softenings[no]
    for t in range(a, b):
        dx = cx - pos[t, 0]
        dy = cy - pos[t, 1]
        dz = cz - pos[t, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > 0:
            r = sqrt(r2)
            ht = max(hs, soft[t])
            if r < ht:
                acc[t - a, 0] += M * PotentialKernel(r, ht)
            else:
                acc[t - a, 0] += -M / r


@njit(inline="always", fastmath=True)
def _accel_quad(no, a, b, pos, soft, tree, acc):
    """Add source ``no``'s acceleration to every target in [a, b), with the quadrupole term for nodes.

    Leaf particles carry no quadrupole moment, so they contribute only the softened monopole.

    Components and the separation vector are held in scalars: a ``tree.Quadrupoles[no]`` view or an
    ``np.empty(3)`` inside the target loop costs one NRT allocation per target, and the allocator
    contention made this kernel *anti-scale* -- slower on 8 threads than 1, ~30x monopole cost at 32.
    """
    cx = tree.Coordinates[no, 0]
    cy = tree.Coordinates[no, 1]
    cz = tree.Coordinates[no, 2]
    M = tree.Masses[no]
    hs = tree.Softenings[no]
    is_node = no >= tree.NumParticles  # only nodes carry quadrupole moments
    qxx = qxy = qxz = qyx = qyy = qyz = qzx = qzy = qzz = 0.0
    if is_node:  # loop-invariant: load once, not once per target
        qxx = tree.Quadrupoles[no, 0, 0]
        qxy = tree.Quadrupoles[no, 0, 1]
        qxz = tree.Quadrupoles[no, 0, 2]
        qyx = tree.Quadrupoles[no, 1, 0]
        qyy = tree.Quadrupoles[no, 1, 1]
        qyz = tree.Quadrupoles[no, 1, 2]
        qzx = tree.Quadrupoles[no, 2, 0]
        qzy = tree.Quadrupoles[no, 2, 1]
        qzz = tree.Quadrupoles[no, 2, 2]
    for t in range(a, b):
        dx = cx - pos[t, 0]
        dy = cy - pos[t, 1]
        dz = cz - pos[t, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > 0:
            r = sqrt(r2)
            ht = max(hs, soft[t])
            if r < ht:
                fac = M * ForceKernel(r, ht)
            else:
                fac = M / (r * r2)
            acc[t - a, 0] += fac * dx  # monopole
            acc[t - a, 1] += fac * dy
            acc[t - a, 2] += fac * dz
            if is_node:
                r5inv = 1.0 / (r2 * r2 * r)
                qdx = qxx * dx + qxy * dy + qxz * dz
                qdy = qyx * dx + qyy * dy + qyz * dz
                qdz = qzx * dx + qzy * dy + qzz * dz
                quad_fac = (dx * qdx + dy * qdy + dz * qdz) * r5inv / r2
                acc[t - a, 0] += 2.5 * quad_fac * dx - qdx * r5inv
                acc[t - a, 1] += 2.5 * quad_fac * dy - qdy * r5inv
                acc[t - a, 2] += 2.5 * quad_fac * dz - qdz * r5inv


@njit(inline="always", fastmath=True)
def _pot_quad(no, a, b, pos, soft, tree, acc):
    """Add source ``no``'s potential to every target in [a, b), with the quadrupole term for nodes.

    Leaf particles carry no quadrupole moment, so they contribute only the softened monopole; nodes
    add the quadrupole correction.  Components are hoisted out of the target loop for the same
    allocation reason as in :func:`_accel_quad`.
    """
    cx = tree.Coordinates[no, 0]
    cy = tree.Coordinates[no, 1]
    cz = tree.Coordinates[no, 2]
    M = tree.Masses[no]
    hs = tree.Softenings[no]
    is_node = no >= tree.NumParticles  # only nodes carry quadrupole moments
    qxx = qxy = qxz = qyx = qyy = qyz = qzx = qzy = qzz = 0.0
    if is_node:  # loop-invariant: load once, not once per target
        qxx = tree.Quadrupoles[no, 0, 0]
        qxy = tree.Quadrupoles[no, 0, 1]
        qxz = tree.Quadrupoles[no, 0, 2]
        qyx = tree.Quadrupoles[no, 1, 0]
        qyy = tree.Quadrupoles[no, 1, 1]
        qyz = tree.Quadrupoles[no, 1, 2]
        qzx = tree.Quadrupoles[no, 2, 0]
        qzy = tree.Quadrupoles[no, 2, 1]
        qzz = tree.Quadrupoles[no, 2, 2]
    for t in range(a, b):
        dx = cx - pos[t, 0]
        dy = cy - pos[t, 1]
        dz = cz - pos[t, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 > 0:
            r = sqrt(r2)
            ht = max(hs, soft[t])
            if is_node:
                acc[t - a, 0] += -M / r  # accepted node: r > h, point mass
                r5inv = 1.0 / (r * r2 * r2)
                sq = (
                    dx * (qxx * dx + qxy * dy + qxz * dz)
                    + dy * (qyx * dx + qyy * dy + qyz * dz)
                    + dz * (qzx * dx + qzy * dy + qzz * dz)
                )
                acc[t - a, 0] += -0.5 * sq * r5inv
            elif r < ht:
                acc[t - a, 0] += M * PotentialKernel(r, ht)
            else:
                acc[t - a, 0] += -M / r


# --------------------------------------------------------------------------------------------------
# Shared traversal core.  Factory specializes it per kernel so the kernel inlines.
# --------------------------------------------------------------------------------------------------


def _make_core(kernel, parallel):
    """Build a jitted grouped-walk core specialized to ``kernel`` (inlined) and ``parallel``.

    Returns a njit function core(pos, soft, tree, group_size, theta, G, W) -> (N, W) field array,
    where W is the output width (3 for acceleration, 1 for potential) and pos/soft are in tree
    (Morton) order.  Separate instances are built per kernel so numba inlines each kernel.
    """

    def core(pos, soft, tree, group_size, theta, G, W):
        """Grouped Barnes-Hut walk: traverse the tree once per group of ``group_size`` targets.

        For each group, computes its bounding box and max softening, descends the tree accepting or
        opening nodes by the group's nearest-corner distance (r_min) and max softening, and lets
        ``kernel`` accumulate each accepted element's contribution over the group's targets.  Returns
        G times the accumulated field as an (N, W) array in the input (Morton) order.
        """
        N = pos.shape[0]
        out = np.zeros((N, W))
        ngroups = (N + group_size - 1) // group_size
        # scoped, not global -- see the note in treewalk.PotentialTarget_tree
        with parallel_chunksize(64):
            for gi in prange(ngroups):
                a = gi * group_size
                b = min(a + group_size, N)
                m = b - a
                # group bounding box + max softening
                bmin0 = pos[a, 0]
                bmin1 = pos[a, 1]
                bmin2 = pos[a, 2]
                bmax0 = pos[a, 0]
                bmax1 = pos[a, 1]
                bmax2 = pos[a, 2]
                hmax = soft[a]
                for t in range(a + 1, b):
                    if pos[t, 0] < bmin0:
                        bmin0 = pos[t, 0]
                    elif pos[t, 0] > bmax0:
                        bmax0 = pos[t, 0]
                    if pos[t, 1] < bmin1:
                        bmin1 = pos[t, 1]
                    elif pos[t, 1] > bmax1:
                        bmax1 = pos[t, 1]
                    if pos[t, 2] < bmin2:
                        bmin2 = pos[t, 2]
                    elif pos[t, 2] > bmax2:
                        bmax2 = pos[t, 2]
                    if soft[t] > hmax:
                        hmax = soft[t]

                acc = np.zeros((m, W))
                no = tree.NumParticles  # root
                while no > -1:
                    cx = tree.Coordinates[no, 0]
                    cy = tree.Coordinates[no, 1]
                    cz = tree.Coordinates[no, 2]
                    # nearest distance from node position to the group's bbox (0 if inside)
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
                    h = max(tree.Softenings[no], hmax)
                    if no < tree.NumParticles or acceptance_criterion(r_min, h, tree.Sizes[no], tree.Deltas[no], theta):
                        kernel(no, a, b, pos, soft, tree, acc)
                        no = tree.NextBranch[no]
                    else:
                        no = tree.FirstSubnode[no]
                for t in range(a, b):
                    for k in range(W):
                        out[t, k] = G * acc[t - a, k]
        return out

    return njit(core, fastmath=True, parallel=parallel)


_accel_core = (_make_core(_accel_mono, False), _make_core(_accel_mono, True))
_accel_quad_core = (_make_core(_accel_quad, False), _make_core(_accel_quad, True))
_pot_core = (_make_core(_pot_mono, False), _make_core(_pot_mono, True))
_pot_quad_core = (_make_core(_pot_quad, False), _make_core(_pot_quad, True))


def _morton_order(points):
    """Permutation that sorts arbitrary points into a spatially-compact (Morton) order."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (hi + lo)
    size = float((hi - lo).max())
    return _radix_argsort(_morton_keys(np.ascontiguousarray(points), center, size))


# --------------------------------------------------------------------------------------------------
# Public wrappers.  pos/soft must already be in a spatially-compact order for grouping to help; the
# frontend feeds Morton order (self-gravity is pre-sorted, external targets are sorted via
# _morton_order).  group_size=1 reproduces the per-particle walk.
# --------------------------------------------------------------------------------------------------


def AccelTarget_grouped(pos, soft, tree, group_size=8, G=1.0, theta=0.7, quadrupole=False, parallel=True):
    """Gravitational acceleration at ``pos`` from ``tree``, via the grouped Barnes-Hut walk.

    Arguments:
    pos -- shape (N, 3) target positions, in tree (Morton) order for grouping to be effective
    soft -- shape (N,) minimum softening length of each target
    tree -- Octree containing the source mass distribution
    Keyword arguments:
    group_size -- targets per group (default 8); 1 reproduces the per-particle walk
    G -- gravitational constant (default 1.0)
    theta -- opening-angle accuracy parameter (default 0.7)
    quadrupole -- include quadrupole moments (default False); requires a tree built with quadrupole=True
    parallel -- parallelize over groups (default True)
    Returns:
    shape (N, 3) array of accelerations in the same order as ``pos``.
    """
    core = (_accel_quad_core if quadrupole else _accel_core)[1 if parallel else 0]
    return core(pos, soft, tree, group_size, theta, G, 3)


def PotentialTarget_grouped(pos, soft, tree, group_size=8, G=1.0, theta=0.7, quadrupole=False, parallel=True):
    """Gravitational potential at ``pos`` from ``tree``, via the grouped Barnes-Hut walk.

    Arguments and keywords match :func:`AccelTarget_grouped`.  Returns a shape (N,) array of
    potentials in the same order as ``pos``.
    """
    core = (_pot_quad_core if quadrupole else _pot_core)[1 if parallel else 0]
    return core(pos, soft, tree, group_size, theta, G, 1)[:, 0]
