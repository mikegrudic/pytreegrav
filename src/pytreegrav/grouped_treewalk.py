"""Grouped/vectorized Barnes-Hut treewalk: one traversal per spatially-compact group of targets.

One traversal per *group* of consecutive (Morton-sorted, hence spatially compact) targets rather than per target; at every accepted element the kernel inner-loops over the group.  The win is *amortizing the traversal*: the branchy pointer-chasing descent (a step costs ~3x a force-pair evaluation) runs ~group_size times fewer, which more than pays for the extra force-pairs grouping incurs.  It is NOT SIMD -- the inner loop's branches and ForceKernel call keep it scalar, and fastmath moves runtime ~2%.

Acceptance uses the group bbox's nearest distance (r_min) and max softening, strictly more conservative than per-particle, so a superset of nodes opens: accuracy is equal-or-better for a given theta, at the cost of more force-pairs.  group_size=1 reproduces the per-particle walk; the optimum (~8) is where amortization saturates before interaction-list bloat dominates.

The traversal core is shared; only the small ``inline='always'`` per-interaction kernel varies, and numba inlines it, so sharing costs a few percent.  That kernel is generated from a single parameterized source (:func:`_make_field_kernel`) specialized on which of potential/acceleration/tidal-tensor are wanted and whether quadrupoles are in play -- so any subset can be had from ONE traversal, and there is one copy of the arithmetic rather than one per field.
"""

import numpy as np
from math import sqrt
from numba import njit, prange, parallel_chunksize

from .kernel import ForceKernel, PotentialKernel, TidalKernel
from .treewalk import acceptance_criterion
from .octree import _morton_keys, _radix_argsort


# --------------------------------------------------------------------------------------------------
# Per-interaction kernel.  ONE parameterized source, specialized per requested field set.
#
# Signature: (no, a, b, pos, soft, tree, acc) where [a, b) is the group's range in the Morton-sorted
# target arrays and acc is the group's (m, W) accumulator.  Adds the contribution of source element
# ``no`` (a particle if no < NumParticles, else a node) to every target in the group.
#
# want_pot/want_accel/want_tidal/quadrupole are closure constants, not arguments, so numba folds the
# guards at compile time and the dead branches vanish: the code generated for accel-only contains no
# potential or tidal arithmetic at all.  Passing them as arguments instead would put an indirect call
# and live branches in the innermost loop.  Same reason the traversal core is built per kernel.
#
# The payoff for fusing is NOT traversal amortization -- grouping already did that -- but the shared
# per-interaction setup: one sqrt, one max(hs, soft[t]), one ForceKernel serving both accel and tidal,
# and one qdx/qdy/qdz/A contraction serving all three.  Measured on 1e6 Plummer particles at 32
# threads, theta=0.5, against the sum of the equivalent separate walks: potential+accel 1.54-1.70x,
# potential+tidal 1.31-1.32x, accel+tidal 1.36-1.41x, all three 1.64-1.65x.
#
# Replacing the six hand-written kernels this grew out of cost nothing: A/B against them at the same
# N, run in both orderings on an idle machine, put every single-field case within 2%, with the sign
# of the difference flipping between orderings -- i.e. noise.  Do not reintroduce a hand-written
# variant on the strength of a single timing run; an earlier prototype appeared to show a 5% loss on
# accel+quadrupole that turned out to be the machine, not the code.
# --------------------------------------------------------------------------------------------------


def _make_field_kernel(want_pot, want_accel, want_tidal, quadrupole):
    """Build the per-interaction kernel for one combination of requested fields.

    Output components are packed contiguously in the order potential, acceleration, tidal -- only the
    requested ones -- so the accumulator width is exactly ``1*want_pot + 3*want_accel + 9*want_tidal``.

    The tidal tensor is stored as all 9 components rather than the 6 independent ones: the duplicates
    cost 3 adds against a reshape-free (N,3,3) view, and only 6 products are actually formed.
    """

    @njit(inline="always", fastmath=True)
    def kernel(no, a, b, pos, soft, tree, acc):
        cx = tree.Coordinates[no, 0]
        cy = tree.Coordinates[no, 1]
        cz = tree.Coordinates[no, 2]
        M = tree.Masses[no]
        hs = tree.Softenings[no]
        # only nodes carry quadrupole moments; leaf particles contribute the softened monopole alone
        is_node = no >= tree.NumParticles if quadrupole else False
        qxx = qxy = qxz = qyx = qyy = qyz = qzx = qzy = qzz = 0.0
        if quadrupole and is_node:  # loop-invariant: load once, not once per target
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
            if r2 > 0:  # no self-interaction
                r = sqrt(r2)
                ht = max(hs, soft[t])
                softened = r < ht
                # K is M(<r)/r^3 and Q is -(1/r) dK/dr; both are shared where both are wanted
                phi = 0.0
                K = 0.0
                Q = 0.0
                if want_pot:
                    phi = M * PotentialKernel(r, ht) if softened else -M / r
                if want_accel or want_tidal:
                    K = M * ForceKernel(r, ht) if softened else M / (r * r2)
                if want_tidal:
                    Q = M * TidalKernel(r, ht) if softened else 3.0 * K / r2

                gx = gy = gz = 0.0
                if want_accel:
                    gx = K * dx
                    gy = K * dy
                    gz = K * dz
                txx = tyy = tzz = txy = txz = tyz = 0.0
                if want_tidal:
                    txx = Q * dx * dx - K
                    tyy = Q * dy * dy - K
                    tzz = Q * dz * dz - K
                    txy = Q * dx * dy
                    txz = Q * dx * dz
                    tyz = Q * dy * dz

                if quadrupole and is_node:
                    r5inv = 1.0 / (r2 * r2 * r)
                    r7inv = r5inv / r2
                    # the shared contraction: computed once, consumed by every requested field
                    qdx = qxx * dx + qxy * dy + qxz * dz
                    qdy = qyx * dx + qyy * dy + qyz * dz
                    qdz = qzx * dx + qzy * dy + qzz * dz
                    A = dx * qdx + dy * qdy + dz * qdz
                    if want_pot:
                        phi += -0.5 * A * r5inv
                    if want_accel:
                        quad_fac = A * r7inv
                        gx += 2.5 * quad_fac * dx - qdx * r5inv
                        gy += 2.5 * quad_fac * dy - qdy * r5inv
                        gz += 2.5 * quad_fac * dz - qdz * r5inv
                    if want_tidal:
                        # traceless by construction: -10 - (5/2)*3 + 35/2 = 0
                        diag = -2.5 * A * r7inv  # the delta_ij piece, common to all three diagonals
                        fac = 17.5 * A * r7inv / r2
                        txx += qxx * r5inv - 10.0 * dx * qdx * r7inv + diag + fac * dx * dx
                        tyy += qyy * r5inv - 10.0 * dy * qdy * r7inv + diag + fac * dy * dy
                        tzz += qzz * r5inv - 10.0 * dz * qdz * r7inv + diag + fac * dz * dz
                        txy += qxy * r5inv - 5.0 * (dx * qdy + dy * qdx) * r7inv + fac * dx * dy
                        txz += qxz * r5inv - 5.0 * (dx * qdz + dz * qdx) * r7inv + fac * dx * dz
                        tyz += qyz * r5inv - 5.0 * (dy * qdz + dz * qdy) * r7inv + fac * dy * dz

                o = 0
                if want_pot:
                    acc[t - a, 0] += phi
                    o = 1
                if want_accel:
                    acc[t - a, o] += gx
                    acc[t - a, o + 1] += gy
                    acc[t - a, o + 2] += gz
                    o += 3
                if want_tidal:
                    acc[t - a, o] += txx
                    acc[t - a, o + 1] += txy
                    acc[t - a, o + 2] += txz
                    acc[t - a, o + 3] += txy
                    acc[t - a, o + 4] += tyy
                    acc[t - a, o + 5] += tyz
                    acc[t - a, o + 6] += txz
                    acc[t - a, o + 7] += tyz
                    acc[t - a, o + 8] += tzz

    return kernel


# --------------------------------------------------------------------------------------------------
# Shared traversal core.  Factory specializes it per kernel so the kernel inlines.
# --------------------------------------------------------------------------------------------------


def _make_core(kernel, parallel):
    """Build a jitted grouped-walk core specialized to ``kernel`` (inlined) and ``parallel``.

    Returns a njit function core(pos, soft, tree, group_size, theta, G, W) -> (N, W) field array, where W is the output width (3 for acceleration, 1 for potential) and pos/soft are in tree (Morton) order.  Separate instances are built per kernel so numba inlines each kernel.
    """

    def core(pos, soft, tree, group_size, theta, G, W):
        """Grouped Barnes-Hut walk: traverse the tree once per group of ``group_size`` targets.

        For each group, computes its bounding box and max softening, descends the tree accepting or opening nodes by the group's nearest-corner distance (r_min) and max softening, and lets ``kernel`` accumulate each accepted element's contribution over the group's targets.  Returns G times the accumulated field as an (N, W) array in the input (Morton) order.
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


_CORE_CACHE = {}


def _get_core(want_pot, want_accel, want_tidal, quadrupole, parallel):
    """Jitted grouped-walk core for one field set, built on first use and memoized.

    Lazy rather than a table built at import: there are 2^3-1 field sets x quadrupole x parallel = 28
    specializations, and a session typically wants one or two.  ``njit`` itself is lazy, so holding
    dispatchers costs nothing -- but each *compilation* is real time, and these cores are deliberately
    not ``cache=True`` (see the note in bruteforce.py: both parallel variants would share one on-disk
    entry keyed by qualname and the parallel dispatcher would silently load the serial code).
    """
    key = (want_pot, want_accel, want_tidal, quadrupole, parallel)
    core = _CORE_CACHE.get(key)
    if core is None:
        core = _make_core(_make_field_kernel(want_pot, want_accel, want_tidal, quadrupole), parallel)
        _CORE_CACHE[key] = core
    return core


def _field_width(potential, accel, tidal):
    """Accumulator width for a field set, matching the kernel's packing order."""
    return 1 * potential + 3 * accel + 9 * tidal


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
    return FieldsTarget_grouped(
        pos,
        soft,
        tree,
        accel=True,
        group_size=group_size,
        G=G,
        theta=theta,
        quadrupole=quadrupole,
        parallel=parallel,
    )["accel"]


def PotentialTarget_grouped(pos, soft, tree, group_size=8, G=1.0, theta=0.7, quadrupole=False, parallel=True):
    """Gravitational potential at ``pos`` from ``tree``, via the grouped Barnes-Hut walk.

    Arguments and keywords match :func:`AccelTarget_grouped`.  Returns a shape (N,) array of potentials in the same order as ``pos``.
    """
    return FieldsTarget_grouped(
        pos,
        soft,
        tree,
        potential=True,
        group_size=group_size,
        G=G,
        theta=theta,
        quadrupole=quadrupole,
        parallel=parallel,
    )["potential"]


def TidalTensorTarget_grouped(pos, soft, tree, group_size=8, G=1.0, theta=0.7, quadrupole=False, parallel=True):
    """Tidal tensor at ``pos`` from ``tree``, via the grouped Barnes-Hut walk.

    Arguments and keywords match :func:`AccelTarget_grouped`.  Returns a shape (N,3,3) array of tidal tensors in the same order as ``pos``; the core accumulates them flattened, and the reshape is a view.
    """
    return FieldsTarget_grouped(
        pos,
        soft,
        tree,
        tidal=True,
        group_size=group_size,
        G=G,
        theta=theta,
        quadrupole=quadrupole,
        parallel=parallel,
    )["tidal"]


def FieldsTarget_grouped(
    pos,
    soft,
    tree,
    potential=False,
    accel=False,
    tidal=False,
    group_size=8,
    G=1.0,
    theta=0.7,
    quadrupole=False,
    parallel=True,
):
    """Any subset of {potential, accel, tidal} at ``pos`` from ``tree``, in a SINGLE traversal.

    Requesting several fields together is markedly cheaper than one walk each, because the per-interaction
    setup is shared -- see the note above :func:`_make_field_kernel` for the measured factors.  The three
    single-field wrappers above are thin calls into this.

    Arguments and keywords otherwise match :func:`AccelTarget_grouped`.

    One caveat, and it is the reason this is opt-in rather than automatic: a single traversal means a
    single acceptance criterion, hence one ``theta`` for every field requested.  The tidal tensor is a
    derivative higher than the potential and so wants a *smaller* theta for equal fractional accuracy, so
    a fused call runs everything at the strictest requirement.  If you want the potential at theta=0.7 and
    the tidal tensor at 0.4, two separate calls are genuinely cheaper.  Conversely, when the fields are
    wanted at one theta, fusing gives them a common accepted-node set and hence a mutually consistent
    truncation, which separate walks at different theta do not.

    Returns:
    dict with the requested keys: ``"potential"`` shape (N,), ``"accel"`` shape (N,3), ``"tidal"`` shape
    (N,3,3) -- each a view into one contiguous (N, W) buffer, in the same order as ``pos``.
    """
    if not (potential or accel or tidal):
        raise ValueError("request at least one of potential=True, accel=True, tidal=True")
    core = _get_core(potential, accel, tidal, quadrupole, parallel)
    out = core(pos, soft, tree, group_size, theta, G, _field_width(potential, accel, tidal))
    result = {}
    o = 0
    if potential:
        result["potential"] = out[:, 0]
        o = 1
    if accel:
        result["accel"] = out[:, o : o + 3]
        o += 3
    if tidal:
        result["tidal"] = out[:, o : o + 9].reshape(-1, 3, 3)
    return result
