"""Prototype grouped/vectorized Barnes-Hut treewalk (monopole acceleration).

WORK IN PROGRESS - not yet wired into the frontend (Accel/AccelTarget). This is the single-kernel
prototype validated in benchmarking: ~2.2x faster than the per-particle walk on a 16M-particle
GIZMO snapshot and ~2.9-3.6x on a Plummer sphere, at equal-or-better accuracy for a given theta.

Idea: instead of walking the tree once per target particle, walk it once per *group* of spatially
compact targets (contiguous chunks of the Morton-sorted target array, which are compact for free).
Each group traverses the tree once; at every accepted element we inner-loop over the group's
targets - the dense, vectorizable kernel that provides the speedup. Group acceptance uses the
group bounding box's nearest distance (r_min) and the group's max softening, which is strictly more
conservative than the per-particle criterion, so the result opens a superset of nodes and accuracy
is preserved (equal-or-better) for a given theta.

To productionize: factor a shared grouped-traversal core and port the Potential/quadrupole/target
variants, then wire group_size (default 8 works well) through ConstructTree/Accel.
"""

import numpy as np
from numba import njit, prange, set_parallel_chunksize

from .kernel import ForceKernel
from .treewalk import acceptance_criterion


@njit(fastmath=True, parallel=True)
def grouped_accel(pos, soft, tree, group_size, theta=0.7, G=1.0):
    """Gravitational acceleration at pos via a grouped (vectorized) Barnes-Hut monopole walk.

    pos, soft must be in tree (Morton) order - i.e. pos = pos_source[tree.TreewalkIndices], as the
    frontend already arranges. Returns a (N, 3) array in that same order.
    """
    N = pos.shape[0]
    ngroups = (N + group_size - 1) // group_size
    result = np.zeros((N, 3))
    set_parallel_chunksize(64)
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

        g_local = np.zeros((m, 3))
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
            r_min = np.sqrt(dxm * dxm + dym * dym + dzm * dzm)
            h = max(tree.Softenings[no], hmax)

            interact = False
            if no < tree.NumParticles:  # leaf particle -> always a direct interaction
                interact = True
                nxt = tree.NextBranch[no]
            elif acceptance_criterion(r_min, h, tree.Sizes[no], tree.Deltas[no], theta):
                interact = True
                nxt = tree.NextBranch[no]
            else:
                no = tree.FirstSubnode[no]
                continue

            if interact:
                M = tree.Masses[no]
                hs = tree.Softenings[no]
                # dense inner loop over the group's targets (the vectorizable kernel)
                for t in range(a, b):
                    ddx = cx - pos[t, 0]
                    ddy = cy - pos[t, 1]
                    ddz = cz - pos[t, 2]
                    r2 = ddx * ddx + ddy * ddy + ddz * ddz
                    if r2 > 0:
                        r = np.sqrt(r2)
                        ht = max(hs, soft[t])
                        if r < ht:
                            fac = M * ForceKernel(r, ht)
                        else:
                            fac = M / (r * r2)
                        g_local[t - a, 0] += fac * ddx
                        g_local[t - a, 1] += fac * ddy
                        g_local[t - a, 2] += fac * ddz
            no = nxt

        for t in range(a, b):
            result[t, 0] = G * g_local[t - a, 0]
            result[t, 1] = G * g_local[t - a, 1]
            result[t, 2] = G * g_local[t - a, 2]
    return result
