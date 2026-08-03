"""Implementation of the Octree jitclass"""

import numpy as np
from numpy import zeros, ones, concatenate
from numba import float64, boolean, int64, njit, prange, get_num_threads
from numba.experimental import jitclass

spec = [
    ("Sizes", float64[:]),  # side length of tree nodes
    ("Deltas", float64[:]),  # distance between COM and geometric center of node
    # location of center of mass of node (actually stores _geometric_ center before we do the moments pass)
    ("Coordinates", float64[:, :]),
    ("Masses", float64[:]),  # total mass of node
    ("Quadrupoles", float64[:, :, :]),  # Quadrupole moment of the node
    # Allow us to quickly check if Quadrupole moments exist to keep monopole calculations fast
    ("HasQuads", boolean),
    ("NumParticles", int64),  # number of particles in the tree
    ("NumNodes", int64),  # number of particles + nodes (i.e. mass elements) in the tree
    # individual softenings for particles, _maximum_ softening of inhabitant particles for nodes
    ("Softenings", float64[:]),
    ("NextBranch", int64[:]),
    ("FirstSubnode", int64[:]),
    ("TreewalkIndices", int64[:]),
    # True if the build found positions coincident to floating-point precision. The build has to
    # detect these anyway to avoid subdividing forever, so this is free -- and exact, unlike
    # scanning the input up front.
    ("HasCoincidentPoints", boolean),
]


octant_offsets = 0.25 * np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ]
)

# Number of bits used per dimension in a Morton key.  3 * 21 = 63 fits in a signed int64.
MORTON_BITS = 21
MORTON_SCALE = float(1 << MORTON_BITS)  # 2**21
MORTON_MAXQ = (1 << MORTON_BITS) - 1  # largest quantized coordinate
# How many times we are willing to re-key a group of colliding points relative to their sub-cell.
# 3 rounds -> 3*21 = 63 bits/dim of subdivision, exceeding the 52-bit float64 mantissa, so any two
# points that differ in their floating-point representation are guaranteed to be separated.
MORTON_MAX_ROUNDS = 3


@njit(int64(int64), fastmath=False, cache=True)
def _spread_bits(x):
    """Spread the low 21 bits of x so that they occupy every third bit (bit i -> bit 3i).

    Standard libmorton magic-number method for interleaving a 21-bit coordinate into a 63-bit
    Morton key.
    """
    x &= 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


@njit(cache=True)
def _morton_keys(points, center, size):
    """Compute 63-bit Morton (Z-order) keys for points within the cube of given center and side.

    The cube spans [center - size/2, center + size/2] in each dimension.  Coordinates are quantized
    to MORTON_BITS bits per dimension and interleaved with x in the least-significant bit of each
    triplet, matching the octant convention used elsewhere (octant = (x>c) + 2*(y>c) + 4*(z>c)).
    """
    n = points.shape[0]
    keys = np.empty(n, dtype=np.int64)
    inv = 1.0 / size if size > 0 else 0.0
    cmin0 = center[0] - 0.5 * size
    cmin1 = center[1] - 0.5 * size
    cmin2 = center[2] - 0.5 * size
    for i in range(n):
        qx = int((points[i, 0] - cmin0) * inv * MORTON_SCALE)
        qy = int((points[i, 1] - cmin1) * inv * MORTON_SCALE)
        qz = int((points[i, 2] - cmin2) * inv * MORTON_SCALE)
        if qx < 0:
            qx = 0
        elif qx > MORTON_MAXQ:
            qx = MORTON_MAXQ
        if qy < 0:
            qy = 0
        elif qy > MORTON_MAXQ:
            qy = MORTON_MAXQ
        if qz < 0:
            qz = 0
        elif qz > MORTON_MAXQ:
            qz = MORTON_MAXQ
        keys[i] = _spread_bits(qx) | (_spread_bits(qy) << 1) | (_spread_bits(qz) << 2)
    return keys


@njit(cache=True)
def _radix_argsort(keys):
    """Stable LSD radix sort of nonnegative int64 keys, returning the sorting permutation.

    Carries the keys alongside the indices so every pass reads sequentially. Reading
    keys[idx[i]] instead (an indirect gather, twice per pass) measured 3.4x slower at N=1e7.
    11 bits per pass covers 63 bits in 6 passes with a 2048-entry histogram that stays in L1.
    """
    n = keys.shape[0]
    idx = np.arange(n).astype(np.int64)
    k_cur = keys.copy()
    idx_t = np.empty(n, dtype=np.int64)
    k_t = np.empty(n, dtype=np.int64)
    RADIX_BITS = 11
    RADIX = 1 << RADIX_BITS
    mask = RADIX - 1
    count = np.empty(RADIX + 1, dtype=np.int64)
    shift = 0
    while shift < 63:
        count[:] = 0
        for i in range(n):
            count[((k_cur[i] >> shift) & mask) + 1] += 1
        for b in range(RADIX):
            count[b + 1] += count[b]
        for i in range(n):
            d = (k_cur[i] >> shift) & mask
            p = count[d]
            idx_t[p] = idx[i]
            k_t[p] = k_cur[i]
            count[d] += 1
        idx, idx_t = idx_t, idx
        k_cur, k_t = k_t, k_cur
        shift += RADIX_BITS
    return idx


# Above this range size the split locates octant boundaries by binary search instead of by
# scanning. Linear costs O(hi-lo) per node, i.e. ~N per tree level; binary costs 8*log2(hi-lo),
# far cheaper near the root and far more expensive near the leaves, so the split uses whichever
# suits the range in front of it. Both produce identical boundaries.
BSEARCH_MIN = 512


@njit
def _grow_moment_arrays(acc, parent, n):
    """Re-allocate the moment accumulators to n node slots, preserving contents.

    Called whenever increase_tree_size grows the tree underneath them; kept as a free function
    because increase_tree_size only knows about the jitclass, not these build-local arrays.
    """
    acc_new = zeros((n, 3))
    acc_new[: acc.shape[0]] = acc
    parent_new = -ones(n, dtype=np.int64)
    parent_new[: parent.shape[0]] = parent
    return acc_new, parent_new


@njit(inline="always")
def _digit_upper_bound(keys, lo, hi, shift, d):
    """First index in [lo, hi) whose Morton digit at `shift` exceeds d.

    Valid because every bit above `shift` is equal across a node's range, so within that range
    sorted keys imply a non-decreasing digit at `shift`.
    """
    while lo < hi:
        mid = (lo + hi) >> 1
        if ((keys[mid] >> shift) & 7) <= d:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(inline="always")
def _octant_runs(keys, lo, hi, shift, bs, be, bd):
    """Split [lo, hi) into its per-octant contiguous runs; returns the number of runs found.

    Fills bs/be/bd (each length 8) with the start, end and octant digit of each run. Since the
    keys are Morton-sorted, each octant's members are already contiguous.
    """
    nb = 0
    start = lo
    if hi - lo > BSEARCH_MIN:
        for d in range(8):
            if start >= hi:
                break
            end = _digit_upper_bound(keys, start, hi, shift, d)
            if end > start:
                bs[nb], be[nb], bd[nb] = start, end, d
                nb += 1
                start = end
    else:
        while start < hi:
            d = (keys[start] >> shift) & 7
            end = start + 1
            while end < hi and ((keys[end] >> shift) & 7) == d:
                end += 1
            bs[nb], be[nb], bd[nb] = start, end, d
            nb += 1
            start = end
    return nb


@njit
def RollupMoments(tree, acc, parent, node_hi):
    """Bottom-up moments as a reverse linear scan, replacing the ComputeMoments traversal.

    The split allocates node indices in DFS discovery order, so every descendant of a node has a
    higher index than the node itself. Walking indices downward therefore reaches every child
    before its parent with no links dereferenced -- the traversal that made ComputeMomentsLinked
    DRAM-latency bound disappears, leaving one sequential pass with a single scattered write per
    node.

    Particle contributions are already folded in by the split, which sees each particle leaf
    while it is standing on the parent node. This pass only rolls nodes into their parents.

    `acc` holds the running sum of mass * position, indexed by (node - NumParticles), kept apart
    from Coordinates because Coordinates still holds each node's geometric centre -- needed both
    by the split, to place children, and here, to get Deltas. Reading that centre and replacing
    it with the centre of mass in the same step is safe: the split is finished with it.

    Monopole only. Quadrupoles need each parent's centre of mass to shift onto, so they come
    from RollupQuadrupoles in a second pass once this one has finished.
    """
    npart = tree.NumParticles
    for node in range(node_hi - 1, npart - 1, -1):
        a = node - npart
        m = tree.Masses[node]
        if m == 0.0:
            continue  # empty node (a degenerate cell the split created but never filled)
        inv = 1.0 / m
        d2 = 0.0
        p = parent[a]
        for k in range(3):
            com = acc[a, k] * inv
            dx = com - tree.Coordinates[node, k]
            d2 += dx * dx
            tree.Coordinates[node, k] = com
        tree.Deltas[node] = np.sqrt(d2)
        if p >= npart:  # the root has no parent
            pa = p - npart
            for k in range(3):
                acc[pa, k] += acc[a, k]
            tree.Masses[p] += m
            tree.Softenings[p] = max(tree.Softenings[p], tree.Softenings[node])


@njit(inline="always")
def _shift_quad_into(Q, p, m, r0, r1, r2):
    """Add a mass m displaced by r from node p's centre of mass into p's quadrupole.

    The parallel-axis term of the reduced (traceless) quadrupole: m * (3 r_k r_l - delta_kl r^2).
    Written out rather than looped over k,l so nothing has to allocate a length-3 vector inside
    the per-particle loop.
    """
    rsq = r0 * r0 + r1 * r1 + r2 * r2
    Q[p, 0, 0] += m * (3.0 * r0 * r0 - rsq)
    Q[p, 0, 1] += m * 3.0 * r0 * r1
    Q[p, 0, 2] += m * 3.0 * r0 * r2
    Q[p, 1, 0] += m * 3.0 * r1 * r0
    Q[p, 1, 1] += m * (3.0 * r1 * r1 - rsq)
    Q[p, 1, 2] += m * 3.0 * r1 * r2
    Q[p, 2, 0] += m * 3.0 * r2 * r0
    Q[p, 2, 1] += m * 3.0 * r2 * r1
    Q[p, 2, 2] += m * (3.0 * r2 * r2 - rsq)


@njit
def RollupQuadrupoles(tree, parent, pparent, node_hi):
    """Second bottom-up pass giving every node its quadrupole, again with no traversal.

    Must run after RollupMoments: shifting a child's moment onto its parent needs the parent's
    centre of mass, Q_p += Q_c + m_c (3 r r - r^2 I) with r = com_c - com_p, and that is only
    known once the monopole pass has finished.

    Two stages, in this order:

      1. every particle into the node that owns it (a particle's own quadrupole is zero, so it
         contributes the shift term alone).  `pparent` is recorded by the split, which sees each
         particle leaf while standing on its parent;
      2. every node into its parent, indices descending.  By the time a node is reached its own
         quadrupole is complete -- its node children were pushed earlier in this same descending
         scan, and all of its particle children were pushed in stage 1.

    Both stages are streaming scans over contiguous index ranges, like RollupMoments.
    """
    npart = tree.NumParticles
    Q = tree.Quadrupoles
    C = tree.Coordinates
    for i in range(npart):
        p = pparent[i]
        if p < npart:
            continue  # unreachable on a well-formed tree; a stray particle contributes nothing
        _shift_quad_into(Q, p, tree.Masses[i], C[i, 0] - C[p, 0], C[i, 1] - C[p, 1], C[i, 2] - C[p, 2])
    for node in range(node_hi - 1, npart - 1, -1):
        p = parent[node - npart]
        if p < npart:
            continue  # the root, which has no parent to shift onto
        m = tree.Masses[node]
        if m == 0.0:
            continue
        r0 = C[node, 0] - C[p, 0]
        r1 = C[node, 1] - C[p, 1]
        r2 = C[node, 2] - C[p, 2]
        Q[p, 0, 0] += Q[node, 0, 0]
        Q[p, 0, 1] += Q[node, 0, 1]
        Q[p, 0, 2] += Q[node, 0, 2]
        Q[p, 1, 0] += Q[node, 1, 0]
        Q[p, 1, 1] += Q[node, 1, 1]
        Q[p, 1, 2] += Q[node, 1, 2]
        Q[p, 2, 0] += Q[node, 2, 0]
        Q[p, 2, 1] += Q[node, 2, 1]
        Q[p, 2, 2] += Q[node, 2, 2]
        _shift_quad_into(Q, p, m, r0, r1, r2)


@njit
def ComputeMomentsLinked(tree, no):
    """ComputeMoments without a `children` array: iterate children via FirstSubnode/NextBranch.

    A node's children are the chain starting at FirstSubnode[no] and following NextBranch until
    it reaches NextBranch[no] (which the last child points at), so no separate child table is
    needed. Same recursion and same arithmetic as ComputeMoments otherwise.
    """
    quad = zeros((3, 3))
    if no < tree.NumParticles:
        return tree.Softenings[no], tree.Masses[no], quad, tree.Coordinates[no]
    m = 0.0
    com = zeros(3)
    hmax = 0.0
    stop = tree.NextBranch[no]
    c = tree.FirstSubnode[no]
    # `c >= 0` is redundant on a well-formed tree (the last child points at `stop`), but a
    # mislinked chain would otherwise run off the end and index NextBranch[-1] forever
    while c != stop and c >= 0:
        hi, mi, quadi, comi = ComputeMomentsLinked(tree, c)
        m += mi
        com += mi * comi
        hmax = max(hi, hmax)
        c = tree.NextBranch[c]
    tree.Masses[no] = m
    com = com / m
    if tree.HasQuads:
        c = tree.FirstSubnode[no]
        while c != stop and c >= 0:
            mi = tree.Masses[c]
            comi = tree.Coordinates[c]
            quadi = tree.Quadrupoles[c]
            ri = comi - com
            r2 = 0.0
            for k in range(3):
                r2 += ri[k] * ri[k]
            for k in range(3):
                for l in range(3):
                    quad[k, l] += quadi[k, l] + mi * 3 * ri[k] * ri[l]
                    if k == l:
                        quad[k, l] -= mi * r2
            c = tree.NextBranch[c]
        tree.Quadrupoles[no] = quad
    delta = 0.0
    for dim in range(3):
        dx = com[dim] - tree.Coordinates[no, dim]
        delta += dx * dx
    tree.Deltas[no] = np.sqrt(delta)
    tree.Coordinates[no] = com
    tree.Softenings[no] = hmax
    return hmax, m, quad, com


# ---------------------------------------------------------------------------------------------
# Parallel split
#
# The largest single phase of the build, and naturally divide-and-conquer: once the top of the tree
# is laid out, each sub-octant owns a disjoint contiguous range of the Morton-sorted particles. What
# stops a naive prange is node-index allocation -- the serial split hands out indices from one counter.
#
# So: descend serially to a frontier of subtrees, count each subtree's nodes exactly (a parallel pass
# mirroring the split with every store removed), prefix-sum the counts into disjoint contiguous
# blocks, and let each worker build into its own. Contiguous blocks also keep a worker's node writes
# out of everyone else's region.
#
# Groups that exhaust the Morton key are deferred, not built: resolving them means RekeySlice, whose
# node count cannot be known without doing the work, so it cannot be budgeted. The serial loop
# finishes them afterwards.
# ---------------------------------------------------------------------------------------------

# Below this N the frontier descent and counting pass cost more than the split they save. Measured
# on 16 threads (Plummer), serial -> parallel build: 0.27x at N=2e3, 0.70x at 1e4, 1.06x at 2e4,
# 1.21x at 3e4, 1.39x at 5e4, 1.58x at 1e5. Break-even is ~2e4; 3e4 keeps a clear margin.
PARALLEL_SPLIT_NMIN = 30_000
# Frontier subtrees per thread. More gives better load balance, but the descent that produces them
# is serial, so pushing it too fine just moves the time rather than removing it. Flat between 32
# and 512 at N=1e7 (697/704/709 ms on 16 threads); 8 is measurably worse at 727 ms.
FRONTIER_PER_THREAD = 32

# Below this N the thread spawn costs more than the gather saves. Measured on 16 threads, whole
# build with the permute serial -> parallel: 0.88x at N=2e4, 1.00x at 5e4, 1.04x at 1e5, 1.10x at
# 1e6, 1.18x at 3e6. Break-even is ~5e4; 1e5 keeps a margin, and the phase itself only becomes
# worth threading at large N (195ms -> 54ms at N=1e7).
PARALLEL_PERMUTE_NMIN = 100_000


@njit(cache=True)
def _count_subtree(keys, lo, hi, shift):
    """Return (nodes, deferrals) for the subtree the split would build under [lo, hi).

    Mirrors the split exactly with every store removed, so the two agree by construction -- which
    is what makes it safe to hand each worker a private, exactly-sized block of node indices. The
    partition depends only on the Morton digits, never on coordinates, which is why counting can
    skip all the geometry.

    A group whose key is exhausted (shift < 0) is counted as one deferral and not descended into.
    """
    cap = 64
    st_lo = np.empty(cap, dtype=np.int64)
    st_hi = np.empty(cap, dtype=np.int64)
    st_shift = np.empty(cap, dtype=np.int64)
    st_lo[0], st_hi[0], st_shift[0] = lo, hi, shift
    top = 1
    n = 0
    ndef = 0
    bs = np.empty(8, dtype=np.int64)
    be = np.empty(8, dtype=np.int64)
    bd = np.empty(8, dtype=np.int64)

    while top > 0:
        top -= 1
        lo = st_lo[top]
        hi = st_hi[top]
        shift = st_shift[top]
        if shift < 0:
            ndef += 1
            continue
        nb = _octant_runs(keys, lo, hi, shift, bs, be, bd)
        for b in range(nb):
            i = bs[b]
            j = be[b]
            if j - i > 1:
                n += 1
                if top >= st_lo.shape[0]:
                    add = st_lo.shape[0]
                    st_lo = concatenate((st_lo, np.empty(add, dtype=np.int64)))
                    st_hi = concatenate((st_hi, np.empty(add, dtype=np.int64)))
                    st_shift = concatenate((st_shift, np.empty(add, dtype=np.int64)))
                st_lo[top], st_hi[top], st_shift[top] = i, j, shift - 3
                top += 1
    return n, ndef


@njit(parallel=True, cache=True)
def _count_frontier(keys, nf, f_lo, f_hi, f_shift, counts, ndefs):
    """Exact per-subtree node and deferral counts, computed concurrently."""
    for t in prange(nf):
        counts[t], ndefs[t] = _count_subtree(keys, f_lo[t], f_hi[t], f_shift[t])


@njit(cache=True)
def _split_subtree(
    keys,
    node,
    lo,
    hi,
    shift,
    rnd,
    next_idx,
    npart,
    Coord,
    Sizes,
    First,
    Next,
    Mass,
    Soft,
    Acc,
    Parent,
    PPar,
    has_quads,
    d_node,
    d_lo,
    d_hi,
    d_rnd,
    d_at,
):
    """Build the subtree under `node`, taking node indices from `next_idx`.

    Identical to the body of the serial split loop, but writing into a caller-owned index block
    and recording key-exhausted groups into a caller-owned deferral slice instead of re-keying
    them. Everything it touches -- the particle range [lo, hi), the node block, the deferral
    slots -- belongs to this call alone, which is what makes concurrent calls safe.

    Returns (next free node index, number of deferrals recorded).
    """
    cap = 64
    st_node = np.empty(cap, dtype=np.int64)
    st_lo = np.empty(cap, dtype=np.int64)
    st_hi = np.empty(cap, dtype=np.int64)
    st_shift = np.empty(cap, dtype=np.int64)
    st_round = np.empty(cap, dtype=np.int64)
    st_node[0], st_lo[0], st_hi[0], st_shift[0], st_round[0] = node, lo, hi, shift, rnd
    top = 1
    nd = 0
    bs = np.empty(8, dtype=np.int64)
    be = np.empty(8, dtype=np.int64)
    bd = np.empty(8, dtype=np.int64)

    while top > 0:
        top -= 1
        node = st_node[top]
        lo = st_lo[top]
        hi = st_hi[top]
        shift = st_shift[top]
        rnd = st_round[top]

        if shift < 0:
            # hand back to the serial pass: re-keying permutes particle data and its node count
            # cannot be budgeted for ahead of time
            d_node[d_at + nd] = node
            d_lo[d_at + nd] = lo
            d_hi[d_at + nd] = hi
            d_rnd[d_at + nd] = rnd
            nd += 1
            continue

        last_child = -1
        nb = _octant_runs(keys, lo, hi, shift, bs, be, bd)
        for b in range(nb):
            i = bs[b]
            j = be[b]
            d = bd[b]
            if j - i == 1:
                c = i
                mi = Mass[i]
                Mass[node] += mi
                an = node - npart
                for k in range(3):
                    Acc[an, k] += mi * Coord[i, k]
                Soft[node] = max(Soft[node], Soft[i])
                if has_quads:
                    PPar[i] = node
            else:
                cnode = next_idx
                next_idx += 1
                for k in range(3):
                    Coord[cnode, k] = Coord[node, k] + Sizes[node] * octant_offsets[d, k]
                Sizes[cnode] = Sizes[node] * 0.5
                Mass[cnode] = 0.0
                Soft[cnode] = 0.0
                Parent[cnode - npart] = node
                c = cnode
                if top >= st_node.shape[0]:
                    add = st_node.shape[0]
                    st_node = concatenate((st_node, np.empty(add, dtype=np.int64)))
                    st_lo = concatenate((st_lo, np.empty(add, dtype=np.int64)))
                    st_hi = concatenate((st_hi, np.empty(add, dtype=np.int64)))
                    st_shift = concatenate((st_shift, np.empty(add, dtype=np.int64)))
                    st_round = concatenate((st_round, np.empty(add, dtype=np.int64)))
                st_node[top], st_lo[top], st_hi[top] = cnode, i, j
                st_shift[top], st_round[top] = shift - 3, rnd
                top += 1
            if last_child < 0:
                First[node] = c
            else:
                Next[last_child] = c
            last_child = c
        if last_child >= 0:
            Next[last_child] = Next[node]
    return next_idx, nd


@njit(cache=True)
def _bounding_cube(points):
    """(centre, side) of the smallest cube covering points, as BuildTree defines it: per-dimension
    midpoint centre, side = the largest per-dimension extent.

    One fused row-major pass. The obvious `points[:, dim].min()` per dimension walks the array six
    times with a 24-byte stride. Min and max are order-independent, so this is bit-identical.
    """
    n = points.shape[0]
    lo0 = hi0 = points[0, 0]
    lo1 = hi1 = points[0, 1]
    lo2 = hi2 = points[0, 2]
    for i in range(1, n):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        if x < lo0:
            lo0 = x
        elif x > hi0:
            hi0 = x
        if y < lo1:
            lo1 = y
        elif y > hi1:
            hi1 = y
        if z < lo2:
            lo2 = z
        elif z > hi2:
            hi2 = z
    center = zeros(3)
    center[0] = 0.5 * (hi0 + lo0)
    center[1] = 0.5 * (hi1 + lo1)
    center[2] = 0.5 * (hi2 + lo2)
    size = hi0 - lo0
    if hi1 - lo1 > size:
        size = hi1 - lo1
    if hi2 - lo2 > size:
        size = hi2 - lo2
    return center, size


@njit(parallel=True, cache=True)
def _permute_particles(order, points, masses, softening, keys, Coord, Mass, Soft, keys_sorted):
    """Gather particle data into Morton order.

    A pure gather: each iteration writes one distinct destination row and reads nothing another
    iteration writes, so it needs no coordination whatsoever. Reads are scattered (that is what a
    permutation is) but writes are sequential, so the threads mostly stream.

    A free function because prange is unavailable inside a jitclass method.
    """
    for i in prange(order.shape[0]):
        oi = order[i]
        for dim in range(3):
            Coord[i, dim] = points[oi, dim]
        Mass[i] = masses[oi]
        Soft[i] = softening[oi]
        keys_sorted[i] = keys[oi]


@njit(cache=True)
def _permute_particles_serial(order, points, masses, softening, keys, Coord, Mass, Soft, keys_sorted):
    """Serial _permute_particles, for N below the point where spawning threads pays for itself."""
    for i in range(order.shape[0]):
        oi = order[i]
        for dim in range(3):
            Coord[i, dim] = points[oi, dim]
        Mass[i] = masses[oi]
        Soft[i] = softening[oi]
        keys_sorted[i] = keys[oi]


@njit(parallel=True, cache=True)
def _split_frontier(
    keys,
    nf,
    f_node,
    f_lo,
    f_hi,
    f_shift,
    f_rnd,
    offsets,
    d_offsets,
    npart,
    Coord,
    Sizes,
    First,
    Next,
    Mass,
    Soft,
    Acc,
    Parent,
    PPar,
    has_quads,
    d_node,
    d_lo,
    d_hi,
    d_rnd,
    d_used,
):
    """Build every frontier subtree concurrently, each into its own node-index block."""
    for t in prange(nf):
        _, nd = _split_subtree(
            keys,
            f_node[t],
            f_lo[t],
            f_hi[t],
            f_shift[t],
            f_rnd[t],
            offsets[t],
            npart,
            Coord,
            Sizes,
            First,
            Next,
            Mass,
            Soft,
            Acc,
            Parent,
            PPar,
            has_quads,
            d_node,
            d_lo,
            d_hi,
            d_rnd,
            d_offsets[t],
        )
        d_used[t] = nd


@jitclass(spec)
class Octree:
    """Octree implementation."""

    def __init__(
        self,
        points,
        masses,
        softening,
        morton_order=True,
        quadrupole=False,
        compute_moments=True,
        radix=True,
    ):
        self.NumNodes = 0
        self.TreewalkIndices = -ones(points.shape[0], dtype=np.int64)
        self.HasQuads = quadrupole
        self.HasCoincidentPoints = False

        radix_path = radix and morton_order
        children = -ones((1, 8), dtype=np.int64)  # unused on the radix path; typing placeholder
        acc = zeros((1, 3))  # typing placeholders, replaced on the radix path
        parent = -ones(1, dtype=np.int64)
        pparent = -ones(1, dtype=np.int64)
        node_hi = 0
        if radix_path:
            # Radix-sort build: a single linear pass over Morton-sorted points produces the tree
            # already in Morton order, and links FirstSubnode/NextBranch as it goes -- so there is
            # no `children` table (0.96 GB at N=1e7, two-thirds of it unused) and no SetupTreewalk.
            node_hi, acc, parent, pparent = self.BuildTreeRadix(points, masses, softening)
        else:
            # Legacy insertion build (used for radix=False or morton_order=False).
            # first provisional treebuild to get the ordering right
            children = self.BuildTree(points, masses, softening)
            # set up the order of the treewalk
            SetupTreewalk(self, self.NumParticles, children)
            self.GetWalkIndices()  # get the Morton ordering of the points

            # if enabled, we rebuild the tree in Morton order (the order that points are visited in the depth-first traversal)
            if morton_order:
                children = self.BuildTree(
                    points[self.TreewalkIndices],
                    np.take(masses, self.TreewalkIndices),
                    np.take(softening, self.TreewalkIndices),
                )  # now re-build the tree with everything in order
                # re-do the treewalk order with the new indices
                SetupTreewalk(self, self.NumParticles, children)

        if compute_moments:
            # compute centers of mass, etc.
            if radix_path:
                # The split already folded in every particle, so these are reverse linear scans
                # over the node array rather than tree traversals.
                RollupMoments(self, acc, parent, node_hi)
                if quadrupole:
                    # Separate pass: the parallel-axis shift needs each parent's COM, which is
                    # only known once RollupMoments has finished.
                    RollupQuadrupoles(self, parent, pparent, node_hi)
            else:
                ComputeMoments(self, self.NumParticles, children)

    def BuildTree(self, points, masses, softening):
        # initialize random seed in case of non-unique positions
        np.random.seed(42)

        self.Initialize(len(points), self.NumNodes)

        # set the properties of the root node
        self.Sizes[self.NumParticles] = max(
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        )
        for dim in range(3):
            self.Coordinates[self.NumParticles, dim] = 0.5 * (points[:, dim].max() + points[:, dim].min())

        # set values for particles
        self.Coordinates[: self.NumParticles] = points
        self.Masses[: self.NumParticles] = masses
        self.Softenings[: self.NumParticles] = softening
        children = -ones((self.NumNodes, 8), dtype=np.int64)
        new_node_idx = self.NumParticles + 1
        # now we insert particles into the tree one at a time, setting up child pointers and initializing node properties as we go
        for i in range(self.NumParticles):
            pos = points[i]

            no = self.NumParticles  # walk the tree, starting at the root
            while no > -1:
                # first make sure we have enough storage
                while new_node_idx + 1 > self.NumNodes:
                    size_increase = increase_tree_size(self)
                    children = concatenate((children, -ones((size_increase, 8), dtype=np.int64)))

                octant = 0  # the index of the octant that the present point lives in
                for dim in range(3):
                    if pos[dim] > self.Coordinates[no, dim]:
                        octant += 1 << dim
                # check if there is a pre-existing node among the present node's children
                child_candidate = children[no, octant]
                if child_candidate > -1:
                    # it exists, now check if it's a node or a particle
                    if child_candidate < self.NumParticles:
                        # it's a particle - we have to create a new node of index new_node_idx containing the 2 points we've got, and point the pre-existing particle to the new particle
                        # EXCEPTION: if the pre-existing particle is at the same coordinate, we will perturb the position of the new particle slightly and start over
                        same_coord = True
                        for k in range(3):
                            if self.Coordinates[i, k] != self.Coordinates[child_candidate, k]:
                                same_coord = False
                        if same_coord:
                            self.HasCoincidentPoints = True
                            # Nudge the duplicate off its twin, ADDITIVELY: the old
                            # `Coordinates[i] *= exp(...)` cannot move an exactly-0.0 coordinate, so
                            # two particles at the origin re-tested equal forever. A zero component
                            # steps relative to the root cube, not its own ULP (5e-324) -- that would
                            # separate them but need ~1000 tree levels instead of ~52.
                            box = self.Sizes[self.NumParticles]
                            if box == 0.0:
                                box = 1.0  # every point identical; the answer is garbage anyway
                            pert = np.random.rand(3)
                            for k in range(3):
                                ck = self.Coordinates[i, k]
                                step = abs(ck)
                                if step == 0.0:
                                    step = box
                                self.Coordinates[i, k] = ck + step * 3e-16 * (pert[k] - 0.5)
                            points[i] = self.Coordinates[i]
                            no = self.NumParticles  # restart the tree traversal
                            continue
                        # end exception

                        children[no, octant] = new_node_idx
                        # set the center of the new node
                        self.Coordinates[new_node_idx] = self.Coordinates[no] + self.Sizes[no] * octant_offsets[octant]
                        # set the size of the new node
                        self.Sizes[new_node_idx] = self.Sizes[no] / 2
                        new_octant = 0
                        for dim in range(3):
                            if self.Coordinates[child_candidate, dim] > self.Coordinates[new_node_idx, dim]:
                                # get the octant of the new node that pre-existing particle lives in
                                new_octant += 1 << dim
                        # set the pre-existing particle as a child of the new node
                        children[new_node_idx, new_octant] = child_candidate
                        no = new_node_idx
                        new_node_idx += 1
                        continue  # restart the loop looking at the new node
                    else:  # if the child is an existing node, go to that one and start the loop anew
                        no = children[no, octant]
                        continue
                else:  # if the child does not exist, we let this point be that child (inserting it in the tree) and we're done with this point
                    children[no, octant] = i
                    no = -1
        return children

    def BuildTreeRadix(self, points, masses, softening):
        """Build the octree from Morton-sorted points in a single linear pass.

        Computes a 63-bit Morton (Z-order) key per point, radix-sorts them, then walks the sorted
        array top-down splitting each node's contiguous range by the 3-bit Morton digit at that
        level.  Because the sort already yields the depth-first (Morton) ordering, the particles are
        stored in that order and TreewalkIndices records the permutation - no second build is needed.

        Also seeds the moments as it goes: whenever the split emits a single-particle child it is
        already standing on the parent node, so that particle's mass, mass-weighted position and
        softening are folded straight in, and `parent` records each node's parent. RollupMoments
        then finishes the job with one reverse scan instead of a tree traversal.

        Returns (new_node_idx, acc, parent, pparent), where acc holds the running sum of
        mass * position and parent the owning node, both indexed by (node - NumParticles), and
        pparent maps each particle to its owning node (populated only when HasQuads).
        """
        N = points.shape[0]

        # --- root cube: per-dimension midpoint center, side = maximum extent (matches BuildTree) ---
        center, size = _bounding_cube(points)

        # --- Morton keys + radix sort give the Morton ordering directly ---
        keys = _morton_keys(points, center, size)
        order = _radix_argsort(keys)

        self.Initialize(N, self.NumNodes)
        self.TreewalkIndices = order

        # lay out particle data in Morton order; keys is permuted to stay aligned with the arrays
        keys_sorted = np.empty(N, dtype=np.int64)
        if N >= PARALLEL_PERMUTE_NMIN:
            _permute_particles(
                order,
                points,
                masses,
                softening,
                keys,
                self.Coordinates,
                self.Masses,
                self.Softenings,
                keys_sorted,
            )
        else:
            _permute_particles_serial(
                order,
                points,
                masses,
                softening,
                keys,
                self.Coordinates,
                self.Masses,
                self.Softenings,
                keys_sorted,
            )

        # set root node properties
        root = self.NumParticles
        self.Sizes[root] = size
        for dim in range(3):
            self.Coordinates[root, dim] = center[dim]

        new_node_idx = self.NumParticles + 1

        # Moment accumulators, sized for nodes only (particles need none) and so indexed by
        # node - NumParticles. They have to grow in step with increase_tree_size below.
        nnode = self.NumNodes - self.NumParticles
        acc = zeros((nnode, 3))
        parent = -ones(nnode, dtype=np.int64)
        # Which node owns each particle. Only the quadrupole rollup needs it (the monopole folds
        # particles in during the split), so it stays length-1 otherwise rather than costing 8N.
        pparent = -ones(N if self.HasQuads else 1, dtype=np.int64)
        bs = np.empty(8, dtype=np.int64)  # scratch for the per-octant run boundaries
        be = np.empty(8, dtype=np.int64)
        bd = np.empty(8, dtype=np.int64)

        # explicit stack of node ranges still to be partitioned: (node, lo, hi, shift, round)
        # shift is the bit position of the current 3-bit Morton digit; round counts re-keyings.
        cap = 64
        st_node = np.empty(cap, dtype=np.int64)
        st_lo = np.empty(cap, dtype=np.int64)
        st_hi = np.empty(cap, dtype=np.int64)
        st_shift = np.empty(cap, dtype=np.int64)
        st_round = np.empty(cap, dtype=np.int64)
        top = 0
        st_node[0] = root
        st_lo[0] = 0
        st_hi[0] = N
        st_shift[0] = 3 * (MORTON_BITS - 1)  # top digit of a fresh key
        st_round[0] = 0
        top = 1

        # Frontier for the parallel split: subtrees small enough to be independent work units.
        # The descent below stops at them, the parallel pass builds them, and whatever it defers
        # comes back onto this same stack afterwards.
        want = FRONTIER_PER_THREAD * get_num_threads()
        parallel_split = N >= PARALLEL_SPLIT_NMIN and want > 1
        min_work = N // want if parallel_split else 0
        fcap = 4 * want + 64
        f_node = np.empty(fcap, dtype=np.int64)
        f_lo = np.empty(fcap, dtype=np.int64)
        f_hi = np.empty(fcap, dtype=np.int64)
        f_shift = np.empty(fcap, dtype=np.int64)
        f_rnd = np.empty(fcap, dtype=np.int64)
        nf = 0

        # Two passes. The first descends to the frontier and stops; the parallel split then
        # runs and hands back whatever it could not budget for, which the second pass finishes
        # on the ordinary serial path. Sharing one loop body keeps re-keying and bucketing in
        # exactly one place.
        for _phase in range(2):
            while top > 0:
                top -= 1
                node = st_node[top]
                lo = st_lo[top]
                hi = st_hi[top]
                shift = st_shift[top]
                rnd = st_round[top]

                if shift < 0:
                    # We have exhausted the current key's bits but >1 point remains in this cell.
                    if rnd >= MORTON_MAX_ROUNDS:
                        # points are coincident to floating-point precision: park them in a bucket.
                        # Reserve its worst case (one node per 7 particles) up front so BuildBucket
                        # cannot reallocate behind acc/parent's back.
                        while new_node_idx + (hi - lo) // 7 + 2 > self.NumNodes:
                            size_increase = increase_tree_size(self)
                        if acc.shape[0] < self.NumNodes - self.NumParticles:
                            acc, parent = _grow_moment_arrays(acc, parent, self.NumNodes - self.NumParticles)
                        new_node_idx = self.BuildBucket(node, lo, hi, new_node_idx, acc, parent, pparent)
                        continue
                    # re-key this group relative to the node's (sub-)cell and re-sort the slice
                    self.RekeySlice(keys_sorted, node, lo, hi)
                    shift = 3 * (MORTON_BITS - 1)
                    rnd += 1

                # partition [lo, hi) by the 3-bit digit at the current shift (contiguous, since sorted)
                # Children are discovered in increasing digit order, which IS sibling order, so the
                # FirstSubnode/NextBranch links can be written here directly -- no `children` array
                # and no SetupTreewalk pass. NextBranch[node] is already set by node's own parent
                # (the split is top-down), and the root's is -1 from Initialize.
                last_child = -1
                nb = _octant_runs(keys_sorted, lo, hi, shift, bs, be, bd)
                for b in range(nb):
                    i = bs[b]
                    j = be[b]
                    d = bd[b]
                    if j - i == 1:  # single particle -> becomes a leaf child directly
                        c = i
                        # fold the particle into its parent's moments while we are standing on it
                        mi = self.Masses[i]
                        self.Masses[node] += mi
                        an = node - self.NumParticles
                        for k in range(3):
                            acc[an, k] += mi * self.Coordinates[i, k]
                        self.Softenings[node] = max(self.Softenings[node], self.Softenings[i])
                        if self.HasQuads:
                            pparent[i] = node  # the quadrupole shift needs this node's COM later
                    else:  # >1 particle in this octant -> create a child node and recurse
                        while new_node_idx + 1 > self.NumNodes:
                            size_increase = increase_tree_size(self)
                        if acc.shape[0] < self.NumNodes - self.NumParticles:
                            acc, parent = _grow_moment_arrays(acc, parent, self.NumNodes - self.NumParticles)
                        cnode = new_node_idx
                        new_node_idx += 1
                        for k in range(3):
                            self.Coordinates[cnode, k] = (
                                self.Coordinates[node, k] + self.Sizes[node] * octant_offsets[d, k]
                            )
                        self.Sizes[cnode] = self.Sizes[node] * 0.5
                        parent[cnode - self.NumParticles] = node
                        c = cnode
                        if min_work > 0 and j - i <= min_work and nf < fcap:
                            # small enough to hand to a worker; do not descend into it here
                            f_node[nf] = cnode
                            f_lo[nf] = i
                            f_hi[nf] = j
                            f_shift[nf] = shift - 3
                            f_rnd[nf] = rnd
                            nf += 1
                        else:
                            if top >= st_node.shape[0]:  # grow the stack if needed
                                add = st_node.shape[0]
                                st_node = concatenate((st_node, np.empty(add, dtype=np.int64)))
                                st_lo = concatenate((st_lo, np.empty(add, dtype=np.int64)))
                                st_hi = concatenate((st_hi, np.empty(add, dtype=np.int64)))
                                st_shift = concatenate((st_shift, np.empty(add, dtype=np.int64)))
                                st_round = concatenate((st_round, np.empty(add, dtype=np.int64)))
                            st_node[top] = cnode
                            st_lo[top] = i
                            st_hi[top] = j
                            st_shift[top] = shift - 3
                            st_round[top] = rnd
                            top += 1
                    if last_child < 0:
                        self.FirstSubnode[node] = c
                    else:
                        self.NextBranch[last_child] = c
                    last_child = c
                if last_child >= 0:
                    self.NextBranch[last_child] = self.NextBranch[node]

            if nf > 0:
                # exact node count per subtree, then a prefix sum gives each worker a private block
                counts = np.empty(nf, dtype=np.int64)
                ndefs = np.empty(nf, dtype=np.int64)
                _count_frontier(keys_sorted, nf, f_lo, f_hi, f_shift, counts, ndefs)

                offsets = np.empty(nf, dtype=np.int64)
                d_offsets = np.empty(nf, dtype=np.int64)
                run = new_node_idx
                drun = 0
                for t in range(nf):
                    offsets[t] = run
                    run += counts[t]
                    d_offsets[t] = drun
                    drun += ndefs[t]

                # size everything up front: workers must never reallocate underneath each other
                while run > self.NumNodes:
                    size_increase = increase_tree_size(self)
                if acc.shape[0] < self.NumNodes - self.NumParticles:
                    acc, parent = _grow_moment_arrays(acc, parent, self.NumNodes - self.NumParticles)

                d_node = np.empty(drun + 1, dtype=np.int64)
                d_lo = np.empty(drun + 1, dtype=np.int64)
                d_hi = np.empty(drun + 1, dtype=np.int64)
                d_rnd = np.empty(drun + 1, dtype=np.int64)
                d_used = np.zeros(nf, dtype=np.int64)

                _split_frontier(
                    keys_sorted,
                    nf,
                    f_node,
                    f_lo,
                    f_hi,
                    f_shift,
                    f_rnd,
                    offsets,
                    d_offsets,
                    self.NumParticles,
                    self.Coordinates,
                    self.Sizes,
                    self.FirstSubnode,
                    self.NextBranch,
                    self.Masses,
                    self.Softenings,
                    acc,
                    parent,
                    pparent,
                    self.HasQuads,
                    d_node,
                    d_lo,
                    d_hi,
                    d_rnd,
                    d_used,
                )
                new_node_idx = run

                # Finish the key-exhausted groups on the ordinary serial path. Re-entering the loop
                # above is deliberate: it already knows how to re-key and bucket, and these are rare.
                for t in range(nf):
                    for q in range(d_used[t]):
                        e = d_offsets[t] + q
                        if top >= st_node.shape[0]:
                            add = st_node.shape[0]
                            st_node = concatenate((st_node, np.empty(add, dtype=np.int64)))
                            st_lo = concatenate((st_lo, np.empty(add, dtype=np.int64)))
                            st_hi = concatenate((st_hi, np.empty(add, dtype=np.int64)))
                            st_shift = concatenate((st_shift, np.empty(add, dtype=np.int64)))
                            st_round = concatenate((st_round, np.empty(add, dtype=np.int64)))
                        st_node[top] = d_node[e]
                        st_lo[top] = d_lo[e]
                        st_hi[top] = d_hi[e]
                        st_shift[top] = -3  # exhausted, so the loop re-keys or buckets
                        st_round[top] = d_rnd[e]
                        top += 1

            if _phase == 1 or nf == 0:
                break
            nf = 0
            min_work = 0  # everything left is finished serially by the second pass
        return new_node_idx, acc, parent, pparent

    def RekeySlice(self, keys_sorted, node, lo, hi):
        """Recompute Morton keys for particles [lo, hi) relative to node's cell and re-sort them.

        Used when a group of points collides (shares a key) at the deepest level: treating the
        node's cell as a fresh root gives MORTON_BITS more bits of resolution per dimension.  The
        particle data (Coordinates/Masses/Softenings/TreewalkIndices) and keys_sorted are all
        permuted consistently so the slice stays Morton-sorted.
        """
        g = hi - lo
        sub = self.Coordinates[lo:hi]
        center = zeros(3)
        for dim in range(3):
            center[dim] = self.Coordinates[node, dim]
        subkeys = _morton_keys(sub, center, self.Sizes[node])
        perm = _radix_argsort(subkeys)

        # gather the slice into temporaries in the new order, then write back
        new_coord = zeros((g, 3))
        new_mass = zeros(g)
        new_soft = zeros(g)
        new_idx = zeros(g, dtype=np.int64)
        new_keys = zeros(g, dtype=np.int64)
        for a in range(g):
            p = perm[a]
            for dim in range(3):
                new_coord[a, dim] = self.Coordinates[lo + p, dim]
            new_mass[a] = self.Masses[lo + p]
            new_soft[a] = self.Softenings[lo + p]
            new_idx[a] = self.TreewalkIndices[lo + p]
            new_keys[a] = subkeys[p]
        for a in range(g):
            for dim in range(3):
                self.Coordinates[lo + a, dim] = new_coord[a, dim]
            self.Masses[lo + a] = new_mass[a]
            self.Softenings[lo + a] = new_soft[a]
            self.TreewalkIndices[lo + a] = new_idx[a]
            keys_sorted[lo + a] = new_keys[a]

    def BuildBucket(self, node, lo, hi, new_node_idx, acc, parent, pparent):
        """Attach particles [lo, hi), coincident to float precision, as a chain of nodes under node.

        Each node holds up to 8 particle children; when more remain, the 8th slot links to a
        further node. Links FirstSubnode/NextBranch directly, as the digit loop does. Geometry is
        degenerate (the points coincide) but valid, and no particle is dropped.

        Seeds moments the same way the digit loop does. The caller must have reserved enough node
        slots for the whole chain, since growing the tree here would leave acc/parent behind.
        """
        self.HasCoincidentPoints = True  # only reachable for bit-identical positions
        count = hi - lo
        cur = node
        slot = 0
        last = -1
        for jj in range(count):
            remaining = count - 1 - jj
            if slot == 7 and remaining > 0:
                nn = new_node_idx
                new_node_idx += 1
                for k in range(3):
                    self.Coordinates[nn, k] = self.Coordinates[cur, k] + self.Sizes[cur] * octant_offsets[7, k]
                self.Sizes[nn] = self.Sizes[cur] * 0.5
                parent[nn - self.NumParticles] = cur
                # nn is the last child of cur; then we continue underneath nn
                if last < 0:
                    self.FirstSubnode[cur] = nn
                else:
                    self.NextBranch[last] = nn
                self.NextBranch[nn] = self.NextBranch[cur]
                cur = nn
                slot = 0
                last = -1
            c = lo + jj
            mi = self.Masses[c]
            self.Masses[cur] += mi
            ac = cur - self.NumParticles
            for k in range(3):
                acc[ac, k] += mi * self.Coordinates[c, k]
            self.Softenings[cur] = max(self.Softenings[cur], self.Softenings[c])
            if self.HasQuads:
                pparent[c] = cur
            if last < 0:
                self.FirstSubnode[cur] = c
            else:
                self.NextBranch[last] = c
            last = c
            slot += 1
        if last >= 0:
            self.NextBranch[last] = self.NextBranch[cur]
        return new_node_idx

    def GetWalkIndices(self):  # gets the ordering of the particles in the treewalk
        index = 0
        no = self.NumParticles
        while no > -1:
            if no < self.NumParticles:
                self.TreewalkIndices[index] = no
                index += 1
                no = self.NextBranch[no]
            else:
                no = self.FirstSubnode[no]

    def Initialize(self, Npart, NumNodes):
        """Allocate all attribute arrays and initialize"""
        self.NumParticles = Npart
        # this is the number of elements in the tree, whether nodes or particles. can make this smaller but this has a safety factor
        if NumNodes:
            self.NumNodes = NumNodes
        else:
            # initial guess for storage needed; can always increase if needed
            self.NumNodes = int(1.5 * Npart + 1)
        self.Sizes = zeros(self.NumNodes)
        self.Deltas = zeros(self.NumNodes)
        self.Masses = zeros(self.NumNodes)
        # No need to initialize this beyond zero, all n>0 moments are 0 for a single particle
        if self.HasQuads:
            self.Quadrupoles = zeros((self.NumNodes, 3, 3))
        self.Softenings = zeros(self.NumNodes)
        self.Coordinates = zeros((self.NumNodes, 3))
        self.NextBranch = -ones(self.NumNodes, dtype=np.int64)
        self.FirstSubnode = -ones(self.NumNodes, dtype=np.int64)


@njit
def ComputeMoments(tree, no, children):
    """Does a recursive pass through the tree and computes centers of mass, total mass, max softening, and distance between geometric center and COM"""
    quad = zeros((3, 3))
    if no < tree.NumParticles:  # if this is a particle, just return the properties
        return tree.Softenings[no], tree.Masses[no], quad, tree.Coordinates[no]
    else:
        m = 0
        com = zeros(3)
        hmax = 0
        for c in children[no]:
            if c > -1:
                hi, mi, quadi, comi = ComputeMoments(tree, c, children)
                m += mi
                com += mi * comi
                hmax = max(hi, hmax)
        tree.Masses[no] = m
        com = com / m
        if tree.HasQuads:
            for c in children[no]:
                if c > -1:
                    mi = tree.Masses[c]  # per-child mass: 'mi' from the loop above is the LAST child's
                    comi = tree.Coordinates[c]
                    quadi = tree.Quadrupoles[c]
                    ri = comi - com
                    r2 = 0
                    for k in range(3):
                        r2 += ri[k] * ri[k]
                    for k in range(3):
                        for l in range(3):
                            quad[k, l] += quadi[k, l] + mi * 3 * ri[k] * ri[l]
                            if k == l:
                                quad[k, l] -= (
                                    mi * r2
                                )  # Calculate the quadrupole moment based on the moments of the subcells
            tree.Quadrupoles[no] = quad
        delta = 0
        for dim in range(3):
            dx = com[dim] - tree.Coordinates[no, dim]
            delta += dx * dx
        tree.Deltas[no] = np.sqrt(delta)
        tree.Coordinates[no] = com
        tree.Softenings[no] = hmax
        return hmax, m, quad, com


@njit
def SetupTreewalk(tree, no, children):
    """Recursively link each node to its first subnode and its next sibling branch, so a walk
    can traverse the tree iteratively (opening a node -> FirstSubnode, accepting -> NextBranch)."""
    if no < tree.NumParticles:
        return  # leaf nodes are handled from above
    last_node = -1
    for c in children[no]:
        if c < 0:
            continue
        # if we haven't yet set current node's next node, do so
        if tree.FirstSubnode[no] < 0:
            tree.FirstSubnode[no] = c
        # set this up to point to the next "branch" of the tree to look at if we sum the force for the current branch
        if last_node > -1:
            tree.NextBranch[last_node] = c
        last_node = c

    # need to deal with the last child: must link it up to the sibling of the present node
    tree.NextBranch[last_node] = tree.NextBranch[no]

    for c in children[no]:
        if c >= tree.NumParticles:  # if we have a node, call routine recursively
            SetupTreewalk(tree, c, children)


@njit
def increase_tree_size(tree, fac=1.2):
    """Reallocate the tree data with storage increased by factor fac"""
    old_size = tree.NumNodes
    size_increase = max(int(old_size * fac + 1) - old_size, 1)
    #    print("Increasing size of node list by ", size_increase)  # by %g" % fac)

    tree.Sizes = concatenate((tree.Sizes, zeros(size_increase)))
    tree.Deltas = concatenate((tree.Deltas, zeros(size_increase)))
    tree.Masses = concatenate((tree.Masses, zeros(size_increase)))
    tree.Softenings = concatenate((tree.Softenings, zeros(size_increase)))
    tree.NextBranch = concatenate((tree.NextBranch, -ones(size_increase, dtype=np.int64)))
    tree.FirstSubnode = concatenate((tree.FirstSubnode, -ones(size_increase, dtype=np.int64)))
    tree.Coordinates = concatenate((tree.Coordinates, zeros((size_increase, 3))))
    if tree.HasQuads:
        tree.Quadrupoles = concatenate((tree.Quadrupoles, zeros((size_increase, 3, 3))))
    tree.NumNodes += size_increase

    return size_increase
