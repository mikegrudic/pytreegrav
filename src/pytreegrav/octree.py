"""Implementation of the Octree jitclass"""

import numpy as np
from numpy import zeros, ones, concatenate
from numba import float64, boolean, int64, njit
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

    Processes 8 bits per pass (base 256), covering the 63 significant bits of a Morton key in 8
    passes.  O(N) in the number of keys.
    """
    n = keys.shape[0]
    idx = np.arange(n).astype(np.int64)
    tmp = np.empty(n, dtype=np.int64)
    RADIX_BITS = 8
    RADIX = 1 << RADIX_BITS  # 256
    mask = RADIX - 1
    count = np.empty(RADIX + 1, dtype=np.int64)
    shift = 0
    while shift < 63:
        count[:] = 0
        for i in range(n):
            digit = (keys[idx[i]] >> shift) & mask
            count[digit + 1] += 1
        for b in range(RADIX):
            count[b + 1] += count[b]
        for i in range(n):
            digit = (keys[idx[i]] >> shift) & mask
            tmp[count[digit]] = idx[i]
            count[digit] += 1
        idx, tmp = tmp, idx
        shift += RADIX_BITS
    return idx


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

        if radix and morton_order:
            # Radix-sort build: a single linear pass over Morton-sorted points produces the tree
            # already in Morton order, so no provisional build or GetWalkIndices pass is needed.
            children = self.BuildTreeRadix(points, masses, softening)
            SetupTreewalk(self, self.NumParticles, children)
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
                            self.Coordinates[i] *= np.exp(3e-16 * (np.random.rand(3) - 0.5))  # random perturbation
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

        Returns the (NumNodes, 8) child-index array, exactly as BuildTree does.
        """
        N = points.shape[0]

        # --- root cube: per-dimension midpoint center, side = maximum extent (matches BuildTree) ---
        center = zeros(3)
        size = 0.0
        for dim in range(3):
            lo = points[:, dim].min()
            hi = points[:, dim].max()
            center[dim] = 0.5 * (hi + lo)
            if hi - lo > size:
                size = hi - lo

        # --- Morton keys + radix sort give the Morton ordering directly ---
        keys = _morton_keys(points, center, size)
        order = _radix_argsort(keys)

        self.Initialize(N, self.NumNodes)
        self.TreewalkIndices = order

        # lay out particle data in Morton order; keys is permuted to stay aligned with the arrays
        keys_sorted = np.empty(N, dtype=np.int64)
        for i in range(N):
            oi = order[i]
            for dim in range(3):
                self.Coordinates[i, dim] = points[oi, dim]
            self.Masses[i] = masses[oi]
            self.Softenings[i] = softening[oi]
            keys_sorted[i] = keys[oi]

        # set root node properties
        root = self.NumParticles
        self.Sizes[root] = size
        for dim in range(3):
            self.Coordinates[root, dim] = center[dim]

        children = -ones((self.NumNodes, 8), dtype=np.int64)
        new_node_idx = self.NumParticles + 1

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
                    # points are coincident to floating-point precision: park them in a bucket
                    children, new_node_idx = self.BuildBucket(children, node, lo, hi, new_node_idx)
                    continue
                # re-key this group relative to the node's (sub-)cell and re-sort the slice
                self.RekeySlice(keys_sorted, node, lo, hi)
                shift = 3 * (MORTON_BITS - 1)
                rnd += 1

            # partition [lo, hi) by the 3-bit digit at the current shift (contiguous, since sorted)
            i = lo
            while i < hi:
                d = (keys_sorted[i] >> shift) & 7
                j = i + 1
                while j < hi and ((keys_sorted[j] >> shift) & 7) == d:
                    j += 1
                if j - i == 1:  # single particle -> becomes a leaf child directly
                    children[node, d] = i
                else:  # >1 particle in this octant -> create a child node and recurse
                    while new_node_idx + 1 > self.NumNodes:
                        size_increase = increase_tree_size(self)
                        children = concatenate((children, -ones((size_increase, 8), dtype=np.int64)))
                    cnode = new_node_idx
                    new_node_idx += 1
                    for k in range(3):
                        self.Coordinates[cnode, k] = (
                            self.Coordinates[node, k] + self.Sizes[node] * octant_offsets[d, k]
                        )
                    self.Sizes[cnode] = self.Sizes[node] * 0.5
                    children[node, d] = cnode
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
                i = j

        return children

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

    def BuildBucket(self, children, node, lo, hi, new_node_idx):
        """Attach particles [lo, hi), coincident to float precision, as a chain of nodes under node.

        Each node holds up to 8 particle children; when more remain, slot 7 links to a further node.
        Geometry is degenerate (the points coincide) but valid, and no particle is dropped.  Returns
        the (possibly reallocated) children array and the updated new_node_idx.
        """
        count = hi - lo
        cur = node
        slot = 0
        for jj in range(count):
            remaining = count - 1 - jj
            if slot == 7 and remaining > 0:
                # reserve slot 7 as a link to a continuation node
                while new_node_idx + 1 > self.NumNodes:
                    size_increase = increase_tree_size(self)
                    children = concatenate((children, -ones((size_increase, 8), dtype=np.int64)))
                nn = new_node_idx
                new_node_idx += 1
                for k in range(3):
                    self.Coordinates[nn, k] = self.Coordinates[cur, k] + self.Sizes[cur] * octant_offsets[7, k]
                self.Sizes[nn] = self.Sizes[cur] * 0.5
                children[cur, 7] = nn
                cur = nn
                slot = 0
            children[cur, slot] = lo + jj
            slot += 1
        return children, new_node_idx

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
        self.Deltas = zeros(self.NumNodes)
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
