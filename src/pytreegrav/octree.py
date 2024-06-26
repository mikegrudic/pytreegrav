from numba import (
    int32,
    deferred_type,
    optional,
    float64,
    boolean,
    int64,
    njit,
    jit,
    prange,
    types,
)
from numba.experimental import jitclass
import numpy as np
from numpy import empty, empty_like, zeros, zeros_like, sqrt, ones, concatenate

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
    ):
        self.NumNodes = 0
        self.TreewalkIndices = -ones(points.shape[0], dtype=np.int64)
        self.HasQuads = quadrupole
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
        self.Initialize(len(points), self.NumNodes)

        # set the properties of the root node
        self.Sizes[self.NumParticles] = max(
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min(),
        )
        for dim in range(3):
            self.Coordinates[self.NumParticles, dim] = 0.5 * (
                points[:, dim].max() + points[:, dim].min()
            )

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
                    children = concatenate(
                        (children, -ones((size_increase, 8), dtype=np.int64))
                    )

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
                            if (
                                self.Coordinates[i, k]
                                != self.Coordinates[child_candidate, k]
                            ):
                                same_coord = False
                        if same_coord:
                            self.Coordinates[i] *= np.exp(
                                3e-16 * (np.random.rand(3) - 0.5)
                            )  # random perturbation
                            points[i] = self.Coordinates[i]
                            no = self.NumParticles  # restart the tree traversal
                            continue
                        # end exception

                        children[no, octant] = new_node_idx
                        # set the center of the new node
                        self.Coordinates[new_node_idx] = (
                            self.Coordinates[no]
                            + self.Sizes[no] * octant_offsets[octant]
                        )
                        # set the size of the new node
                        self.Sizes[new_node_idx] = self.Sizes[no] / 2
                        new_octant = 0
                        for dim in range(3):
                            if (
                                self.Coordinates[child_candidate, dim]
                                > self.Coordinates[new_node_idx, dim]
                            ):
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
    tree.NextBranch = concatenate(
        (tree.NextBranch, -ones(size_increase, dtype=np.int64))
    )
    tree.FirstSubnode = concatenate(
        (tree.FirstSubnode, -ones(size_increase, dtype=np.int64))
    )
    tree.Coordinates = concatenate((tree.Coordinates, zeros((size_increase, 3))))
    if tree.HasQuads:
        tree.Quadrupoles = concatenate((tree.Quadrupoles, zeros((size_increase, 3, 3))))
    tree.NumNodes += size_increase

    return size_increase
