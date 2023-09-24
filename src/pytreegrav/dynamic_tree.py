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
from numpy import empty, empty_like, zeros, zeros_like, sqrt, ones

spec = [
    ("Sizes", float64[:]),  # side length of tree nodes
    ("Deltas", float64[:]),  # distance between COM and geometric center of node
    (
        "Coordinates",
        float64[:, :],
    ),  # location of center of mass of node (actually stores _geometric_ center before we do the moments pass)
    ("Velocities", float64[:, :]),  # velocity of the center of mass of node
    ("VelocityDisp", float64[:]),  # center-of-mass velocity dispersion
    ("Masses", float64[:]),  # total mass of node
    ("Quadrupoles", float64[:, :, :]),  # Quadrupole moment of the node
    (
        "HasQuads",
        boolean,
    ),  # Allow us to quickly check if Quadrupole moments exist to keep monopole calculations fast
    ("NumParticles", int64),  # number of particles in the tree
    ("NumNodes", int64),  # number of particles + nodes (i.e. mass elements) in the tree
    (
        "Softenings",
        float64[:],
    ),  # individual softenings for particles, _maximum_ softening of inhabitant particles for nodes
    ("NextBranch", int64[:]),
    ("FirstSubnode", int64[:]),
    ("TreewalkIndices", int64[:]),
    # ('children',int64[:,:]) # indices of child nodes
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
class DynamicOctree(object):
    """Octree implementation that stores node velocities for correlation functions and dynamic updates."""

    def __init__(
        self, points, masses, softening, vels, morton_order=True, quadrupole=False
    ):
        self.TreewalkIndices = -ones(points.shape[0], dtype=np.int64)
        self.HasQuads = quadrupole
        children = self.BuildTree(
            points, masses, softening, vels
        )  # first provisional treebuild to get the ordering right
        SetupTreewalk(
            self, self.NumParticles, children
        )  # set up the order of the treewalk
        ComputeMomentsDynamic(
            self, self.NumParticles, children
        )  # compute centers of mass, etc.
        self.GetWalkIndices()  # get the Morton ordering of the points

        if (
            morton_order
        ):  # if enabled, we rebuild the tree in Morton order (the order that points are visited in the depth-first traversal)
            children = self.BuildTree(
                points[self.TreewalkIndices],
                np.take(masses, self.TreewalkIndices),
                np.take(softening, self.TreewalkIndices),
                vels[self.TreewalkIndices],
            )  # now re-build the tree with everything in order
            SetupTreewalk(
                self, self.NumParticles, children
            )  # re-do the treewalk order with the new indices

        ComputeMomentsDynamic(
            self, self.NumParticles, children
        )  # compute centers of mass, etc.

    def BuildTree(self, points, masses, softening, vels):
        # initialize all attributes
        self.NumParticles = points.shape[0]
        self.NumNodes = (
            2 * self.NumParticles
        )  # this is the number of elements in the tree, whether nodes or particles. can make this smaller but this has a safety factor
        self.Sizes = zeros(self.NumNodes)
        self.Deltas = zeros(self.NumNodes)
        self.Masses = zeros(self.NumNodes)
        if self.HasQuads:
            self.Quadrupoles = zeros(
                (self.NumNodes, 3, 3)
            )  # No need to initialize this beyond zero, all n>0 moments are 0 for a single particle
        self.Softenings = zeros(self.NumNodes)
        self.Coordinates = zeros((self.NumNodes, 3))
        self.Velocities = zeros((self.NumNodes, 3))
        self.VelocityDisp = zeros(self.NumNodes)
        self.Deltas = zeros(self.NumNodes)
        self.NextBranch = -ones(self.NumNodes, dtype=np.int64)
        self.FirstSubnode = -ones(self.NumNodes, dtype=np.int64)
        #        self.ParentNode = -ones(self.NumNodes, dtype=np.int64)

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
        self.Velocities[: self.NumParticles] = vels
        self.Masses[: self.NumParticles] = masses
        self.Softenings[: self.NumParticles] = softening
        children = -ones((self.NumNodes, 8), dtype=np.int64)
        new_node_idx = self.NumParticles + 1

        # now we insert particles into the tree one at a time, setting up child pointers and initializing node properties as we go
        for i in range(self.NumParticles):
            pos = points[i]

            no = self.NumParticles  # walk the tree, starting at the root
            while no > -1:
                octant = 0  # the index of the octant that the present point lives in
                for dim in range(3):
                    if pos[dim] > self.Coordinates[no, dim]:
                        octant += 1 << dim

                # check if there is a pre-existing node among the present node's children
                child_candidate = children[no, octant]
                if (
                    child_candidate > -1
                ):  # it exists, now check if it's a node or a particle
                    if (
                        child_candidate < self.NumParticles
                    ):  # it's a particle - we have to create a new node of index new_node_idx containing the 2 points we've got, and point the pre-existing particle to the new particle
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
                        self.Coordinates[new_node_idx] = (
                            self.Coordinates[no]
                            + self.Sizes[no] * octant_offsets[octant]
                        )  # set the center of the new node
                        self.Sizes[new_node_idx] = (
                            self.Sizes[no] / 2
                        )  # set the size of the new node
                        new_octant = 0
                        for dim in range(3):
                            if (
                                self.Coordinates[child_candidate, dim]
                                > self.Coordinates[new_node_idx, dim]
                            ):
                                new_octant += (
                                    1 << dim
                                )  # get the octant of the new node that pre-existing particle lives in
                        children[
                            new_node_idx, new_octant
                        ] = child_candidate  # set the pre-existing particle as a child of the new node
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

    def ReorderTree(self):
        no = self.NumParticles

    def GetWalkIndices(self):  # gets the ordering of the particles in the treewalk
        index = 0
        node_index = 0
        no = self.NumParticles
        while no > -1:
            if no < self.NumParticles:
                self.TreewalkIndices[index] = no
                index += 1
                no = self.NextBranch[no]
            else:
                no = self.FirstSubnode[no]


@njit
def ComputeMomentsDynamic(
    tree, no, children
):  # does a recursive pass through the tree and computes centers of mass, total mass, max softening, and distance between geometric center and COM
    quad = zeros((3, 3))
    if no < tree.NumParticles:  # if this is a particle, just return the properties
        return (
            tree.Softenings[no],
            tree.Masses[no],
            quad,
            tree.Coordinates[no],
            tree.Velocities[no],
            0,
        )
    else:
        m = 0
        com = zeros(3)
        vel = zeros(3)
        vdisp = 0
        hmax = 0
        for c in children[no]:
            if c > -1:
                hi, mi, quadi, comi, veli, vdispi = ComputeMomentsDynamic(
                    tree, c, children
                )
                m += mi
                com += mi * comi
                vel += mi * veli
                vdisp += mi * vdispi
                hmax = max(hi, hmax)
        tree.Masses[no] = m
        com = com / m
        vel = vel / m
        #        vdisp = vdisp/m
        for c in children[no]:
            if c > -1:
                dv = tree.Velocities[c] - vel
                vdisp += tree.Masses[c] * (
                    dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]
                )
        vdisp = vdisp / m
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
        tree.Velocities[no] = vel
        tree.VelocityDisp[no] = vdisp
        return hmax, m, quad, com, vel, vdisp


@njit
def SetupTreewalk(tree, no, children):
    # print(no)
    if no < tree.NumParticles:
        return  # leaf nodes are handled from above
    last_node = -1
    last_child = -1
    for c in children[no]:
        if c < 0:
            continue
        #        tree.ParentNode[c] = no
        if tree.FirstSubnode[no] < 0:
            tree.FirstSubnode[
                no
            ] = c  # if we haven't yet set current node's next node, do so

        if last_node > -1:
            tree.NextBranch[
                last_node
            ] = c  # set this up to point to the next "branch" of the tree to look at if we sum the force for the current branch
        last_node = c

    # need to deal with the last child: must link it up to the sibling of the present node
    tree.NextBranch[last_node] = tree.NextBranch[no]

    for c in children[no]:
        if c >= tree.NumParticles:  # if we have a node, call routine recursively
            SetupTreewalk(tree, c, children)
