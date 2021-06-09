from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, prange, types
from numba.experimental import jitclass
import numpy as np
from numpy import empty, empty_like, zeros, zeros_like, sqrt, ones

spec = [
    ('Sizes', float64[:]), # side length of tree nodes
    ('Deltas', float64[:]), # distance between COM and geometric center of node
    ('Coordinates', float64[:,:]), # location of center of mass of node (actually stores _geometric_ center before we do the moments pass)
    ('Masses', float64[:]), # total mass of node
    ('NumParticles', int64), # number of particles in the tree
    ('NumNodes',int64), # number of particles + nodes (i.e. mass elements) in the tree
    ('Softenings', float64[:]), # individual softenings for particles, _maximum_ softening of inhabitant particles for nodes
    ('NextBranch',int64[:]), 
    ('FirstSubnode',int64[:]),
    ('ParentNode',int64[:]),
   # ('children',int64[:,:]) # indices of child nodes
]


@jitclass(spec)
class Octree(object):
    """Octree implementation."""

    def __init__(self, points, masses, softening):
        # initialize all attributes
        self.NumParticles = points.shape[0]
        self.NumNodes = 2*self.NumParticles # this is the number of elements in the tree, whether nodes or particles. can make this smaller but this has a safety factor
        self.Sizes = zeros(self.NumNodes)
        self.Deltas = zeros(self.NumNodes)
        self.Masses = zeros(self.NumNodes)
        self.Softenings = zeros(self.NumNodes)
        self.Coordinates = zeros((self.NumNodes,3))
        self.Deltas = zeros(self.NumNodes)
# below will be used for future GADGET-like treewalk, which obviates the need to store all 8 children and allows a non-recursive treewalk
        self.NextBranch = -ones(self.NumNodes, dtype=np.int64) # 
        self.FirstSubnode = -ones(self.NumNodes, dtype=np.int64)
        self.ParentNode = -ones(self.NumNodes, dtype=np.int64)

        children = -ones((self.NumNodes,8),dtype=np.int64)

        octant_offsets = 0.25 * np.array([[-1,-1,-1],
                                          [1,-1,-1],
                                          [-1,1,-1],
                                          [1,1,-1],
                                          [-1,-1,1],
                                          [1,-1,1],
                                          [-1,1,1],
                                          [1,1,1]])

        # set values for particles
        self.Coordinates[:self.NumParticles] = points
        self.Masses[:self.NumParticles] = masses
        self.Softenings[:self.NumParticles] = softening

        # set the properties of the root node
        self.Sizes[self.NumParticles] = max(points[:,0].max()-points[:,0].min(), points[:,1].max()-points[:,1].min(), points[:,2].max()-points[:,2].min())
        for dim in range(3): self.Coordinates[self.NumParticles,dim] = 0.5*(points[:,dim].max() + points[:,dim].min())

        new_node_idx = self.NumParticles + 1
        
        # now we insert particles into the tree one at a time, setting up child pointers and initializing node properties as we go
        for i in range(self.NumParticles):
            pos = points[i]
            
            no = self.NumParticles  # walk the tree, starting at the root                  
            while no > -1:
                octant = 0 #the index of the octant that the present point lives in
                for dim in range(3):
                    if pos[dim] > self.Coordinates[no,dim]: octant += 1 << dim
                
                # check if there is a pre-existing node among the present node's children
                child_candidate = children[no,octant]
                if child_candidate > -1: # it exists, now check if it's a node or a particle
                    if child_candidate < self.NumParticles: # it's a particle - we have to create a new node of index new_node_idx containing the 2 points we've got, and point the pre-existing particle to the new particle
                        children[no,octant] = new_node_idx;                        
                        self.Coordinates[new_node_idx] = self.Coordinates[no] + self.Sizes[no] * octant_offsets[octant] # set the center of the new node
                        self.Sizes[new_node_idx] = self.Sizes[no] / 2 # set the size of the new node
                        new_octant = 0;
                        for dim in range(3):
                            if self.Coordinates[child_candidate,dim] > self.Coordinates[new_node_idx,dim]: new_octant += 1 << dim # get the octant of the new node that pre-existing particle lives in
                        children[new_node_idx,new_octant] = child_candidate # set the pre-existing particle as a child of the new node
                        no = new_node_idx
                        new_node_idx += 1
                        continue # restart the loop looking at the new node
                    else: # if the child is an existing node, go to that one and start the loop anew
                        no = children[no,octant]
                        continue 
                else: # if the child does not exist, we let this point be that child (inserting it in the tree) and we're done with this point
                    children[no,octant] = i
                    no = -1
                    
        ComputeMoments(self,self.NumParticles,children)
        SetupTreewalk(self,self.NumParticles,children)
#        print(self.NextNode)
        
        
@njit
def ComputeMoments(tree, no, children): # does a recursive pass through the tree and computes centers of mass, total mass, max softening, and distance between geometric center and COM
    if no < tree.NumParticles: # if this is a particle, just return the properties
        return tree.Softenings[no], tree.Masses[no], tree.Coordinates[no] 
    else:
        m = 0
        com = zeros(3)
        hmax = 0
        for c in children[no]:
            if c > -1:
                hi, mi, comi = ComputeMoments(tree,c,children)
                m += mi
                com += mi*comi
                hmax = max(hi, hmax)
        tree.Masses[no] = m
        com = com/m
        delta = 0
        for dim in range(3):
            dx = com[dim] - tree.Coordinates[no,dim]
            delta += dx * dx
        tree.Deltas[no] = np.sqrt(delta)
        tree.Coordinates[no] = com
        tree.Softenings[no] = hmax
        return hmax, m, com

@njit            
def SetupTreewalk(tree,no,children):
   # print(no)
    if no < tree.NumParticles: return # leaf nodes are handled from above
    last_node = -1
    last_child = -1
    for c in children[no]:
        if c < 0: continue        
        tree.ParentNode[c] = no
        if tree.FirstSubnode[no] < 0: tree.FirstSubnode[no] = c  # if we haven't yet set current node's next node, do so

        if last_node > -1:
            tree.NextBranch[last_node] = c # set this up to point to the next "branch" of the tree to look at if we sum the force for the current branch
        last_node = c            

    # need to deal with the last child: must link it up to the sibling of the present node
    tree.NextBranch[last_node] = tree.NextBranch[no]

    for c in children[no]:
        if c >= tree.NumParticles: # if we have a node, call routine recursively
            SetupTreewalk(tree,c,children)    


