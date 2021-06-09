from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, prange, types
from numba.experimental import jitclass
import numpy as np
from numpy import empty, empty_like, zeros, zeros_like, sqrt
from numba.typed import List

node_type = deferred_type()

spec = [
    ('bounds', float64[:,:]),
    ('size', float64),
    ('delta', float64),
    ('points', float64[:,:]),
    ('masses', float64[:]),
    ('Npoints', int64),
    ('h', float64),
    ('softening', float64[:]),
    ('mass', float64),
    ('COM', float64[:]),
    ('IsLeaf', boolean),
    ('HasLeft', boolean),
    ('HasRight', boolean),
    ('left', optional(node_type)),
    ('right', optional(node_type)),
]

@jitclass(spec)
class KDNode(object):
    def __init__(self, points, masses, softening):
        self.bounds = empty((3,2))
        self.bounds[0,0] = points[:,0].min()
        self.bounds[0,1] = points[:,0].max()
        self.bounds[1,0] = points[:,1].min()
        self.bounds[1,1] = points[:,1].max()
        self.bounds[2,0] = points[:,2].min()
        self.bounds[2,1] = points[:,2].max()

        self.softening = softening
        self.h = self.softening.max()
        
        self.size = max(self.bounds[0,1]-self.bounds[0,0],self.bounds[1,1]-self.bounds[1,0],self.bounds[2,1]-self.bounds[2,0])
        self.points = points
        self.Npoints = points.shape[0]
        self.masses = masses
        self.mass = np.sum(masses)
        self.delta = 0.
        if self.Npoints == 1:
            self.IsLeaf = True
            self.COM = points[0]
        else:
            self.IsLeaf = False
            self.COM = zeros(3)
            for k in range(3):
                for i in range(self.Npoints):
                    self.COM[k] += points[i,k]*masses[i]
                self.COM[k] /= self.mass
                self.delta += (0.5*(self.bounds[k,1]+self.bounds[k,0]) - self.COM[k])**2
            self.delta = sqrt(self.delta)

        self.HasLeft = False
        self.HasRight = False        
        self.left = None
        self.right = None

    def GenerateChildren(self, axis):
        if self.IsLeaf:
            return 0
        x = self.points[:,axis]
        med = (self.bounds[axis,0] + self.bounds[axis,1])/2
        index = (x<med)
        if np.any(index):
            self.left = KDNode(self.points[index], self.masses[index], self.softening[index])
            self.HasLeft = True
        index = np.invert(index)
        if np.any(index):
            self.right = KDNode(self.points[index],self.masses[index], self.softening[index])
            self.HasRight = True
        self.points = empty((1,1))
        self.masses = empty(1)
        self.softening = empty(1)
        return 1

node_type.define(KDNode.class_type.instance_type)

@njit
def ConstructKDTree(x, m, softening):
    if len(np.unique(x[:,0])) < len(x):
        raise Exception("Non-unique particle positions are currently not supported by the tree-building algorithm. Consider perturbing your positions with a bit of noise if you really want to proceed.")
    root = KDNode(x, m, softening)
    nodes = [root,]
    axis = 0
    divisible_nodes = 1
    count = 0
    while divisible_nodes > 0:
        N = len(nodes)
        divisible_nodes = 0
        for i in range(count, N): # loop through the nodes we spawned in the previous pass
            count += 1
            if nodes[i].IsLeaf:
                continue                
            else:
                generated_children = nodes[i].GenerateChildren(axis)
                divisible_nodes += generated_children
                if nodes[i].HasLeft:
                    nodes.append(nodes[i].left)
                if nodes[i].HasRight:
                    nodes.append(nodes[i].right)
                    
        axis = (axis+1)%3
    return root
            
