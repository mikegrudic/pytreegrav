from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, jitclass, prange
import numpy as np
from numpy import empty, empty_like, zeros, zeros_like, sqrt

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
            return False
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
        return True

node_type.define(KDNode.class_type.instance_type)

@jit
def ConstructKDTree(x, m, softening=None):
    if softening is None:
        softening = np.zeros_like(m)
    root = KDNode(x, m, softening)
    
    nodes = np.array([root,],dtype=KDNode)
    new_nodes = empty(2,dtype=KDNode)
    axis = 0
    divisible_nodes = True
    while divisible_nodes:
        N = len(nodes)
        divisible_nodes = False
        count = 0
        for i in range(N):
            if nodes[i].IsLeaf:
                continue
            else:
                divisible_nodes += nodes[i].GenerateChildren(axis)
                if nodes[i].HasLeft:
                    new_nodes[count] = nodes[i].left
                    count += 1
                if nodes[i].HasRight:
                    new_nodes[count] = nodes[i].right
                    count += 1
                    
        axis = (axis+1)%3
        if divisible_nodes:
            nodes = new_nodes[:count]
            new_nodes = empty(count*2, dtype=KDNode)
    return root        