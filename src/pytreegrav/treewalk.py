from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from .kernel import *
import numpy as np

@njit(fastmath=True)
def PotentialWalk(pos,  tree, softening=0, no=-1, theta=0.7):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree instance

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure    

    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0: no = tree.NumParticles # we default to the top-level node index        
    phi = 0
    dx = np.empty(3,dtype=np.float64)
    
    while no > -1:
        r = 0
        for k in range(3): 
            dx[k] = tree.Coordinates[no,k] - pos[k]
            r += dx[k]*dx[k]
        r = sqrt(r)
        h = max(tree.Softenings[no],softening)        
        
        if no < tree.NumParticles: # if we're looking at a leaf/particle
            if r>0: # by default we neglect the self-potential
                if r < h:
                    phi += tree.Masses[no] * PotentialKernel(r,h) 
                else:
                    phi -= tree.Masses[no] / r
            no = tree.NextBranch[no]
        elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], h+tree.Sizes[no]*0.6+tree.Deltas[no]): # if we satisfy the criteria for accepting the monopole
            phi -= tree.Masses[no]/r
            no = tree.NextBranch[no]
        else: # open the node
            no = tree.FirstSubnode[no]
            
    return phi


@njit(fastmath=True)
def AccelWalk(pos,  tree, softening=0, no=-1, theta=0.7): #,include_self_potential=False):
    """Returns the gravitational acceleration field at position x by performing the Barnes-Hut treewalk using the provided octree instance

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree instance storing the tree structure    

    Keyword arguments:
    softening - softening radius of the particle at which the force is being evaluated - we use the greater of the target and source softenings when evaluating the softened potential
    no - index of the top-level node whose field is being summed - defaults to the global top-level node, can use a subnode in principle for e.g. parallelization
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7 gives ~1% accuracy)
    """
    if no < 0: no = tree.NumParticles # we default to the top-level node index
    g = np.zeros(3,dtype=np.float64)
    dx = np.empty(3,dtype=np.float64)    
    
    while no > -1: # loop until we get to the end of the tree
        r2 = 0
        for k in range(3): 
            dx[k] = tree.Coordinates[no,k] - pos[k]
            r2 += dx[k]*dx[k]
        r = sqrt(r2)
        h = max(tree.Softenings[no],softening)
        
        sum_field = False
        
        if no < tree.NumParticles: # if we're looking at a leaf/particle
            if r > 0:  # no self-force
                if r < h: # within the softening radius
                    fac = tree.Masses[no] * ForceKernel(r,h) # fac stores the quantity M(<R)/R^3 to be used later for force computation
                else: # use point mass force
                    fac = tree.Masses[no]/(r*r2)
                sum_field = True
            no = tree.NextBranch[no]
        elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], h+tree.Sizes[no]*0.6+tree.Deltas[no]): # if we satisfy the criteria for accepting the monopole            
            fac = tree.Masses[no]/(r*r2)
            sum_field = True
            no = tree.NextBranch[no] # go to the next branch in the tree
        else: # open the node
            no = tree.FirstSubnode[no]
            continue
            
        if sum_field: # OK, we have M(<R)/R^3 for this element and can now sum the force
            for k in range(3): g[k] += fac * dx[k]
            
    return g

def PotentialTarget_tree(pos_target, softening_target, tree, G=1., theta=0.7):
    """Returns the gravitational potential at the specified points, given a tree containing the mass distribution

    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the potential
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all potential computations
    tree -- Octree instance containing the positions, masses, and softenings of the source particles

    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)

    Returns:
    shape (N,) array of potential values at each point in pos
    """
    result = empty(pos_target.shape[0])    
    for i in prange(pos_target.shape[0]):
        result[i] = G*PotentialWalk(pos_target[i], tree, softening=softening_target[i], theta=theta)
    return result

# JIT this function and its parallel version
PotentialTarget_tree_parallel = njit(PotentialTarget_tree,fastmath=True,parallel=True)
PotentialTarget_tree = njit(PotentialTarget_tree,fastmath=True)

def AccelTarget_tree(pos_target, softening_target, tree, G=1., theta=0.7):
    """Returns the gravitational acceleration at the specified points, given a tree containing the mass distribution

    Arguments:
    pos_target -- shape (N,3) array of positions at which to evaluate the field
    softening_target -- shape (N,) array of *minimum* softening lengths to be used in all accel computations
    tree -- Octree instance containing the positions, masses, and softenings of the source particles

    Optional arguments:
    G -- gravitational constant (default 1.0)
    theta -- accuracy parameter, smaller is more accurate, larger is faster (default 0.7)

    Returns:
    shape (N,3) array of acceleration values at each point in pos_target
    """
    if softening_target is None: softening_target = zeros(pos_target.shape[0])
    result = empty(pos_target.shape)
    for i in prange(pos_target.shape[0]):
        result[i] = G*AccelWalk(pos_target[i], tree, softening=softening_target[i], theta=theta)
    return result

# JIT this function and its parallel version
AccelTarget_tree_parallel = njit(AccelTarget_tree,fastmath=True,parallel=True)
AccelTarget_tree = njit(AccelTarget_tree,fastmath=True)
