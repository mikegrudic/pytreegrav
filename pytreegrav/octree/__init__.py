import numpy as np
from numba import njit
from numpy import zeros_like, sqrt

## merge the namespaces
from .octree import *
from .treewalk import *
from ..kernel import *

def ConstructTree(x,m,softening):
    """Wrapper function around Octree.
    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of particle softening lengths"""

    return Octree(x,m,softening)

def Potential(pos, m, softening=None, G=1., theta=.7, parallel=False, tree=None, return_tree=False):
    """Returns the approximate gravitational potential for a set of particles with positions x and masses m.

    Arguments:
    pos -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (N,) array containing kernel support radii for gravitational softening
    theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False)
    tree -- optional pre-generated kd-tree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree -- return the tree used for future use (default False)
    """
    if softening == None:
        softening = np.zeros_like(m)
    if tree is None: tree = ConstructTree(np.float64(pos),np.float64(m), np.float64(softening))
    result = zeros(len(m))

    if parallel:
        pot = GetPotentialParallel(np.float64(pos),tree , G=G,theta=theta)
    else:
        pot = GetPotential(np.float64(pos), tree, G=G,theta=theta)
    if return_tree:
        return pot, tree
    else:
        return pot
    

def Accel(pos, m, softening=None, G=1., theta=7., parallel=False, tree=None, return_tree=False):
    """Returns the approximate gravitational acceleration for a set of particles with positions x and masses m.

    Arguments:
    pos -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (N,) array containing kernel support radii for gravitational softening
    theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1% accuracy)    
    parallel -- If True, will parallelize the force summation over all available cores. (default False) 
    tree -- optional pre-generated kd-tree: this can contain any set of particles, not necessarily the ones at pos (default None)
    return_tree -- return the tree used for future use (default False)
"""
    
    if softening == None:
        softening = np.zeros_like(m)
    if not tree: tree = ConstructTree(np.float64(pos),np.float64(m), np.float64(softening))
    if parallel:
        acc = GetAccelParallel(np.float64(pos), tree, softening=softening, G=G, theta=theta)
    else:
        acc =  GetAccel(np.float64(pos), tree, softening=softening, G=G, theta=theta)
    if return_tree:
        return acc, tree
    else:
        return acc

