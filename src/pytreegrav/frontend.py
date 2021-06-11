import numpy as np
from numba import njit, prange
from numpy import zeros_like, sqrt
from .kernel import *
from .octree import *
from .treewalk import *
from .bruteforce import *

def ConstructTree(pos,m,softening=None):
    """Builds the tree containing particle data, for subsequent potential/field evaluation
    Arguments:
    pos -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of particle softening lengths

    Returns:
    Octree instance built from particle data
    """
    if softening is None: softening = zeros_like(m)

    return Octree(pos,m,softening)

def Potential(pos, m, softening=None, G=1., theta=.7, tree=None, return_tree=False,parallel=False,method='adaptive'):
    """Returns the gravitational potential for a set of particles with positions x and masses m, at the positions of those particles, using either brute force or tree-based methods depending on the number of particles.

    Arguments:
    pos -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Optional arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (N,) array containing kernel support radii for gravitational softening - set to 0 by default
    theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False)
    tree -- optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree -- return the tree used for future use (default False)
    method -- 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    """
    if softening is None: softening = np.zeros_like(m)

    # figure out which method to use
    if method == 'adaptive':
        if len(pos) > 1000: method = 'tree'
        else: method = 'bruteforce'

    if method == 'bruteforce': # we're using brute force
        if parallel:
            phi = Potential_bruteforce_parallel(pos,m,softening,G=G)
        else:
            phi = Potential_bruteforce(pos,m,softening,G=G)
        if return_tree:
            tree = None
    else: # we're using the tree algorithm
        if tree is None: tree = ConstructTree(np.float64(pos),np.float64(m), np.float64(softening)) # build the tree if needed
        if parallel:
            phi = PotentialTarget_tree_parallel(pos,softening,tree,theta=theta,G=G)
        else:
            phi = PotentialTarget_tree(pos,softening,tree,theta=theta,G=G)

    if return_tree:
        return tree, phi
    else:
        return phi
            

def Accel(pos, m, softening=None, G=1., theta=.7, tree=None, return_tree=False,parallel=False,method='adaptive'):
    """Returns the gravitational acceleration for a set of particles, due to those particles.

    Arguments:
    pos -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Optional arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (N,) array containing kernel support radii for gravitational softening
    theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)    
    parallel -- If True, will parallelize the force summation over all available cores. (default False) 
    tree -- optional pre-generated Octree: this can contain any set of particles, not necessarily the ones at pos (default None)
    return_tree -- return the tree used for future use (default False)
    method -- 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)
    """
    if softening is None: softening = np.zeros_like(m)

    # figure out which method to use
    if method == 'adaptive':
        if len(pos) > 1000: method = 'tree'
        else: method = 'bruteforce'

    if method == 'bruteforce': # we're using brute force
        if parallel:
            g = Accel_bruteforce_parallel(pos,m,softening,G=G)
        else:
            g = Accel_bruteforce(pos,m,softening,G=G)
        if return_tree:
            tree = None
    else: # we're using the tree algorithm
        if tree is None: tree = ConstructTree(np.float64(pos),np.float64(m), np.float64(softening)) # build the tree if needed
        if parallel:
            g = AccelTarget_tree_parallel(pos,softening,tree,theta=theta,G=G)
        else:
            g = AccelTarget_tree(pos,softening,tree,theta=theta,G=G)

    if return_tree:
        return tree, g
    else:
        return g
