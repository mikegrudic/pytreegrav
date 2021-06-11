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

    Returns:
    phi -- shape (N,) array of potentials at the particle positions
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
        return phi, tree
    else:
        return phi

def PotentialTarget(pos_target, pos_source, m_source, h_target=None, h_source=None, G=1., theta=.7, tree=None, return_tree=False,parallel=False,method='adaptive'):
    """Returns the gravitational potential due to a set of particles, at a set of "target" positions that can be different from the particle positions, using either brute force or tree-based methods depending on the number of particles and target positions.

    Arguments:
    pos_target -- shape (N,3) array of target positions where the potential is to be computed
    pos_source -- shape (M,3) array of source positions of the particles generating the potential
    m_source -- shape (M,) array of particle masses    
    

    Optional arguments:
    h_target -- shape (N,) array of target softening radii - this will be the _minimum_ softening length used in any interaction computed for this target point
    h_source -- shape (M,) array of source softening radii
    G -- gravitational constant (default 1.0)
    theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False)
    tree -- optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree -- return the tree used for future use (default False)
    method -- 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)

    Returns:
    phi -- shape (N,) array of potentials at the target positions
    """
    if h_target is None: h_target = np.zeros(len(pos_target))
    if h_source is None: h_source = np.zeros(len(pos_source))

    # figure out which method to use
    if method == 'adaptive':
        if len(pos_target)*len(pos_source) > 10**6: method = 'tree'
        else: method = 'bruteforce'

    if method == 'bruteforce': # we're using brute force
        if parallel:
            phi = PotentialTarget_bruteforce_parallel(pos_target,h_target,pos_source,m_source,h_source,G=G)
        else:
            phi = PotentialTarget_bruteforce(pos_target,h_target,pos_source,m_source,h_source,G=G)
        if return_tree:
            tree = None
    else: # we're using the tree algorithm
        if tree is None: tree = ConstructTree(np.float64(pos_source),np.float64(m_source), np.float64(h_source)) # build the tree if needed
        if parallel:
            phi = PotentialTarget_tree_parallel(pos_target,h_target,tree,theta=theta,G=G)
        else:
            phi = PotentialTarget_tree(pos_target,h_target,tree,theta=theta,G=G)

    if return_tree:
        return phi, tree
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
        return g, tree
    else:
        return g

def AccelTarget(pos_target, pos_source, m_source, h_target=None, h_source=None, G=1., theta=.7, tree=None, return_tree=False, parallel=False, method='adaptive'):
    """Returns the gravitational acceleration due to a set of particles, at a set of "target" positions that can be different from the particle positions, using either brute force or tree-based methods depending on the number of particles and target positions.

    Arguments:
    pos_target -- shape (N,3) array of target positions where the field is to be computed
    pos_source -- shape (M,3) array of source positions of the particles generating the field
    m_source -- shape (M,) array of particle masses    
    

    Optional arguments:
    h_target -- shape (N,) array of target softening radii - this will be the _minimum_ softening length used in any interaction computed for this target point
    h_source -- shape (M,) array of source softening radii
    G -- gravitational constant (default 1.0)
    softening -- shape (N,) array containing kernel support radii for gravitational softening - set to 0 by default
    theta -- cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 0.7, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False)
    tree -- optional pre-generated Octree: this can contain any set of particles, not necessarily the target particles at pos (default None)
    return_tree -- return the tree used for future use (default False)
    method -- 'adaptive', 'tree', or 'bruteforce' (default adaptive tries to pick the faster choice)

    Returns:
    g -- shape (N,3) array of gravitational fields at the target positions
    """
    if h_target is None: h_target = np.zeros(len(pos_target))
    if h_source is None: h_source = np.zeros(len(pos_source))

    # figure out which method to use
    if method == 'adaptive':
        if len(pos_target)*len(pos_source) > 10**6: method = 'tree'
        else: method = 'bruteforce'

    if method == 'bruteforce': # we're using brute force
        if parallel:
            g = AccelTarget_bruteforce_parallel(pos_target,h_target,pos_source,m_source,h_source,G=G)
        else:
            g = AccelTarget_bruteforce(pos_target,h_target,pos_source,m_source,h_source,G=G)
        if return_tree:
            tree = None
    else: # we're using the tree algorithm
        if tree is None: tree = ConstructTree(np.float64(pos_source),np.float64(m_source), np.float64(h_source)) # build the tree if needed
        if parallel:
            g = AccelTarget_tree_parallel(pos_target,h_target,tree,theta=theta,G=G)
        else:
            g = AccelTarget_tree(pos_target,h_target,tree,theta=theta,G=G)

    if return_tree:
        return g, tree
    else:
        return g
