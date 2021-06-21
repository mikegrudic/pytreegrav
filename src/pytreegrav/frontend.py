import numpy as np
from numba import njit, prange
from numpy import zeros_like, sqrt
from .kernel import *
from .octree import *
from .treewalk import *
from .bruteforce import *

def valueTestMethod(method):
    methods = ["adaptive","bruteforce","tree"]

    ## check if method is a str
    if type(method) != str:
        raise TypeError("Invalid method type %s, must be str"%type(method))

    ## check if method is a valid method
    if method not in methods:
        raise ValueError("Invalid method %s. Must be one of: %s"%(method,str(methods)))

def ConstructTree(pos,m,softening=None,quadrupole=False):
    """Builds the tree containing particle data, for subsequent potential/field evaluation
    Arguments:
    pos -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of particle softening lengths

    Returns:
    Octree instance built from particle data
    """
    if softening is None: softening = zeros_like(m)
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(m)) and np.all(np.isfinite(softening))):
        print("Invalid input detected - aborting treebuild to avoid going into an infinite loop!")
        raise
    return Octree(pos,m,softening,quadrupole=quadrupole)

def Potential(pos, m, softening=None, G=1., theta=.7, tree=None, return_tree=False,parallel=False,method='adaptive',quadrupole=False):
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
    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

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
        if tree is None:
            tree = ConstructTree(np.float64(pos),np.float64(m), np.float64(softening), quadrupole=quadrupole) # build the tree if needed            
        idx = tree.TreewalkIndices

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos,idx,axis=0)
        h_sorted = np.take(softening,idx)
        
        if parallel:
            phi = PotentialTarget_tree_parallel(pos_sorted,h_sorted,tree,theta=theta,G=G)
        else:
            phi = PotentialTarget_tree(pos_sorted,h_sorted,tree,theta=theta,G=G)

        # now reorder phi back to the order of the input positions
        phi = np.take(phi,idx.argsort())

    if return_tree:
        return phi, tree
    else:
        return phi

def PotentialTarget(pos_target, pos_source, m_source, h_target=None, h_source=None, G=1., theta=.7, tree=None, return_tree=False,parallel=False,method='adaptive',quadrupole=False):
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

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

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
        if tree is None:
            tree = ConstructTree(np.float64(pos_source),np.float64(m_source), np.float64(h_source), quadrupole=quadrupole) # build the tree if needed            
        
        if parallel:
            phi = PotentialTarget_tree_parallel(pos_target,h_target,tree,theta=theta,G=G)
        else:
            phi = PotentialTarget_tree(pos_target,h_target,tree,theta=theta,G=G)

    if return_tree:
        return phi, tree
    else:
        return phi
            

def Accel(pos, m, softening=None, G=1., theta=.7, tree=None, return_tree=False,parallel=False,method='adaptive',quadrupole=False):
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

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

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
        if tree is None:
            tree = ConstructTree(np.float64(pos),np.float64(m), np.float64(softening), quadrupole=quadrupole) # build the tree if needed
        idx = tree.TreewalkIndices            

        # sort by the order they appear in the treewalk to improve access pattern efficiency
        pos_sorted = np.take(pos,idx,axis=0)
        h_sorted = np.take(softening,idx)
        
        if parallel:
            g = AccelTarget_tree_parallel(pos_sorted,h_sorted,tree,theta=theta,G=G)
        else:
            g = AccelTarget_tree(pos_sorted,h_sorted,tree,theta=theta,G=G)

        # now g is in the tree-order: reorder it back to the original order
        g = np.take(g,idx.argsort(),axis=0)

    if return_tree:
        return g, tree
    else:
        return g

def AccelTarget(pos_target, pos_source, m_source, h_target=None, h_source=None, G=1., theta=.7, tree=None, return_tree=False, parallel=False, method='adaptive',quadrupole=False):
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

    ## test if method is correct, otherwise raise a ValueError
    valueTestMethod(method)

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
        if tree is None: tree = ConstructTree(np.float64(pos_source),np.float64(m_source), np.float64(h_source), quadrupole=quadrupole) # build the tree if needed
        if parallel:
            g = AccelTarget_tree_parallel(pos_target,h_target,tree,theta=theta,G=G)
        else:
            g = AccelTarget_tree(pos_target,h_target,tree,theta=theta,G=G)

    if return_tree:
        return g, tree
    else:
        return g
