from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from ..kernel import *
import numpy as np

@njit(fastmath=True)
def PotentialWalk(pos, tree, no=-1, softening=0,theta=1):
    """Returns the gravitational potential at pos by performing the Barnes-Hut treewalk using the provided octree structure

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure    

    Keyword arguments:
    no - index of the tree node to do the walk for - defaults to starting with the top-level node
    softening - softening radius of the particle at which the force is being evaluated - needed if you want the short-range force to be momentum-conserving
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1\
% accuracy)
    """
    if no < 0: no = tree.NumParticles # default to the top-level node
    phi = 0.
    dx = tree.Coordinates[no,0]-pos[0]
    dy = tree.Coordinates[no,1]-pos[1]
    dz = tree.Coordinates[no,2]-pos[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)

    if no < tree.NumParticles:
        if r==0: return 0. # by default we neglect the self-potential.
        return tree.Masses[no] * PotentialKernel(r,softening)
    elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], max(tree.Softenings[no],softening)+tree.Sizes[no]):
        return -tree.Masses[no]/r # we can sum the monopole
    else: # we have to open the node
        for c in tree.children[no]: # loop over subnodes
            if c < 0:
                continue
            else:
                phi += PotentialWalk(pos, tree, c, softening,theta=theta) # add up the potential contribution you get for each subnode
    return phi

@njit(fastmath=True)
def ForceWalk(pos, tree, no=-1, softening=0,theta=1):
    """Returns the gravitational force at pos by performing the Barnes-Hut treewalk using the provided octree structure

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure    

    Keyword arguments:
    no - index of the tree node to do the walk for - defaults to starting with the top-level node
    softening - softening radius of the particle at which the force is being evaluated - needed if you want the short-range force to be momentum-conserving
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1\
% accuracy)
    """
    if no < 0: no = tree.NumParticles # default to the top-level node

    force = np.zeros(3)

    dx = tree.Coordinates[no,0]-pos[0]
    dy = tree.Coordinates[no,1]-pos[1]
    dz = tree.Coordinates[no,2]-pos[2]

    r2 = dx*dx + dy*dy + dz*dz
    r = sqrt(r2)

    if no < tree.NumParticles:
        if r==0.: return force # by default we neglect the self-acceleration
        fac = -tree.Masses[no] * ForceKernel(r,softening)
        force[0] = dx*fac
        force[1] = dy*fac
        force[2] = dz*fac
        return force
    elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], max(tree.Softenings[no],softening)+tree.Sizes[no]):
        fac = -tree.Masses[no]/(r2*r) ## -M /|r|^2 * r/|r| (vector eqn)
        force[0] = dx*fac
        force[1] = dy*fac
        force[2] = dz*fac
        return force
    else: # we have to open the node
        for c in tree.children[no]: # loop over subnodes
            if c < 0:
                continue
            else:
                force += ForceWalk(pos, tree, c, softening=softening, theta=theta) # add up the force contribution you get for each subnode
    return force 

@njit(parallel=True, fastmath=True)
def GetPotentialParallel(pos,tree, softening=None, G=1., theta=0.7):
    if softening is None: softening = zeros(pos.shape[0])
    result = empty(pos.shape[0])
    for i in prange(pos.shape[0]):
        result[i] = G*PotentialWalk(pos[i], tree, softening=softening[i], theta=theta)
    return result

@njit(fastmath=True)
def GetPotential(pos,tree, softening=None, G=1., theta=0.7):
    if softening is None: softening = zeros(pos.shape[0])
    result = empty(pos.shape[0])
    for i in range(pos.shape[0]):
        result[i] = G*PotentialWalk(pos[i], tree, softening=softening[i], theta=theta)
    return result

@njit(fastmath=True)
def GetAccel(pos, tree, softening=None, G=1., theta=0.7):
    """ Get's the vector gravitational acceleration
    Arguments:
    pos -- the target positions to evaluate the force
    tree -- an octree instance
    Keyword Arguments:
    softening=None -- target softening, defaults to 0
    G=1 -- gravitational constant, pass in your units here
    theta=0.7 -- opening angle of the tree"""
    if softening is None: softening = zeros(pos.shape[0])
    result = empty(pos.shape)
    for i in range(pos.shape[0]):
        result[i] = G*ForceWalk(pos[i], tree, softening=softening[i], theta=theta)
    return result

@njit(parallel=True, fastmath=True)
def GetAccelParallel(pos, tree, softening, G=1., theta=0.7):
    if softening is None: softening = zeros(len(pos), dtype=np.float64)    
    result = empty(pos.shape)
    for i in prange(pos.shape[0]):
        result[i] = G*ForceWalk(pos[i], tree, softening=softening[i], theta=theta)
    return result
