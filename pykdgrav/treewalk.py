from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from .kernel import *
import numpy as np

@njit(fastmath=True)
def PotentialWalk(pos, node, phi, theta=0.7):
    """Returns the gravitational field at position x by performing the Barnes-Hut treewalk using the provided KD-tree node

    Arguments:
    pos - (3,) array containing position of interest
    node - KD-tree to walk

    Keyword arguments:
    g - (3,) array containing initial value of the gravitational field, used when adding up the contributions in recursive calls
    softening - softening radius of the particle at which the force is being evaluated - needed if you want the short-range force to be momentum-conserving
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1\
% accuracy)
    """
    dx = node.COM[0]-pos[0]
    dy = node.COM[1]-pos[1]
    dz = node.COM[2]-pos[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)
    if node.IsLeaf:
        if r>0:
            phi += node.mass * PotentialKernel(r,node.h)
    elif r > max(node.size/theta, node.h+node.size):
        phi -= node.mass/r
    else:
        if node.HasLeft:
            phi = PotentialWalk(pos, node.left, phi, theta=theta)
        if node.HasRight:
            phi = PotentialWalk(pos, node.right, phi, theta=theta)
    return phi

@njit(fastmath=True)
def ForceWalk(pos, node, g, softening=0.0, theta=0.7):
    """Returns the gravitational field at position pos by performing the Barnes-Hut treewalk using the provided KD-tree node

    Arguments:
    pos - (3,) array containing position of interest
    node - KD-tree to walk

    Parameters:
    g - (3,) array containing initial value of the gravitational field, used when adding up the contributions in recursive calls
    softening - softening radius of the particle at which the force is being evaluated - needed if you want the short-range force to be momentum-conserving
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1\
% accuracy)
    """
    dx = node.COM[0]-pos[0]
    dy = node.COM[1]-pos[1]
    dz = node.COM[2]-pos[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)
    add_accel = False
    fac = 0
    if r>0:
        if node.IsLeaf:
            add_accel = True
            if r < max(node.h, softening):
                fac = node.mass * ForceKernel(r, max(node.h, softening))
            else:
                fac = node.mass/(r*r*r)
        elif r > max(node.size/theta + node.delta, node.h+node.size):
            add_accel = True  
            fac = node.mass/(r*r*r)

    if add_accel:
        g[0] += dx*fac
        g[1] += dy*fac
        g[2] += dz*fac
    else:
        if node.HasLeft:
            g = ForceWalk(pos, node.left, g, softening=softening, theta=theta)
        if node.HasRight:
            g = ForceWalk(pos, node.right, g, softening=softening, theta=theta)
    return g

@njit(parallel=True, fastmath=True)
def GetPotentialParallel(pos,tree, G, theta):
    result = empty(pos.shape[0])
    for i in prange(pos.shape[0]):
        result[i] = G*PotentialWalk(pos[i], tree, 0., theta=theta)
    return result

@njit(fastmath=True)
def GetPotential(pos,tree, G, theta):
    result = empty(pos.shape[0])
    for i in range(pos.shape[0]):
        result[i] = G*PotentialWalk(pos[i], tree, 0., theta=theta)
    return result

@njit(fastmath=True)
def GetAccel(pos, tree, softening=None, G=1., theta=0.7):
    if softening is None: softening = zeros(pos.shape[0])
    result = empty(pos.shape)
    for i in range(pos.shape[0]):
        result[i] = G*ForceWalk(pos[i], tree, zeros(3), softening=softening[i], theta=theta)
    return result

@njit(parallel=True, fastmath=True)
def GetAccelParallel(pos, tree, softening, G=1., theta=0.7):
    if softening is None: softening = zeros(len(pos), dtype=np.float64)    
    result = empty(pos.shape)
    for i in prange(pos.shape[0]):
        result[i] = G*ForceWalk(pos[i], tree, zeros(3), softening=softening[i], theta=theta)
    return result