import numpy as np
from numba import njit
from numpy import zeros_like, sqrt
from .kernel import *
from .kdtree import *
from .treewalk import *


def Potential(pos, m, softening=None, G=1., theta=1., parallel=False, tree=None, return_tree=False):
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
    if softening is None:
        softening = np.zeros_like(m)
    if tree is None: tree = ConstructKDTree(np.float64(pos),np.float64(m), np.float64(softening))
    result = zeros(len(m))

    if parallel:
        pot = GetPotentialParallel(np.float64(pos),tree , G=G,theta=theta)
    else:
        pot = GetPotential(np.float64(pos), tree, G=G,theta=theta)
    if return_tree:
        return pot, tree
    else:
        return pot
    

def Accel(pos, m, softening=None, G=1., theta=1., parallel=False, tree=None, return_tree=False):
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
    
    if softening is None:
        softening = np.zeros_like(m)
    if not tree: tree = ConstructKDTree(np.float64(pos),np.float64(m), np.float64(softening))
    if parallel:
        acc = GetAccelParallel(np.float64(pos), tree, softening=softening, G=G, theta=theta)
    else:
        acc =  GetAccel(np.float64(pos), tree, softening=softening, G=G, theta=theta)
    if return_tree:
        return acc, tree
    else:
        return acc

@njit(fastmath=True)
def BruteForcePotentialTarget(x_target,x_source, m, softening=None,G=1.):
    """Returns the exact gravitational potential due to a set of particles, at a set of positions that need not be the same as the particle positions.

    Arguments:
    x_target -- shape (N,3) array of positions where the potential is to be evaluated
    x_source -- shape (M,3) array of positions of gravitating particles
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (M,) array containing kernel support radii for gravitational softening
    """    
    if softening is None: softening = np.zeros(x_target.shape[0])
    potential = np.zeros(x_target.shape[0])
    for i in range(x_target.shape[0]):
        for j in range(x_source.shape[0]):
            dx = x_target[i,0]-x_source[j,0]
            dy = x_target[i,1]-x_source[j,1]
            dz = x_target[i,2]-x_source[j,2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if r < softening[j]:
                potential[i] += m[j] * PotentialKernel(r, softening[j])
            else:
                if r>0: potential[i] -= m[j]/r
    return G*potential

@njit(fastmath=True)
def BruteForcePotential(x,m,softening=None,G=1.):
    """Returns the exact mutually-interacting gravitational potential for a set of particles with positions x and masses m.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (N,) array containing kernel support radii for gravitational softening
    """    
    if softening is None: softening = np.zeros_like(m)
    potential = zeros_like(m)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[i,0]-x[j,0]
            dy = x[i,1]-x[j,1]
            dz = x[i,2]-x[j,2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            rinv = 1/r
            if r < softening[i]:
                potential[j] += m[i] * PotentialKernel(r, softening[i])
            else:
                potential[j] -= m[i]*rinv
            if r < softening[j]:
                potential[i] += m[j] * PotentialKernel(r, softening[j])
            else:
                potential[i] -= m[j]*rinv
    return G*potential

@njit(fastmath=True)
def BruteForceAccel(x,m,softening=None,G=1.):
    """Returns the exact mutually-interacting gravitational accelerations of a set of particles.

    Arguments:
    x_target -- shape (N,3) array of positions where the potential is to be evaluated
    x_source -- shape (M,3) array of positions of gravitating particles
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    softening -- shape (M,) array containing kernel support radii for gravitational softening
    """
    if softening is None: softening = np.zeros_like(m)
    accel = zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            hmax = max(softening[i],softening[j]) # if there is overlap, we symmetrize the softenings to maintain momentum conservation
            dx = x[j,0]-x[i,0]
            dy = x[j,1]-x[i,1]
            dz = x[j,2]-x[i,2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            kernel = ForceKernel(r, hmax)            
            if r < hmax:
                fac = m[i]*kernel
            else:
                fac = m[i]/(r*r*r)
            accel[j,0] -= fac*dx
            accel[j,1] -= fac*dy
            accel[j,2] -= fac*dz

            if r < hmax:
                fac = m[j]*kernel
            else:
                fac = m[j]/(r*r*r)
            accel[i,0] += fac*dx
            accel[i,1] += fac*dy
            accel[i,2] += fac*dz
    return G*accel
    