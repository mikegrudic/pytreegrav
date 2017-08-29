import numpy as np
from .kdtree import *
from .treewalk import *

def Potential(x, m, softening=None, G=1., theta=1., parallel=False):
    """Returns the approximate gravitational potential for a set of particles with positions x and masses m.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    theta -- cell opening angle used to control force accuracy; smaller is faster but more accurate. (default 1.0, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False)
    """
    if softening is None:
        softening = np.zeros_like(m)
    tree = ConstructKDTree(np.float64(x),np.float64(m), np.float64(softening))
    result = zeros(len(m))
    if parallel:
        return GetPotentialParallel(np.float64(x),tree,G,theta)
    else:
        return GetPotential(np.float64(x),tree,G,theta)

def Accel(x, m, softening=None, G=1., theta=1., parallel=False):
    """Returns the approximate gravitational acceleration for a set of particles with positions x and masses m.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    theta -- cell opening angle used to control force accuracy; smaller is faster but more accurate. (default 1.0, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False) """
    
    if softening is None:
        softening = np.zeros_like(m)
    tree = ConstructKDTree(np.float64(x),np.float64(m), np.float64(softening))
    result = zeros_like(x)
    if parallel:
        return GetAccelParallel(np.float64(x), tree, G, theta)
    else:
        return GetAccel(np.float64(x), tree, G, theta)

def CorrelationFunction(x, m, rbins, frac=1.):
    N = len(x)
    tree = ConstructKDTree(np.float64(x), np.float64(m))
    counts = zeros(len(rbins)-1, dtype=np.int64)
    for i in range(N):
        if np.random.rand() < frac:
            CorrelationWalk(counts, rbins, np.float64(x[i]), tree)

    return counts / (4*np.pi/3 * np.diff(rbins**3)) / frac

