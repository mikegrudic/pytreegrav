from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from .kernel import *
import numpy as np

@njit(fastmath=True)
def PotentialWalk(pos, tree, no=-1, softening=0,theta=1):
    """Returns the gravitational potential at position x by performing the Barnes-Hut treewalk using the provided octree structure

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
    phi = 0
    dx = tree.Coordinates[no,0]-pos[0]
    dy = tree.Coordinates[no,1]-pos[1]
    dz = tree.Coordinates[no,2]-pos[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)

    if no < tree.NumParticles:
        if r==0: return 0 # by default we neglect the self-potential
        return tree.Masses[no] * PotentialKernel(r,softening)
    elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], max(tree.Softenings[no],softening)+tree.Sizes[no]):
        return -tree.Masses[no]/r # we can sum the monopole
    else: # we have to open the node
        phi = 0
        for c in tree.children[no]: # loop over subnodes
            if c < 0:
                continue
            else:
                phi += PotentialWalk(pos, tree, c, softening,theta=theta) # add up the potential contribution you get for each subnode
    return phi
