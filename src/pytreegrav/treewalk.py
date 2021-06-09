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
            if r>0:  phi += tree.Masses[no] * PotentialKernel(r,h) # by default we neglect the self-potential
            no = tree.NextBranch[no]
        elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], h+tree.Sizes[no]): # if we satisfy the criteria for accepting the monopole
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
        elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], h+tree.Sizes[no]): # if we satisfy the criteria for accepting the monopole
            fac = tree.Masses[no]/(r*r2)
            sum_field = True
            no = tree.NextBranch[no] # go to the next branch in the tree
        else: # open the node
            no = tree.FirstSubnode[no]
            continue
            
        if sum_field: # OK, we have M(<R)/R^3 for this element and can now sum the force
            for k in range(3): g[k] += fac * dx[k]
            
    return g

@njit(fastmath=True)
def PotentialWalkRecursive(pos, tree, no=-1, softening=0,theta=1):
    """Returns the gravitational potential at pos by performing the Barnes-Hut treewalk using the provided octree structure

    Arguments:
    pos - (3,) array containing position of interest
    tree - octree object storing the tree structure    

    Keyword arguments:
    no - index of the tree node to do the walk for - defaults to starting with the top-level node
    softening - softening radius of the particle at which the force is being evaluated - needed if you want the short-range force to be momentum-conserving
    theta - cell opening angle used to control force accuracy; smaller is slower (runtime ~ theta^-3) but more accurate. (default 1.0, gives ~1% accuracy)
    """
    if no < 0: no = tree.NumParticles # default to the top-level node
    phi = 0.
    dx = tree.Coordinates[no,0]-pos[0]
    dy = tree.Coordinates[no,1]-pos[1]
    dz = tree.Coordinates[no,2]-pos[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)
    h = max(tree.Softenings[no],softening)

    if no < tree.NumParticles:
        if r==0: return 0. # by default we neglect the self-potential.
        return tree.Masses[no] * PotentialKernel(r,softening)
    elif r > max(tree.Sizes[no]/theta + tree.Deltas[no], h+tree.Sizes[no]):
        return -tree.Masses[no]/r # we can sum the monopole
    else: # we have to open the node
        for c in tree.children[no]: # loop over subnodes
            if c < 0:
                continue
            else:
                phi += PotentialWalk(pos, tree, c, softening,theta=theta) # add up the potential contribution you get for each subnode
    return phi

@njit(fastmath=True)
def ForceWalkRecursive(pos, tree, no=-1, softening=0,theta=1):
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
                c_force = ForceWalk(pos, tree, c, softening=softening, theta=theta) # add up the force contribution you get for each subnode
                force[0]+=c_force[0]
                force[1]+=c_force[1]
                force[2]+=c_force[2]
    return force 
