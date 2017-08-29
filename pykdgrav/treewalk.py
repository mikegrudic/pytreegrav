from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from .kernel import *

@njit
def PotentialWalk(x, phi, node, theta=0.7):
    r = sqrt((x[0]-node.COM[0])**2 + (x[1]-node.COM[1])**2 + (x[2]-node.COM[2])**2)
    if node.IsLeaf:
        if r>0:
            phi += node.mass * PotentialKernel(r,node.h)
    elif r > max(node.size/theta, node.h+node.size):
        phi -= node.mass/r
    else:
        if node.HasLeft:
            phi = PotentialWalk(x, phi, node.left, theta)
        if node.HasRight:
            phi = PotentialWalk(x, phi, node.right,  theta)
    return phi

@njit
def ForceWalk(x, g, node, theta=0.7):
    dx = node.COM[0]-x[0]
    dy = node.COM[1]-x[1]
    dz = node.COM[2]-x[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)
    add_accel = False
    if r>0:
        if node.IsLeaf:
            add_accel = True
            if r < node.h:
                mr3inv = node.mass * ForceKernel(r, node.h)
            else:
                mr3inv = node.mass/(r*r*r)
        elif r > max(node.size/theta + node.delta, node.h+node.size):
            add_accel = True        
            mr3inv = node.mass/(r*r*r)

    if add_accel:
        g[0] += dx*mr3inv
        g[1] += dy*mr3inv
        g[2] += dz*mr3inv
    else:
        if node.HasLeft:
            g = ForceWalk(x, g, node.left, theta)
        if node.HasRight:
            g = ForceWalk(x, g, node.right, theta)
    return g

@njit
def CorrelationWalk(counts, rbins, x, node):
    #idea: if the center of the node is in a bin and the bounds also lie in the same bin, add to that bin. If all bounds are outside all bins, return 0. Else,repeat for children
    dx = 0.5*(node.bounds[0,0]+node.bounds[0,1])-x[0]
    dy = 0.5*(node.bounds[1,0]+node.bounds[1,1])-x[1]
    dz = 0.5*(node.bounds[2,0]+node.bounds[2,1])-x[2]
    r = (dx**2 + dy**2 + dz**2)**0.5

    sizebound = node.size*1.73
    rmin, rmax = r-sizebound/2, r+sizebound/2
    if rmin > rbins[-1]:
        return
    if rmax < rbins[0]:
        return

    N = rbins.shape[0]

    for i in range(1,N):
        if rbins[i] > r: break
        
    if rbins[i] > rmax and rbins[i-1] < rmin:
        counts[i-1] += node.Npoints
    else:
        if node.HasLeft:
            CorrelationWalk(counts, rbins, x, node.left)
        if node.HasRight:
            CorrelationWalk(counts, rbins, x, node.right)
    return

@njit(parallel=True)
def GetPotentialParallel(x,tree, G, theta):
    result = empty(x.shape[0])
    for i in prange(x.shape[0]):
        result[i] = G*PotentialWalk(x[i],0.,tree,theta)
    return result

@njit
def GetPotential(x,tree, G, theta):
    result = empty(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = G*PotentialWalk(x[i],0.,tree, theta)
    return result

@njit
def GetAccel(x, tree, G, theta):
    result = empty(x.shape)
    for i in range(x.shape[0]):
        result[i] = G*ForceWalk(x[i], zeros(3), tree, theta)
    return result

@njit(parallel=True)
def GetAccelParallel(x, tree, G, theta):
    result = empty(x.shape)
    for i in prange(x.shape[0]):
        result[i] = G*ForceWalk(x[i], zeros(3), tree, theta)
    return result