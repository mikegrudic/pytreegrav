from numba import njit
import numpy as np
from numpy import zeros_like, sqrt
from .kernel import *

@njit
def BruteForcePotential(x,m,h=None,G=1.):
    if h is None: h = np.zeros_like(m)
    potential = zeros_like(m)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[i,0]-x[j,0]
            dy = x[i,1]-x[j,1]
            dz = x[i,2]-x[j,2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            rinv = 1/r
            if r < h[i]:
                potential[j] += m[i] * PotentialKernel(r, h[i])
            else:
                potential[j] -= m[i]*rinv
            if r < h[j]:
                potential[i] += m[j] * PotentialKernel(r, h[j])
            else:
                potential[i] -= m[j]*rinv
    return G*potential

@njit
def BruteForceAccel(x,m,h=None,G=1.):
    if h is None: h = np.zeros_like(m)
    accel = zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[j,0]-x[i,0]
            dy = x[j,1]-x[i,1]
            dz = x[j,2]-x[i,2]
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if r < h[i]:
                mr3inv = m[i]*ForceKernel(r,h[i])
            else:
                mr3inv = m[i]/(r*r*r)
            accel[j,0] -= mr3inv*dx
            accel[j,1] -= mr3inv*dy
            accel[j,2] -= mr3inv*dz

            if r < h[j]:
                mr3inv = m[j]*ForceKernel(r,h[j])
            else:
                mr3inv = m[j]/(r*r*r)
            accel[i,0] += mr3inv*dx
            accel[i,1] += mr3inv*dy
            accel[i,2] += mr3inv*dz
    return G*accel