from numba import njit
import numpy as np
from numpy import zeros_like, sqrt

@njit
def BruteForcePotential(x,m,G=1.):
    potential = zeros_like(m)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[i,0]-x[j,0]
            dy = x[i,1]-x[j,1]
            dz = x[i,2]-x[j,2]
            rinv = 1./sqrt(dx*dx + dy*dy + dz*dz)
            potential[i] += m[j]*rinv
            potential[j] += m[i]*rinv
    return -G*potential

@njit
def BruteForceAccel(x,m,G=1.):
    accel = zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[j,0]-x[i,0]
            dy = x[j,1]-x[i,1]
            dz = x[j,2]-x[i,2]
            r3inv = (1./sqrt(dx*dx + dy*dy + dz*dz))**3
            accel[i,0] += m[j]*dx*r3inv
            accel[i,1] += m[j]*dy*r3inv
            accel[i,2] += m[j]*dz*r3inv
            accel[j,0] -= m[i]*dx*r3inv
            accel[j,1] -= m[i]*dy*r3inv
            accel[j,2] -= m[i]*dz*r3inv
    return G*accel