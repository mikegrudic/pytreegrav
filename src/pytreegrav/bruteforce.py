import numpy as np
from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from .kernel import *


def PotentialTarget_bruteforce(
    x_target, softening_target, x_source, m_source, softening_source, G=1.0
):
    """Returns the exact gravitational potential due to a set of particles, at a set of positions that need not be the same as the particle positions.

    Arguments:
    x_target -- shape (N,3) array of positions where the potential is to be evaluated
    softening_target -- shape (N,) array of minimum softening lengths to be used
    x_source -- shape (M,3) array of positions of gravitating particles
    m_source -- shape (M,) array of particle masses
    softening_source -- shape (M,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 0.7)

    Returns:
    shape (N,) array of potential values
    """
    potential = np.zeros(x_target.shape[0])
    dx = np.empty(3)
    for i in prange(x_target.shape[0]):
        for j in range(x_source.shape[0]):
            for k in range(3):
                dx[k] = x_target[i, k] - x_source[j, k]
            r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])

            h = max(softening_source[j], softening_target[i])
            if r < h:
                potential[i] += m_source[j] * PotentialKernel(r, h)
            else:
                if r > 0:
                    potential[i] -= m_source[j] / r
    return G * potential


PotentialTarget_bruteforce_parallel = njit(
    PotentialTarget_bruteforce, fastmath=True, parallel=True
)
PotentialTarget_bruteforce = njit(PotentialTarget_bruteforce, fastmath=True)


@njit(fastmath=True)
def Potential_bruteforce(x, m, softening, G=1.0):
    """Returns the exact mutually-interacting gravitational potential for a set of particles with positions x and masses m, evaluated by brute force.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array containing kernel support radii for gravitational softening

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,) array containing potential values
    """
    potential = zeros_like(m)
    dx = zeros(3)
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            for k in range(3):
                dx[k] = x[i, k] - x[j, k]
            r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])
            h = max(softening[i], softening[j])
            if r < h:
                kernel = PotentialKernel(r, h)
                potential[j] += m[i] * kernel
                potential[i] += m[j] * kernel
            elif r > 0:
                potential[i] -= m[j] / r
                potential[j] -= m[i] / r
    return G * potential


@njit(fastmath=True, parallel=True)
def Potential_bruteforce_parallel(x, m, softening, G=1.0):
    """Returns the exact mutually-interacting gravitational potential for a set of particles with positions x and masses m, evaluated by brute force.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array containing kernel support radii for gravitational softening

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,) array containing potential values
    """
    potential = zeros_like(m)
    for i in prange(x.shape[0]):
        dx = zeros(3)
        for j in range(x.shape[0]):
            if i == j:
                continue  # neglect self-potential
            for k in range(3):
                dx[k] = x[i, k] - x[j, k]
            r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])
            h = max(softening[i], softening[j])
            if r < h:
                kernel = PotentialKernel(r, h)
                potential[i] += m[j] * kernel
            elif r > 0:
                potential[i] -= m[j] / r
    return G * potential


@njit(fastmath=True)
def Accel_bruteforce(x, m, softening, G=1.0):
    """Returns the exact mutually-interacting gravitational accelerations of a set of particles.

    Arguments:
    x -- shape (N,3) array of positions where the potential is to be evaluated
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3) array of gravitational accelerations
    """
    if softening is None:
        softening = np.zeros_like(m)
    accel = zeros_like(x)
    dx = zeros(3)
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            h = max(
                softening[i], softening[j]
            )  # if there is overlap, we symmetrize the softenings to maintain momentum conservation
            r2 = 0
            for k in range(3):
                dx[k] = x[i, k] - x[j, k]
                r2 += dx[k] * dx[k]
            if r2 == 0:
                continue
            r = sqrt(r2)

            if r < h:
                kernel = ForceKernel(r, h)
                for k in range(3):
                    accel[j, k] += kernel * m[i] * dx[k]
                    accel[i, k] -= kernel * m[j] * dx[k]
            else:
                fac = 1 / (r2 * r)
                for k in range(3):
                    accel[j, k] += m[i] * fac * dx[k]
                    accel[i, k] -= m[j] * fac * dx[k]
    return G * accel


@njit(fastmath=True, parallel=True)
def Accel_bruteforce_parallel(x, m, softening, G=1.0):
    """Returns the exact mutually-interacting gravitational accelerations of a set of particles.

    Arguments:
    x -- shape (N,3) array of positions where the potential is to be evaluated
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3) array of gravitational accelerations
    """
    if softening is None:
        softening = np.zeros_like(m)
    accel = zeros_like(x)
    for i in prange(x.shape[0]):
        dx = zeros(3)
        for j in range(x.shape[0]):
            if i == j:
                continue
            h = max(
                softening[i], softening[j]
            )  # if there is overlap, we symmetrize the softenings to maintain momentum conservation
            r2 = 0
            for k in range(3):
                dx[k] = x[j, k] - x[i, k]
                r2 += dx[k] * dx[k]
            if r2 == 0:
                continue
            r = sqrt(r2)

            if r < h:
                kernel = ForceKernel(r, h)
                for k in range(3):
                    accel[i, k] += kernel * m[j] * dx[k]
            else:
                fac = 1 / (r2 * r)
                for k in range(3):
                    accel[i, k] += m[j] * fac * dx[k]
    return G * accel


def AccelTarget_bruteforce(
    x_target, softening_target, x_source, m_source, softening_source, G=1.0
):
    """Returns the gravitational acceleration at a set of target positions, due to a set of source particles.

    Arguments:
    x_target -- shape (N,3) array of positions where the field is to be evaluated
    softening_target -- shape (N,) array of minimum softening lengths to be used
    x_source -- shape (M,3) array of positions of gravitating particles
    m_source -- shape (M,) array of particle masses
    softening_source -- shape (M,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3) array of gravitational accelerations
    """
    accel = zeros_like(x_target)
    for i in prange(x_target.shape[0]):
        dx = zeros(3)
        for j in range(x_source.shape[0]):
            h = max(
                softening_target[i], softening_source[j]
            )  # if there is overlap, we symmetrize the softenings to maintain momentum conservation
            r2 = 0
            for k in range(3):
                dx[k] = x_source[j, k] - x_target[i, k]
                r2 += dx[k] * dx[k]
            if r2 == 0:
                continue  # no force if at the origin
            r = sqrt(r2)

            if r < h:
                kernel = ForceKernel(r, h)
                for k in range(3):
                    accel[i, k] += kernel * m_source[j] * dx[k]
            else:
                fac = 1 / (r2 * r)
                for k in range(3):
                    accel[i, k] += m_source[j] * fac * dx[k]
    return G * accel


AccelTarget_bruteforce_parallel = njit(
    AccelTarget_bruteforce, fastmath=True, parallel=True
)
AccelTarget_bruteforce = njit(AccelTarget_bruteforce, fastmath=True)
