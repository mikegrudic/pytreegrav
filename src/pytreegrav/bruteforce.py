import numpy as np
from numpy import sqrt, empty, zeros, empty_like, zeros_like
from numba import njit, prange
from .kernel import *


def PotentialTarget_bruteforce(x_target, softening_target, x_source, m_source, softening_source, G=1.0):
    """Returns the exact gravitational potential due to a set of particles, at a set of positions that need not be the same as the particle positions.

    Arguments:
    x_target -- shape (N,3) array of positions where the potential is to be evaluated
    softening_target -- shape (N,) array of minimum softening lengths to be used
    x_source -- shape (M,3) array of positions of gravitating particles
    m_source -- shape (M,) array of particle masses
    softening_source -- shape (M,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,) array of potential values
    """
    potential = np.zeros(x_target.shape[0])
    for i in prange(x_target.shape[0]):
        dx = np.empty(3)  # allocate inside prange so it is thread-private by construction
        for j in range(x_source.shape[0]):
            for k in range(3):
                dx[k] = x_target[i, k] - x_source[j, k]
            r = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])
            if r == 0:
                # A target sitting exactly on a source contributes nothing. The treewalk and
                # AccelTarget_bruteforce both skip r==0, and Potential_bruteforce excludes it
                # by construction; without this the tree and bruteforce methods disagree by
                # m*PotentialKernel(0,h), so 'adaptive' silently changes the answer with N.
                continue

            h = max(softening_source[j], softening_target[i])
            if r < h:
                potential[i] += m_source[j] * PotentialKernel(r, h)
            else:
                potential[i] -= m_source[j] / r
    return G * potential


# NOTE: no cache=True on these four.  Both variants are njit'd from the SAME Python
# function, so they share one on-disk cache entry (keyed by qualname+lineno) and the
# parallel dispatcher silently loads the serial code -- measured 8.4x -> 1.05x, with no
# warning.  The decorator-form kernels above/below are separate functions and cache fine.
PotentialTarget_bruteforce_parallel = njit(PotentialTarget_bruteforce, fastmath=True, parallel=True)
PotentialTarget_bruteforce = njit(PotentialTarget_bruteforce, fastmath=True)


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
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


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, parallel=True, cache=True)
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


def AccelTarget_bruteforce(x_target, softening_target, x_source, m_source, softening_source, G=1.0):
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


# NOTE: no cache=True on these four.  Both variants are njit'd from the SAME Python
# function, so they share one on-disk cache entry (keyed by qualname+lineno) and the
# parallel dispatcher silently loads the serial code -- measured 8.4x -> 1.05x, with no
# warning.  The decorator-form kernels above/below are separate functions and cache fine.
AccelTarget_bruteforce_parallel = njit(AccelTarget_bruteforce, fastmath=True, parallel=True)
AccelTarget_bruteforce = njit(AccelTarget_bruteforce, fastmath=True)


@njit(inline="always", fastmath=True)
def _add_tidal_pair(T, i, m, dx0, dx1, dx2, r, r2, h):
    """Accumulate the tidal tensor of mass ``m`` at separation ``dx`` into T[i].

    T_ij = m (Q dx_i dx_j - K delta_ij), with K = M(<r)/(m r^3) and Q = -K'/r.  dx points from the target to the source, but both terms are even in dx, so a pair's contribution to each of its two members differs only by which mass multiplies it -- which is what lets the mutual kernel below symmetrize.
    """
    if r < h:
        K = m * ForceKernel(r, h)
        Q = m * TidalKernel(r, h)
    else:
        K = m / (r * r2)
        Q = 3.0 * K / r2
    T[i, 0, 0] += Q * dx0 * dx0 - K
    T[i, 1, 1] += Q * dx1 * dx1 - K
    T[i, 2, 2] += Q * dx2 * dx2 - K
    txy = Q * dx0 * dx1
    txz = Q * dx0 * dx2
    tyz = Q * dx1 * dx2
    T[i, 0, 1] += txy
    T[i, 1, 0] += txy
    T[i, 0, 2] += txz
    T[i, 2, 0] += txz
    T[i, 1, 2] += tyz
    T[i, 2, 1] += tyz


@njit(fastmath=True, cache=True)
def TidalTensor_bruteforce(x, m, softening, G=1.0):
    """Returns the exact mutually-interacting tidal tensor for a set of particles with positions x and masses m, evaluated by brute force.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array containing kernel support radii for gravitational softening

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3,3) array of tidal tensors
    """
    if softening is None:
        softening = np.zeros_like(m)
    T = np.zeros((x.shape[0], 3, 3))
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            # symmetrize the softenings if there is overlap, as in Accel_bruteforce
            h = max(softening[i], softening[j])
            dx0 = x[i, 0] - x[j, 0]
            dx1 = x[i, 1] - x[j, 1]
            dx2 = x[i, 2] - x[j, 2]
            r2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
            if r2 == 0:
                continue
            r = sqrt(r2)
            _add_tidal_pair(T, i, m[j], dx0, dx1, dx2, r, r2, h)
            _add_tidal_pair(T, j, m[i], dx0, dx1, dx2, r, r2, h)
    return G * T


@njit(fastmath=True, parallel=True, cache=True)
def TidalTensor_bruteforce_parallel(x, m, softening, G=1.0):
    """Returns the exact mutually-interacting tidal tensor for a set of particles with positions x and masses m, evaluated by brute force.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array containing kernel support radii for gravitational softening

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3,3) array of tidal tensors
    """
    if softening is None:
        softening = np.zeros_like(m)
    T = np.zeros((x.shape[0], 3, 3))
    for i in prange(x.shape[0]):
        for j in range(x.shape[0]):
            if i == j:
                continue
            h = max(softening[i], softening[j])
            dx0 = x[j, 0] - x[i, 0]
            dx1 = x[j, 1] - x[i, 1]
            dx2 = x[j, 2] - x[i, 2]
            r2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
            if r2 == 0:
                continue
            r = sqrt(r2)
            _add_tidal_pair(T, i, m[j], dx0, dx1, dx2, r, r2, h)
    return G * T


def TidalTensorTarget_bruteforce(x_target, softening_target, x_source, m_source, softening_source, G=1.0):
    """Returns the exact tidal tensor at a set of target positions, due to a set of source particles.

    Arguments:
    x_target -- shape (N,3) array of positions where the tidal tensor is to be evaluated
    softening_target -- shape (N,) array of minimum softening lengths to be used
    x_source -- shape (M,3) array of positions of gravitating particles
    m_source -- shape (M,) array of particle masses
    softening_source -- shape (M,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3,3) array of tidal tensors
    """
    T = np.zeros((x_target.shape[0], 3, 3))
    for i in prange(x_target.shape[0]):
        for j in range(x_source.shape[0]):
            h = max(softening_target[i], softening_source[j])
            dx0 = x_source[j, 0] - x_target[i, 0]
            dx1 = x_source[j, 1] - x_target[i, 1]
            dx2 = x_source[j, 2] - x_target[i, 2]
            r2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
            if r2 == 0:
                continue  # a target sitting exactly on a source contributes nothing
            r = sqrt(r2)
            _add_tidal_pair(T, i, m_source[j], dx0, dx1, dx2, r, r2, h)
    return G * T


# see the caching note above: both variants come from one Python function, so no cache=True
TidalTensorTarget_bruteforce_parallel = njit(TidalTensorTarget_bruteforce, fastmath=True, parallel=True)
TidalTensorTarget_bruteforce = njit(TidalTensorTarget_bruteforce, fastmath=True)
