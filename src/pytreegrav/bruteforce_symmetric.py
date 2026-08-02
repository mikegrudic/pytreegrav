"""Symmetry-exploiting parallel brute-force kernels.

The parallel routines in bruteforce.py give each thread one target particle i and only ever
write potential[i] / accel[i], so they must evaluate all N^2 pairs -- twice the work of the
serial upper-triangular loop.

Here each thread instead takes an interleaved subset of the upper-triangular *rows* (row i
covers j = i+1 .. N-1) and accumulates into its own private buffer, so both sides of every
interaction can be written without a race; the buffers are reduced at the end.  Rows are
dealt round-robin rather than in contiguous blocks because row i holds N-1-i interactions,
so interleaving balances the load with no explicit partitioning.

Cost: nthreads*N doubles of scratch for the potential, 3*nthreads*N for the accel.  That is
the price of the 2x flop saving -- at N=1e5 on 16 threads it's 13/38 MB, which is fine, but
it does not scale forever (N^2 brute force doesn't either).

An earlier version blocked the interaction matrix into tiles and handed threads (I,J) tile
pairs.  Benchmarked against this row decomposition with a byte-identical inner kernel, the
blocking was worth between -14% (2x Xeon 6244, 32t) and +9% (M-series, 16t, N>=40000): no
reliable gain, in exchange for a pair list growing as ntile^2 (19.6 MB at N=1e5) and a tile
-size knob.  The speedup here comes from the symmetry, the private accumulators that make
the symmetry safe under threading, and holding the i-side accumulators in registers -- not
from cache blocking.  The inner loop is scalar and divide-bound (LLVM refuses to vectorize
any loop carrying an fdiv, on both x86 and ARM), which leaves ample latency slack to cover
memory access, so there is little for blocking to recover.

Below SYMMETRIC_NMIN this loses to the untiled kernels anyway -- it runs two parallel
regions to their one, and a prange costs a full thread-team barrier however few iterations
it has -- so the frontend keeps small problems on bruteforce.py.
"""

import numpy as np
from numpy import sqrt
from numba import njit, prange, get_num_threads
from .kernel import PotentialKernel, ForceKernel

# Below this N the simple per-target parallel kernels in bruteforce.py win (two parallel
# regions vs one, and a prange costs a full thread-team barrier however small it is), so the
# frontend should keep using them.  Measured speedup of this module over those, pot/accel at
# each machine's default thread count:
#   M-series, 16t:      N=500 0.69/0.94   N=1000 0.99/1.48   N=2000 1.30/2.11
#   2x Xeon 6244, 32t:  N=500 0.89/0.93   N=1000 1.24/1.26   N=2000 1.61/1.48
# 1000 is the lowest value that is non-regressive on all four columns; 500 is not.  Accel
# crosses earlier than potential, but a second threshold is not worth the knob.
SYMMETRIC_NMIN = 1000


@njit(fastmath=True, inline="always")
def _potential_row(x, m, softening, i, out):
    """Accumulate the mutual potential of particle i with every j > i into out."""
    N = x.shape[0]
    xi = x[i, 0]
    yi = x[i, 1]
    zi = x[i, 2]
    hi = softening[i]
    mi = m[i]
    pot_i = 0.0
    for j in range(i + 1, N):
        dx = xi - x[j, 0]
        dy = yi - x[j, 1]
        dz = zi - x[j, 2]
        r = sqrt(dx * dx + dy * dy + dz * dz)
        h = max(hi, softening[j])
        if r < h:
            kernel = PotentialKernel(r, h)
        elif r > 0:
            kernel = -1.0 / r
        else:
            continue  # coincident and unsoftened: no self-potential
        pot_i += m[j] * kernel
        out[j] += mi * kernel
    out[i] += pot_i


@njit(fastmath=True, inline="always")
def _accel_row(x, m, softening, i, out):
    """Accumulate the mutual acceleration of particle i with every j > i into out."""
    N = x.shape[0]
    xi = x[i, 0]
    yi = x[i, 1]
    zi = x[i, 2]
    hi = softening[i]
    mi = m[i]
    ax = 0.0
    ay = 0.0
    az = 0.0
    for j in range(i + 1, N):
        dx = xi - x[j, 0]
        dy = yi - x[j, 1]
        dz = zi - x[j, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 == 0:
            continue
        r = sqrt(r2)
        h = max(hi, softening[j])
        # kernel is |a| / (m r), symmetric in i<->j, so one evaluation serves both sides
        if r < h:
            kernel = ForceKernel(r, h)
        else:
            kernel = 1.0 / (r2 * r)
        fi = kernel * m[j]
        fj = kernel * mi
        ax -= fi * dx
        ay -= fi * dy
        az -= fi * dz
        out[j, 0] += fj * dx
        out[j, 1] += fj * dy
        out[j, 2] += fj * dz
    out[i, 0] += ax
    out[i, 1] += ay
    out[i, 2] += az


@njit(fastmath=True, parallel=True)
def Potential_bruteforce_symmetric(x, m, softening, G=1.0):
    """Exact mutually-interacting gravitational potential, symmetrized + parallel.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of kernel support radii

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,) array containing potential values
    """
    N = x.shape[0]
    nchunk = get_num_threads()

    buf = np.zeros((nchunk, N))
    for t in prange(nchunk):
        for i in range(t, N, nchunk):
            _potential_row(x, m, softening, i, buf[t])

    potential = np.zeros_like(m)
    for i in prange(N):
        s = 0.0
        for t in range(nchunk):
            s += buf[t, i]
        potential[i] = G * s
    return potential


@njit(fastmath=True, parallel=True)
def Accel_bruteforce_symmetric(x, m, softening, G=1.0):
    """Exact mutually-interacting gravitational acceleration, symmetrized + parallel.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses
    softening -- shape (N,) array of softening lengths

    Optional arguments:
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3) array of gravitational accelerations
    """
    N = x.shape[0]
    nchunk = get_num_threads()

    buf = np.zeros((nchunk, N, 3))
    for t in prange(nchunk):
        for i in range(t, N, nchunk):
            _accel_row(x, m, softening, i, buf[t])

    accel = np.zeros_like(x)
    for i in prange(N):
        for k in range(3):
            s = 0.0
            for t in range(nchunk):
                s += buf[t, i, k]
            accel[i, k] = G * s
    return accel
