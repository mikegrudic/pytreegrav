"""Symmetric AND vectorized direct summation, with softening.

Supersedes bruteforce_avx.py, which gave up the i<->j symmetry to obtain vectorization and
could not handle softening at all. Both restrictions turned out to be avoidable.

Three obstacles had to be removed, each measured separately on 2x Xeon Gold 6244 (AVX2,
32 threads), numba 0.66:

1. The divide. LLVM's loop vectorizer refuses any loop carrying an ``fdiv``. Removing it from
   an otherwise identical loop gives 24 packed ops and a 5.1x speedup (53.6 -> 10.5 us per
   20k interactions). Replaced by a Newton-Raphson reciprocal square root (multiplies only).

2. The runtime loop lower bound. This was the subtle one, and the reason the earlier attempt
   concluded symmetry and SIMD were incompatible. What blocks numba's vectorizer is the
   *combination* of a runtime lower bound and a store:

       for j in range(0, n)   + store  ->  60 packed ops   vectorizes
       for j in range(i0, n)  + store  ->   0 packed ops   DOES NOT
       for j in range(i0, n)  no store ->  56 packed ops   vectorizes
       slice so the loop starts at 0   ->  60 packed ops   vectorizes

   Either alone is fine. The symmetric sweep needs ``range(i+1, n)`` -- exactly the bad case.
   Passing the row kernel a *slice* starting at i+1 makes its loop run from zero, at the cost
   of one array-struct construction per row (O(N), negligible against O(N^2) interactions).

3. The softening branch. ``if r < h`` is data-dependent and blocks vectorization on its own.
   :func:`_force_kernel` below evaluates all three spline pieces and blends them with selects,
   which LLVM if-converts. Its ``0.0667/q^3`` term needs no divide because ``1/q = h*rinv``
   and ``rinv`` is already in hand; ``1/h`` likewise comes from ``rsqrt(h*h)``.

None of this is an LLVM limitation: the same loop compiled from C++ with clang vectorizes
with or without ``__restrict`` (``__restrict`` changes nothing -- identical packed-op counts,
bit-identical output). With the slice trick numba reaches 415 ms single-threaded at N=20000
against 390 ms for the equivalent C++, within 7%.

Measured against Accel_bruteforce_symmetric, 32 threads, 4 Newton steps (max relative error
vs the exact serial reference in parentheses):

    N       unsoftened                softened h=0.05
     5000    5.9 ->  1.6 ms  3.65x     4.1 ->  3.1 ms  1.32x
    20000   61.2 -> 21.4 ms  2.86x    61.1 -> 44.7 ms  1.37x
    50000  402.3 ->166.0 ms  2.42x   412.4 ->298.5 ms  1.38x
                    (2.3e-15)                 (4.1e-15)

Momentum is still conserved to roundoff (1.1e-17), so the symmetry is genuinely preserved.

The softened path gains less because every lane pays for all three spline pieces whether or
not the pair is within the softening radius -- roughly 3x the arithmetic against 4x the
lanes -- and because supporting h == 0 inside the same kernel costs a further ~20% (a
standalone version that assumes h > 0 everywhere measured 1.68x rather than 1.37x). That is
why the unsoftened kernel is kept separate and dispatched to whenever softening is absent or
identically zero. The cost is flat in h: 1.64x at h=0.1 vs 1.67x at h=0.01 for the
h>0-only variant.

ARM/NEON (Apple M-series, 16 threads) is a different story, and this module must NOT be
wired in unconditionally:

    N       unsoftened          softened h=0.05
     5000    3.2 ->   2.9 ms  1.10x    3.2 ->   6.9 ms  0.46x
    20000   40.8 ->  32.0 ms  1.28x   38.5 ->  82.8 ms  0.47x
    50000  212.6 -> 181.6 ms  1.17x  222.1 -> 482.2 ms  0.46x

(Two independent runs agreed to within 0.06x on every entry. An earlier pass on a busier
machine reported 1.18-1.33x / 0.50-0.57x; the background load was slowing the *shipped*
kernel and flattering the comparison, so these lower figures are the honest ones.)

Both paths still vectorize there (99 packed .2d ops) and stay accurate to ~4e-15, so this is
not a codegen failure -- NEON is 2-wide, so the ceiling is 2x before the Newton overhead is
paid. The unsoftened path barely clears that bar (and still beats bruteforce_avx's 0.76x,
precisely because the symmetry is kept); the softened path is roughly 2.2x SLOWER, since the
branchless spline costs ~3x the arithmetic and two lanes cannot cover it.

So: x86 wins on both paths, ARM wins only unsoftened and only by ~1.1-1.3x. Any dispatch needs to
gate on vector width, not just on whether softening is present.
"""

import numpy as np
from llvmlite import ir
from numba import get_num_threads, njit, prange, types
from numba.extending import intrinsic

# float64 analogue of the Quake III 0x5f3759df seed constant
_RSQRT_MAGIC = 0x5FE6EB50C7B537A9

# Newton refinements in the reciprocal sqrt. Each doubles the correct bits (~5 from the seed:
# 5 -> 10 -> 20 -> 40 -> 80), so 4 covers float64's 53. Measured at N=20000 against the
# shipped symmetric kernel, as (speedup unsoftened / softened, max relative error):
#     1 step   3.71x / 2.31x   3e-03     never worth it: step 2 costs +5% for 3 more orders
#     2 steps  3.53x / 2.03x   6e-06
#     3 steps  3.11x / 1.89x   4e-11     good trade unless you need a bit-level reference
#     4 steps  2.88x / 1.68x   4e-15     full double precision (default)
_NEWTON_STEPS = 4


@intrinsic
def _as_i64(typingctx, x):
    """Reinterpret a float64's bits as int64. numba has no scalar equivalent of .view()."""
    if x != types.float64:
        return

    def codegen(context, builder, sig, args):
        """llvmlite codegen callback emitting the bitcast."""
        return builder.bitcast(args[0], ir.IntType(64))

    return types.int64(types.float64), codegen


@intrinsic
def _as_f64(typingctx, x):
    """Inverse of :func:`_as_i64`."""
    if x != types.int64:
        return

    def codegen(context, builder, sig, args):
        """llvmlite codegen callback emitting the bitcast."""
        return builder.bitcast(args[0], ir.DoubleType())

    return types.float64(types.int64), codegen


@njit(fastmath=True, inline="always")
def _rsqrt(x):
    """1/sqrt(x) from multiplies only, so the enclosing loop stays vectorizable.

    Magic-constant bit-trick seed plus :data:`_NEWTON_STEPS` Newton-Raphson refinements.
    Using sqrt/divide here would reintroduce the fdiv that LLVM refuses to vectorize.

    The loop is over a module-level constant, so numba unrolls it at compile time -- editing
    _NEWTON_STEPS is all that is needed to change the accuracy/speed trade.
    """
    y = _as_f64(np.int64(_RSQRT_MAGIC) - (_as_i64(x) >> 1))
    xhalf = 0.5 * x
    for _ in range(_NEWTON_STEPS):
        y = y * (1.5 - xhalf * y * y)
    return y


@njit(fastmath=True, inline="always")
def _force_kernel(r2, rinv, h):
    """M4 cubic-spline force kernel |a|/(m r), branchless and divide-free.

    Equivalent to ``ForceKernel(r, h) if r < h else 1/r**3`` from pytreegrav.kernel, matched to
    9.5e-15 relative over q = 1e-3 .. 10, but with no branches (all three pieces are evaluated
    and blended by selects) and no divisions:

      * ``1/h`` comes from ``rsqrt(h*h)`` rather than a divide;
      * the ``0.0667/q**3`` term uses ``1/q = h*rinv``, which needs no divide either.

    h == 0 is handled explicitly: with hinv == 0 the ``q`` test would otherwise select the
    innermost spline piece and return 0 rather than the Newtonian 1/r**3.

    Arguments:
    r2 -- squared separation
    rinv -- 1/r, already computed by the caller (see :func:`_rsqrt`)
    h -- symmetrized softening length, max(h_i, h_j); may be 0
    """
    hinv = _rsqrt(h * h) if h > 0.0 else 0.0
    q = r2 * rinv * hinv  # r/h
    qinv = h * rinv  # h/r == 1/q, no divide needed
    h3 = hinv * hinv * hinv
    q2 = q * q
    far = rinv * rinv * rinv  # q > 1, or unsoftened
    near = (10.666666666666666666 + q2 * (-38.4 + 32.0 * q)) * h3  # q <= 0.5
    mid = (
        21.333333333333 - 48.0 * q + 38.4 * q2 - 10.666666666667 * q2 * q - 0.066666666667 * qinv * qinv * qinv
    ) * h3  # 0.5 < q <= 1
    inner = near if q <= 0.5 else mid
    return inner if (h > 0.0 and q <= 1.0) else far


@njit(fastmath=True, inline="always")
def _accel_row_soft(px, py, pz, m, sf, ox, oy, oz, mi, hi, xi, yi, zi):
    """One particle's mutual interactions with every element of the given slices, softened.

    The caller passes slices beginning at i+1 so this loop runs ``range(0, len)``; with a
    runtime lower bound the stores below would stop it vectorizing.

    Arguments:
    px, py, pz -- contiguous slices of the source coordinates, from i+1 onward
    m, sf -- matching slices of source masses and softening lengths
    ox, oy, oz -- matching slices of this thread's accumulator, written in place (the j side)
    mi, hi -- mass and softening of particle i
    xi, yi, zi -- position of particle i

    Returns:
    (ax, ay, az) -- particle i's own accumulated acceleration, for the caller to add
    """
    ax = 0.0
    ay = 0.0
    az = 0.0
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        # select, not a branch: coincident particles contribute nothing, and this costs
        # nothing measurable (27.1 vs 26.7 us) unlike the softening branch it replaces
        rinv = _rsqrt(r2) if r2 > 0 else 0.0
        k = _force_kernel(r2, rinv, max(hi, sf[j]))
        qi = k * m[j]
        qj = k * mi
        ax += qi * dx
        ay += qi * dy
        az += qi * dz
        ox[j] -= qj * dx
        oy[j] -= qj * dy
        oz[j] -= qj * dz
    return ax, ay, az


@njit(fastmath=True, inline="always")
def _accel_row_plain(px, py, pz, m, ox, oy, oz, mi, xi, yi, zi):
    """As :func:`_accel_row_soft` but Newtonian only, skipping the spline entirely.

    Worth a separate kernel: the branchless spline costs roughly 3x the arithmetic even when
    no pair is actually softened, which is the difference between 2.88x and 1.68x.
    """
    ax = 0.0
    ay = 0.0
    az = 0.0
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        rinv = _rsqrt(r2) if r2 > 0 else 0.0
        q = rinv * rinv * rinv
        qi = q * m[j]
        qj = q * mi
        ax += qi * dx
        ay += qi * dy
        az += qi * dz
        ox[j] -= qj * dx
        oy[j] -= qj * dy
        oz[j] -= qj * dz
    return ax, ay, az


@njit(fastmath=True, parallel=True)
def _accel_core_soft(px, py, pz, m, sf, G):
    """Softened symmetric upper-triangular sweep. SoA in, (N,3) out.

    Rows are dealt round-robin to threads: row i holds n-1-i interactions, so interleaving
    balances the load without explicit partitioning. Each thread owns a private accumulator
    so both sides of every pair can be written without a race.
    """
    n = px.shape[0]
    nchunk = get_num_threads()
    bx = np.zeros((nchunk, n))
    by = np.zeros((nchunk, n))
    bz = np.zeros((nchunk, n))
    for t in prange(nchunk):
        for i in range(t, n - 1, nchunk):
            ax, ay, az = _accel_row_soft(
                px[i + 1 :],
                py[i + 1 :],
                pz[i + 1 :],
                m[i + 1 :],
                sf[i + 1 :],
                bx[t][i + 1 :],
                by[t][i + 1 :],
                bz[t][i + 1 :],
                m[i],
                sf[i],
                px[i],
                py[i],
                pz[i],
            )
            bx[t][i] += ax
            by[t][i] += ay
            bz[t][i] += az
    out = np.zeros((n, 3))
    for i in prange(n):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for t in range(nchunk):
            sx += bx[t, i]
            sy += by[t, i]
            sz += bz[t, i]
        out[i, 0] = G * sx
        out[i, 1] = G * sy
        out[i, 2] = G * sz
    return out


@njit(fastmath=True, parallel=True)
def _accel_core_plain(px, py, pz, m, G):
    """Unsoftened symmetric sweep; see :func:`_accel_core_soft` for the threading scheme."""
    n = px.shape[0]
    nchunk = get_num_threads()
    bx = np.zeros((nchunk, n))
    by = np.zeros((nchunk, n))
    bz = np.zeros((nchunk, n))
    for t in prange(nchunk):
        for i in range(t, n - 1, nchunk):
            ax, ay, az = _accel_row_plain(
                px[i + 1 :],
                py[i + 1 :],
                pz[i + 1 :],
                m[i + 1 :],
                bx[t][i + 1 :],
                by[t][i + 1 :],
                bz[t][i + 1 :],
                m[i],
                px[i],
                py[i],
                pz[i],
            )
            bx[t][i] += ax
            by[t][i] += ay
            bz[t][i] += az
    out = np.zeros((n, 3))
    for i in prange(n):
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for t in range(nchunk):
            sx += bx[t, i]
            sy += by[t, i]
            sz += bz[t, i]
        out[i, 0] = G * sx
        out[i, 1] = G * sy
        out[i, 2] = G * sz
    return out


def _to_soa(x):
    """Split an (N,3) array into three contiguous columns.

    O(N), and required: the (N,3) layout makes each component a stride-3 access, which the
    vectorizer handles far worse than unit stride.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    return (
        np.ascontiguousarray(x[:, 0]),
        np.ascontiguousarray(x[:, 1]),
        np.ascontiguousarray(x[:, 2]),
    )


def Accel_bruteforce_sym_vec(x, m, softening=None, G=1.0):
    """Exact mutually-interacting gravitational acceleration: symmetric and vectorized.

    Handles softened and unsoftened inputs. When softening is absent or identically zero the
    Newtonian-only kernel is used, which is ~1.7x faster than the general one, so passing an
    all-zero array costs nothing.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Optional arguments:
    softening -- shape (N,) array of M4 cubic-spline support radii, or None for unsoftened
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3) array of gravitational accelerations. Accurate to ~4e-15 (unsoftened) or
    ~1.4e-14 (softened) relative to the exact serial reference; the reciprocal sqrt is not
    bit-exact, so this is not a drop-in where 1e-12 agreement is asserted.
    """
    px, py, pz = _to_soa(x)
    m = np.ascontiguousarray(m, dtype=np.float64)
    if softening is None:
        return _accel_core_plain(px, py, pz, m, G)
    sf = np.ascontiguousarray(softening, dtype=np.float64)
    if not np.any(sf):  # all zero -> the cheaper Newtonian kernel is exact here
        return _accel_core_plain(px, py, pz, m, G)
    return _accel_core_soft(px, py, pz, m, sf, G)
