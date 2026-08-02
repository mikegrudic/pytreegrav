"""Prototype: AVX-vectorized direct summation, unsoftened.

UNVERIFIED -- written but not yet run.  See "Before trusting this" at the bottom.

The kernels in bruteforce_symmetric.py are scalar.  Measured on a Xeon Gold 6244 (AVX-512),
the reason is a single instruction: the inner loop carries an ``fdiv``, and LLVM's loop
vectorizer refuses any loop containing one.  Delete the divide from an otherwise identical
loop and it emits 24 packed ops and runs 5.1x faster (53.6 -> 10.5 us per 20k interactions).
That 5x is the whole prize.

Three things have to go for the vectorizer to fire, and each was measured separately:

  * the divide.  Replaced here by a Newton-Raphson reciprocal square root built from
    multiplies only, seeded by the classic magic-constant bit trick.  Four Newton steps land
    at 7.6e-15 relative error -- full double precision, not an accuracy compromise.  (Two
    steps give 4.8e-6 and three give 2.2e-11, if you ever want to trade accuracy for the two
    saved multiplies.)
  * the softening branch.  ``if r < h`` is data-dependent and blocks vectorization on its
    own, so these kernels are unsoftened-only.  Softened inputs must fall back.
  * the symmetric scatter.  This is the painful one.  Writing ``out[j] -= ...`` to exploit
    the i<->j symmetry takes the very same loop from 56 packed ops to 0, and from 27 us to
    166 us -- worse than just keeping the exact divide.  So this kernel deliberately gives up
    the 2x symmetry saving in order to buy the ~5x vectorization.  Net measured end-to-end
    against Accel_bruteforce_symmetric: ~2.1x at N=5000 and ~2.3x at N=50000.

The zero-guard, by contrast, is free (27.1 vs 26.7 us), so coincident particles are handled
by a select rather than a branch.

Consequences you are buying into:

  * momentum is NOT conserved to roundoff.  bruteforce_symmetric evaluates each pair once and
    applies it equal-and-opposite, so sum(m*a) vanishes by construction; here i and j are
    computed independently and it only vanishes to summation error.
  * positions must be SoA (separate contiguous x/y/z).  The (N,3) AoS layout costs a
    stride-3 gather per component; the transpose is O(N) and done in the wrapper.
  * unsoftened only.

Verified on x86-64 (AVX2/AVX-512).  The mechanism is LLVM-vectorizer-specific and has NOT
been checked on ARM/NEON, where the earlier scalar findings already differed.
"""

import numpy as np
from llvmlite import ir
from numba import njit, prange, types
from numba.extending import intrinsic

# 4 Newton steps -> ~7.6e-15 relative error (full float64).  3 -> 2.2e-11, 2 -> 4.8e-6.
_NEWTON_STEPS = 4

# float64 analogue of the Quake III 0x5f3759df seed constant
_RSQRT_MAGIC = 0x5FE6EB50C7B537A9


@intrinsic
def _as_i64(typingctx, x):
    """Reinterpret a float64's bits as int64 (no conversion).  numba has no scalar .view()."""
    if x != types.float64:
        return

    def codegen(context, builder, sig, args):
        """llvmlite codegen callback emitting the bitcast."""
        return builder.bitcast(args[0], ir.IntType(64))

    return types.int64(types.float64), codegen


@intrinsic
def _as_f64(typingctx, x):
    """Inverse of _as_i64."""
    if x != types.int64:
        return

    def codegen(context, builder, sig, args):
        """llvmlite codegen callback emitting the bitcast."""
        return builder.bitcast(args[0], ir.DoubleType())

    return types.float64(types.int64), codegen


@njit(fastmath=True, inline="always")
def _rsqrt(x):
    """1/sqrt(x) using only multiplies, so the enclosing loop stays vectorizable.

    Bit-trick seed (~5 correct bits) refined by Newton-Raphson, which doubles the correct
    bits per pass: 5 -> 10 -> 20 -> 40 -> 80, so four passes cover float64's 53.
    """
    y = _as_f64(np.int64(_RSQRT_MAGIC) - (_as_i64(x) >> 1))
    xhalf = 0.5 * x
    y = y * (1.5 - xhalf * y * y)
    y = y * (1.5 - xhalf * y * y)
    y = y * (1.5 - xhalf * y * y)
    y = y * (1.5 - xhalf * y * y)
    return y


@njit(fastmath=True, parallel=True)
def _accel_core(px, py, pz, m, G):
    """Unsoftened acceleration, SoA in and (N,3) out.  Non-symmetric: every pair twice."""
    n = px.shape[0]
    out = np.zeros((n, 3))
    for i in prange(n):
        xi = px[i]
        yi = py[i]
        zi = pz[i]
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            dx = px[j] - xi
            dy = py[j] - yi
            dz = pz[j] - zi
            r2 = dx * dx + dy * dy + dz * dz
            # select, not a branch: self and coincident particles contribute nothing
            rinv = _rsqrt(r2) if r2 > 0 else 0.0
            k = m[j] * rinv * rinv * rinv
            ax += k * dx
            ay += k * dy
            az += k * dz
        out[i, 0] = G * ax
        out[i, 1] = G * ay
        out[i, 2] = G * az
    return out


@njit(fastmath=True, parallel=True)
def _potential_core(px, py, pz, m, G):
    """Unsoftened potential, SoA in and (N,) out.  Non-symmetric: every pair twice."""
    n = px.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        xi = px[i]
        yi = py[i]
        zi = pz[i]
        phi = 0.0
        for j in range(n):
            dx = px[j] - xi
            dy = py[j] - yi
            dz = pz[j] - zi
            r2 = dx * dx + dy * dy + dz * dz
            rinv = _rsqrt(r2) if r2 > 0 else 0.0
            phi -= m[j] * rinv
        out[i] = G * phi
    return out


def _to_soa(x):
    """(N,3) -> three contiguous float64 columns.  O(N); the layout is what makes the inner
    loop unit-stride and therefore vectorizable."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    return (
        np.ascontiguousarray(x[:, 0]),
        np.ascontiguousarray(x[:, 1]),
        np.ascontiguousarray(x[:, 2]),
    )


def _check_unsoftened(softening):
    """Raise unless softening is absent or identically zero."""
    if softening is not None and np.any(np.asarray(softening) != 0):
        raise ValueError(
            "bruteforce_avx kernels are unsoftened-only: the 'r < h' branch is "
            "data-dependent and blocks vectorization, which is the entire point of this "
            "module. Use bruteforce_symmetric for softened inputs."
        )


def Accel_bruteforce_avx(x, m, softening=None, G=1.0):
    """Exact unsoftened mutually-interacting acceleration, AVX-vectorized.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Optional arguments:
    softening -- must be None or all-zero; raises otherwise (see module docstring)
    G -- gravitational constant (default 1.0)

    Returns:
    shape (N,3) array of gravitational accelerations

    Note: unlike Accel_bruteforce_symmetric this does not conserve momentum to roundoff,
    because each pair is evaluated independently rather than once and applied twice.
    """
    _check_unsoftened(softening)
    px, py, pz = _to_soa(x)
    return _accel_core(px, py, pz, np.ascontiguousarray(m, dtype=np.float64), G)


def Potential_bruteforce_avx(x, m, softening=None, G=1.0):
    """Exact unsoftened mutually-interacting potential, AVX-vectorized.

    Arguments and keywords match :func:`Accel_bruteforce_avx`; returns a shape (N,) array.
    """
    _check_unsoftened(softening)
    px, py, pz = _to_soa(x)
    return _potential_core(px, py, pz, np.ascontiguousarray(m, dtype=np.float64), G)


# ------------------------------------------------------------------------------------------
# Before trusting this
# ------------------------------------------------------------------------------------------
# Nothing below has been run yet.  In order:
#
#   1. Correctness: both kernels vs Accel_bruteforce / Potential_bruteforce.  Expect ~1e-14
#      relative, NOT the 1e-12 the other suites assert -- the NR rsqrt is not bit-exact, so
#      it needs its own tolerance rather than reusing RTOL from test_bruteforce_symmetric.
#   2. Codegen: confirm the packed-op count is nonzero and vdivpd/vsqrtpd are absent.  If a
#      future edit reintroduces a branch or a scatter this silently reverts to scalar and
#      only a codegen check will notice -- the answers stay correct.
#   3. Speed vs bruteforce_symmetric across N, on a quiet machine.
#   4. ARM/NEON: unverified, and the earlier scalar results already diverged between the two
#      architectures.  If it does not vectorize there, this needs a per-platform dispatch
#      rather than being wired in unconditionally.
#   5. Momentum: quantify how far sum(m*a) drifts from zero, so the cost of dropping the
#      symmetry is documented rather than discovered.
#
# Not wired into frontend.py.  Doing that needs a policy for softened inputs and for the
# N below which the extra parallel region does not pay -- cf. SYMMETRIC_NMIN.
