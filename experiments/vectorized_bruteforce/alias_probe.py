"""Is the symmetry-vs-vectorization conflict fundamental, or just alias analysis?"""

import re, time
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

PD = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd|fmsub|fnmadd|max|min)\w*pd\b", re.M)
"""Count packed (vector) floating-point instructions in a function's assembly."""


def npk(f):
    """Count packed (vector) FP instructions in a compiled function's assembly."""
    return len(PD.findall("\n".join(f.inspect_asm().values())))


@intrinsic
def _as_i64(tc, x):
    """Reinterpret a float64's bits as int64 (numba has no scalar .view())."""

    def cg(c, b, s, a):
        """llvmlite codegen callback emitting the bitcast."""
        return b.bitcast(a[0], ir.IntType(64))

    return types.int64(types.float64), cg


@intrinsic
def _as_f64(tc, x):
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""

    def cg(c, b, s, a):
        """llvmlite codegen callback emitting the bitcast."""
        return b.bitcast(a[0], ir.DoubleType())

    return types.float64(types.int64), cg


@njit(fastmath=True, inline="always")
def rs(x):
    """1/sqrt(x) via magic-constant seed plus four Newton-Raphson steps; multiplies only."""
    y = _as_f64(np.int64(0x5FE6EB50C7B537A9) - (_as_i64(x) >> 1))
    xh = 0.5 * x
    y = y * (1.5 - xh * y * y)
    y = y * (1.5 - xh * y * y)
    y = y * (1.5 - xh * y * y)
    return y * (1.5 - xh * y * y)


# A: gather only (4 array args)
@njit(fastmath=True)
def A(px, py, pz, m, i0, xi, yi, zi):
    """Gather-only row kernel (4 array args): the control that vectorizes."""
    ax = ay = az = 0.0
    for j in range(i0, px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        k = m[j] * ri * ri * ri
        ax += k * dx
        ay += k * dy
        az += k * dz
    return ax, ay, az


# B: + scatter into ONE extra array (5 array args)
@njit(fastmath=True)
def B(px, py, pz, m, ox, i0, mi, xi, yi, zi):
    """As A plus one scatter output array: enough to kill vectorization on its own."""
    ax = ay = az = 0.0
    for j in range(i0, px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        k = ri * ri * ri
        ax += k * m[j] * dx
        ay += k * m[j] * dy
        az += k * m[j] * dz
        ox[j] -= k * mi * dx
    return ax, ay, az


# C: + scatter into THREE extra arrays (7 array args)  <- the symmetric kernel's shape
@njit(fastmath=True)
def C(px, py, pz, m, ox, oy, oz, i0, mi, xi, yi, zi):
    """As A plus three scatter arrays -- the shape the symmetric kernel actually uses."""
    ax = ay = az = 0.0
    for j in range(i0, px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        k = ri * ri * ri
        ax += k * m[j] * dx
        ay += k * m[j] * dy
        az += k * m[j] * dz
        ox[j] -= k * mi * dx
        oy[j] -= k * mi * dy
        oz[j] -= k * mi * dz
    return ax, ay, az


# D: scatter into one 2D (N,3) array instead of three 1D ones (5 array args, stride-3 writes)
@njit(fastmath=True)
def D(px, py, pz, m, out, i0, mi, xi, yi, zi):
    """As A plus a single (N,3) scatter array, testing 2D vs three 1D outputs."""
    ax = ay = az = 0.0
    for j in range(i0, px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        k = ri * ri * ri
        ax += k * m[j] * dx
        ay += k * m[j] * dy
        az += k * m[j] * dz
        out[j, 0] -= k * mi * dx
        out[j, 1] -= k * mi * dy
        out[j, 2] -= k * mi * dz
    return ax, ay, az


N = 20000
rng = np.random.default_rng(0)
px, py, pz = (np.ascontiguousarray(v) for v in rng.random((3, N)))
m = rng.random(N) / N
z = lambda: np.zeros(N)
z2 = lambda: np.zeros((N, 3))
cases = [
    ("A gather only        (4 arrays)", A, (px, py, pz, m, 0, 0.5, 0.5, 0.5)),
    ("B +1 scatter array   (5 arrays)", B, (px, py, pz, m, z(), 0, m[0], 0.5, 0.5, 0.5)),
    ("C +3 scatter arrays  (7 arrays)", C, (px, py, pz, m, z(), z(), z(), 0, m[0], 0.5, 0.5, 0.5)),
    ("D +1 (N,3) scatter   (5 arrays)", D, (px, py, pz, m, z2(), 0, m[0], 0.5, 0.5, 0.5)),
]


def bench(f, *a):
    """Best-of-repeats wall time; assumes the callable is already compiled."""
    f(*a)
    t = np.inf
    for _ in range(200):
        s = time.perf_counter()
        f(*a)
        t = min(t, time.perf_counter() - s)
    return t * 1e6


print(f"{'variant':34s} {'us':>8s} {'packed':>8s}  vectorized?")
print("-" * 66)
for nm, f, a in cases:
    t = bench(f, *a)
    p = npk(f)
    print(f"{nm:34s} {t:8.1f} {p:8d}  {'YES' if p > 30 else 'no'}")
