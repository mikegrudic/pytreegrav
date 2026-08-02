"""Is the runtime loop lower bound the blocker?  And does slicing fix it?"""

import re
import numpy as np
from llvmlite import ir as lir
from numba import njit, types
from numba.extending import intrinsic

PD = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd|max|min)\w*pd\b", re.M)
"""Count packed (vector) floating-point instructions in a function's assembly."""


def pk(f):
    """Count packed (vector) FP instructions in a compiled function's assembly."""
    return len(PD.findall("\n".join(f.inspect_asm().values())))


@intrinsic
def _i(tc, x):
    """Reinterpret a float64's bits as int64."""

    def cg(c, b, s, a):
        """llvmlite codegen callback emitting the bitcast."""
        return b.bitcast(a[0], lir.IntType(64))

    return types.int64(types.float64), cg


@intrinsic
def _f(tc, x):
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
        return b.bitcast(a[0], lir.DoubleType())

    return types.float64(types.int64), cg


@njit(fastmath=True, inline="always")
def rs(x):
    """1/sqrt(x) via magic-constant seed plus four Newton-Raphson steps."""
    y = _f(np.int64(0x5FE6EB50C7B537A9) - (_i(x) >> 1))
    h = 0.5 * x
    y = y * (1.5 - h * y * y)
    y = y * (1.5 - h * y * y)
    y = y * (1.5 - h * y * y)
    return y * (1.5 - h * y * y)


@njit(fastmath=True)  # start at 0 + scatter
def Z(px, py, pz, m, ox, mi, xi, yi, zi):
    """Loop starting at 0, with a scatter store: vectorizes."""
    ax = ay = az = 0.0
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        q = ri * ri * ri
        ax += q * m[j] * dx
        ay += q * m[j] * dy
        az += q * m[j] * dz
        ox[j] -= q * mi * dx
    return ax, ay, az


@njit(fastmath=True)  # runtime start + scatter
def R(px, py, pz, m, ox, i0, mi, xi, yi, zi):
    """Identical to Z but with a runtime lower bound: does NOT vectorize. The blocker."""
    ax = ay = az = 0.0
    for j in range(i0, px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        q = ri * ri * ri
        ax += q * m[j] * dx
        ay += q * m[j] * dy
        az += q * m[j] * dz
        ox[j] -= q * mi * dx
    return ax, ay, az


@njit(fastmath=True)  # runtime start, NO scatter
def RN(px, py, pz, m, i0, xi, yi, zi):
    """Runtime lower bound but no store: vectorizes, showing the bound alone is harmless."""
    ax = ay = az = 0.0
    for j in range(i0, px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        q = m[j] * ri * ri * ri
        ax += q * dx
        ay += q * dy
        az += q * dz
    return ax, ay, az


@njit(fastmath=True)  # THE FIX? slice the arrays so the loop starts at 0
def S(px, py, pz, m, ox, mi, xi, yi, zi):
    """Row kernel over slices, so the loop starts at 0 -- the fix."""
    ax = ay = az = 0.0
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        ri = rs(r2) if r2 > 0 else 0.0
        q = ri * ri * ri
        ax += q * m[j] * dx
        ay += q * m[j] * dy
        az += q * m[j] * dz
        ox[j] -= q * mi * dx
    return ax, ay, az


@njit(fastmath=True)  # driver that slices, calling S per row
def SYM_SLICED(px, py, pz, m, ox, oy, oz):
    """Full symmetric sweep calling S on slices, keeping symmetry and vectorization."""
    n = px.shape[0]
    for i in range(n - 1):
        ax, ay, az = S(px[i + 1 :], py[i + 1 :], pz[i + 1 :], m[i + 1 :], ox[i + 1 :], m[i], px[i], py[i], pz[i])
        ox[i] += ax
        oy[i] += ay
        oz[i] += az


n = 5000
rng = np.random.default_rng(0)
px, py, pz = (np.ascontiguousarray(v) for v in rng.random((3, n)))
m = rng.random(n)
o1, o2, o3 = np.zeros(n), np.zeros(n), np.zeros(n)
Z(px, py, pz, m, o1, m[0], 0.5, 0.5, 0.5)
R(px, py, pz, m, o1, 3, m[0], 0.5, 0.5, 0.5)
RN(px, py, pz, m, 3, 0.5, 0.5, 0.5)
SYM_SLICED(px, py, pz, m, o1, o2, o3)
for nm, f in (
    ("start 0    + scatter", Z),
    ("runtime i0 + scatter", R),
    ("runtime i0, no scatter", RN),
    ("SLICED (start 0) + scatter", S),
):
    p = pk(f)
    print(f"  {nm:28s} packed {p:4d}  {'VECTORIZED' if p > 10 else 'NOT vectorized'}")
