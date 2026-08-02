"""Is it loop-body size?  Vary the Newton-step count with and without a scatter store."""

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
    """llvmlite codegen callback emitting the bitcast."""
    """llvmlite codegen callback emitting the bitcast."""

    def cg(c, b, s, a):
        """llvmlite codegen callback emitting the bitcast."""
        return b.bitcast(a[0], lir.DoubleType())

    return types.float64(types.int64), cg


def build(nsteps, scatter, use_bitcast=True):
    """Build a row-kernel variant with the given Newton-step count and scatter setting."""
    if use_bitcast:

        @njit(fastmath=True, inline="always")
        def rs(x):
            """1/sqrt(x) with a configurable number of Newton steps, so body size can be varied."""
            y = _f(np.int64(0x5FE6EB50C7B537A9) - (_i(x) >> 1))
            h = 0.5 * x
            for _ in range(nsteps):
                y = y * (1.5 - h * y * y)
            return y
    else:  # seed from a plain divide instead of the bit trick

        @njit(fastmath=True, inline="always")
        def rs(x):
            """1/sqrt(x) with a configurable number of Newton steps, so body size can be varied."""
            y = 1.0 / np.sqrt(x)
            h = 0.5 * x
            for _ in range(nsteps):
                y = y * (1.5 - h * y * y)
            return y

    if scatter:

        @njit(fastmath=True)
        def k(px, py, pz, m, ox, mi, xi, yi, zi):
            """Row kernel built with the requested Newton-step count, with or without a scatter store."""
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
    else:

        @njit(fastmath=True)
        def k(px, py, pz, m, ox, mi, xi, yi, zi):
            """Row kernel built with the requested Newton-step count, with or without a scatter store."""
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
            return ax, ay, az

    return k


n = 5000
rng = np.random.default_rng(0)
px, py, pz = (np.ascontiguousarray(v) for v in rng.random((3, n)))
m = rng.random(n)
o = np.zeros(n)
print(f"{'Newton steps':>13s} | {'no scatter':>11s} {'with scatter':>13s}")
print("-" * 44)
for ns in (0, 1, 2, 3, 4):
    a = build(ns, False)
    b = build(ns, True)
    a(px, py, pz, m, o, m[0], 0.5, 0.5, 0.5)
    b(px, py, pz, m, o, m[0], 0.5, 0.5, 0.5)
    print(f"{ns:13d} | {pk(a):11d} {pk(b):13d}")
print("\nsame, but seeding rsqrt from a divide instead of the bitcast intrinsic:")
print(f"{'Newton steps':>13s} | {'no scatter':>11s} {'with scatter':>13s}")
for ns in (0, 2, 4):
    a = build(ns, False, False)
    b = build(ns, True, False)
    a(px, py, pz, m, o, m[0], 0.5, 0.5, 0.5)
    b(px, py, pz, m, o, m[0], 0.5, 0.5, 0.5)
    print(f"{ns:13d} | {pk(a):11d} {pk(b):13d}")
