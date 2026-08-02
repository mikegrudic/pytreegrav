"""How much headroom is there? (a) if divides were free, (b) in float32."""

import re, time
import numpy as np
from numba import njit
from numpy import sqrt

PD = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd|max|min)\w*p[sd]\b", re.M)


def packed(f):
    """Return (packed ops, vdivp*, vsqrtp*, vrsqrtps) counts from a function's assembly."""
    a = "\n".join(f.inspect_asm().values())
    return (
        len(PD.findall(a)),
        len(re.findall(r"vdivp[sd]", a)),
        len(re.findall(r"vsqrtp[sd]", a)),
        len(re.findall(r"vrsqrtp[s]", a)),
    )


@njit(fastmath=True)  # real kernel, f64
def f64_div(px, py, pz, m, xi, yi, zi):
    """float64 row kernel using the real divide -- what the shipped kernel does."""
    ax = ay = az = 0.0
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        k = m[j] / (r2 * sqrt(r2))
        ax += k * dx
        ay += k * dy
        az += k * dz
    return ax, ay, az


@njit(fastmath=True)  # divide+sqrt removed -> upper bound on any SIMD win
def f64_nodiv(px, py, pz, m, rinv, xi, yi, zi):
    """float64 row kernel with the divide replaced by a precomputed reciprocal, giving an upper bound on any SIMD win."""
    ax = ay = az = 0.0
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        k = m[j] * rinv[j] * rinv[j] * rinv[j]
        ax += k * dx
        ay += k * dy
        az += k * dz
    return ax, ay, az


@njit(fastmath=True)  # float32: 8-wide AVX2 lanes, hardware rsqrt exists
def f32_div(px, py, pz, m, xi, yi, zi):
    """float32 row kernel with a divide: tests whether narrower types help (they do not)."""
    ax = np.float32(0)
    ay = np.float32(0)
    az = np.float32(0)
    for j in range(px.shape[0]):
        dx = px[j] - xi
        dy = py[j] - yi
        dz = pz[j] - zi
        r2 = dx * dx + dy * dy + dz * dz
        k = m[j] / (r2 * np.sqrt(r2))
        ax += k * dx
        ay += k * dy
        az += k * dz
    return ax, ay, az


N = 20000
rng = np.random.default_rng(0)
px, py, pz = (np.ascontiguousarray(v) for v in rng.random((3, N)))
m = rng.random(N) / N
r2 = (px - 0.5) ** 2 + (py - 0.5) ** 2 + (pz - 0.5) ** 2
rinv = 1.0 / np.sqrt(r2)
p32, q32, r32 = (v.astype(np.float32) for v in (px, py, pz))
m32 = m.astype(np.float32)

f64_div(px, py, pz, m, 0.5, 0.5, 0.5)
f64_nodiv(px, py, pz, m, rinv, 0.5, 0.5, 0.5)
f32_div(p32, q32, r32, m32, np.float32(0.5), np.float32(0.5), np.float32(0.5))


def bench(f, *a):
    """Best-of-repeats wall time; assumes the callable is already compiled."""
    f(*a)
    t = np.inf
    for _ in range(300):
        s = time.perf_counter()
        f(*a)
        t = min(t, time.perf_counter() - s)
    return t * 1e6


rows = [
    ("f64 with divide", f64_div, (px, py, pz, m, 0.5, 0.5, 0.5)),
    ("f64 divide removed", f64_nodiv, (px, py, pz, m, rinv, 0.5, 0.5, 0.5)),
    ("f32 with divide", f32_div, (p32, q32, r32, m32, np.float32(0.5), np.float32(0.5), np.float32(0.5))),
]
print(f"{'variant':22s} {'us':>8s} {'vs f64':>8s} | {'packed':>7s} {'vdivp':>6s} {'vsqrtp':>7s} {'vrsqrtps':>9s}")
print("-" * 76)
base = None
for nm, f, a in rows:
    t = bench(f, *a)
    base = base or t
    p, d, s, rs = packed(f)
    print(f"{nm:22s} {t:8.1f} {base / t:7.2f}x | {p:7d} {d:6d} {s:7d} {rs:9d}")
