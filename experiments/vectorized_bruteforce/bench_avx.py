"""Benchmark bruteforce_avx vs bruteforce_symmetric.  Arch-aware codegen check."""

import platform, re, time
import numpy as np
from numba import get_num_threads
from pytreegrav.bruteforce import Accel_bruteforce
from pytreegrav.bruteforce_symmetric import Accel_bruteforce_symmetric
from bruteforce_avx import Accel_bruteforce_avx, _accel_core

ARM = platform.machine() in ("arm64", "aarch64")
if ARM:
    PACKED = re.compile(r"^\s+f\w+\.2d\b", re.M)
    SCALAR = re.compile(r"^\s+f(?:add|sub|mul|div|sqrt|madd|msub|nmsub)\s+d\d", re.M)
    WIDTH = "128-bit NEON (.2d = 2 doubles)"
else:
    PACKED = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd|fmsub|fnmadd|max|min)\w*pd\b", re.M)
    SCALAR = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd)\w*sd\b", re.M)
    WIDTH = "AVX (ymm = 4 doubles)"

FLOP_AVX, FLOP_SYM = 34, 19


def bench(f, *a, reps=3):
    """Best-of-repeats wall time; assumes the callable is already compiled."""
    f(*a)
    t = np.inf
    for _ in range(reps):
        s = time.perf_counter()
        f(*a)
        t = min(t, time.perf_counter() - s)
    return t


print(f"platform: {platform.machine()}   threads: {get_num_threads()}   vector width: {WIDTH}")

# correctness once, at a size the serial reference can handle
rng = np.random.default_rng(1)
n = 3000
xc = np.ascontiguousarray(rng.random((n, 3)))
mc = rng.random(n) / n
ref = Accel_bruteforce(xc, mc, np.zeros(n))
got = Accel_bruteforce_avx(xc, mc)
print(f"correctness vs exact serial (N={n}): max rel err {np.max(np.abs(got - ref)) / np.max(np.abs(ref)):.2e}")

asm = "\n".join(_accel_core.inspect_asm().values())
npk, nsc = len(PACKED.findall(asm)), len(SCALAR.findall(asm))
print(f"codegen: packed {npk}, scalar {nsc}  -> {'VECTORIZED' if npk > 30 else 'NOT vectorized'}\n")

print(
    f"{'N':>7s} | {'symmetric':>11s} {'avx':>10s} {'speedup':>8s} | "
    f"{'sym Gint/s':>11s} {'avx Gint/s':>11s} | {'avx GFLOP/s':>12s}"
)
print("-" * 84)
for N in (2000, 5000, 20000, 50000, 100000):
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.random((N, 3)))
    m = rng.random(N) / N
    h = np.zeros(N)
    ts = bench(Accel_bruteforce_symmetric, x, m, h)
    ta = bench(Accel_bruteforce_avx, x, m)
    isym, iavx = 0.5 * N * (N - 1), float(N) * N  # symmetric does each pair once
    print(
        f"{N:7d} | {ts * 1e3:10.2f}ms {ta * 1e3:9.2f}ms {ts / ta:7.2f}x | "
        f"{isym / ts / 1e9:11.2f} {iavx / ta / 1e9:11.2f} | {iavx * FLOP_AVX / ta / 1e9:12.1f}"
    )
