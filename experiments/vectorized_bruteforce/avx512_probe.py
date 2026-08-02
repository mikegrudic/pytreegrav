"""Reaching AVX-512 (512-bit zmm) from numba, and what it costs.

numba emits only 256-bit ymm on Cascade Lake because LLVM defaults to
prefer-vector-width=256 there (AVX-512 downclocks the core). Two ways to override were tried:

  --prefer-vector-width=512   no effect (still 0 zmm). It is a function attribute the frontend
                              stamps on, not a cl::opt the vectorizer reads, so numba never
                              sets it.
  --force-vector-width=8      works: 195 zmm registers, and 1.64x on the non-symmetric kernel.

Run one option per process (the flag is global and cannot be un-set):

    python avx512_probe.py                            # baseline
    python avx512_probe.py --force-vector-width=8

WARNING: --force-vector-width=8 is a process-global LLVM debug option, not a stable API. It
forces VF=8 on every loop numba compiles for the life of the process, including in user code
and any other library sharing the numba runtime. It also cannot be scoped to one function, and
carries no compatibility guarantee across LLVM versions. Fine for a user script that owns the
whole process; not something to put in library code.
"""

import re, sys, threading, time
import numpy as np

opt = sys.argv[1] if len(sys.argv) > 1 else ""
if opt:
    import llvmlite.binding as llvm

    llvm.set_option("numba", opt)

sys.path.insert(0, ".")
from numba import get_num_threads
from bruteforce_avx import Accel_bruteforce_avx
from bruteforce_sym_vec import Accel_bruteforce_sym_vec
from pytreegrav import Accel, ConstructTree
from pytreegrav.bruteforce import Accel_bruteforce
from pytreegrav.bruteforce_symmetric import Accel_bruteforce_symmetric


def ghz():
    """Mean core clock, GHz."""
    with open("/proc/cpuinfo") as f:
        v = [float(l.split(":")[1]) for l in f if l.startswith("cpu MHz")]
    return sum(v) / len(v) / 1000


def bench(f, *a, reps=3):
    """Best-of-reps wall time plus median clock during the run."""
    f(*a)
    sm = []
    stop = threading.Event()

    def s_():
        """Clock sampler."""
        while not stop.is_set():
            sm.append(ghz())
            time.sleep(0.05)

    th = threading.Thread(target=s_)
    th.start()
    t = np.inf
    for _ in range(reps):
        s = time.perf_counter()
        f(*a)
        t = min(t, time.perf_counter() - s)
    stop.set()
    th.join()
    return t, (np.median(sm) if sm else float("nan"))


def plummer(n, seed=42):
    """Plummer sampler."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    r = np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))
    d = rng.normal(size=(n, 3))
    return np.ascontiguousarray((d.T * r / np.sum(d**2, axis=1) ** 0.5).T), np.repeat(1.0 / n, n)


tag = opt or "(baseline)"
print(f"### {tag}   threads={get_num_threads()}")

# correctness
n = 3000
rng = np.random.default_rng(1)
xc = np.ascontiguousarray(rng.random((n, 3)))
mc = rng.random(n) / n
ref = Accel_bruteforce(xc, mc, np.zeros(n))
for nm, f, a in (
    ("avx", Accel_bruteforce_avx, (xc, mc)),
    ("sym_vec", Accel_bruteforce_sym_vec, (xc, mc, np.zeros(n))),
    ("sym_vec soft", Accel_bruteforce_sym_vec, (xc, mc, np.repeat(0.05, n))),
):
    r = ref if "soft" not in nm else Accel_bruteforce(xc, mc, np.repeat(0.05, n))
    g = f(*a)
    print(f"  correctness {nm:14s} {np.max(np.abs(g - r)) / np.max(np.abs(r)):.2e}")

N = 50000
rng = np.random.default_rng(0)
x = np.ascontiguousarray(rng.random((N, 3)))
m = rng.random(N) / N
h = np.zeros(N)
for nm, f, a, pairs in (
    ("avx", Accel_bruteforce_avx, (x, m), float(N) * N),
    ("sym_vec", Accel_bruteforce_sym_vec, (x, m, h), 0.5 * N * (N - 1)),
    ("shipped symmetric", Accel_bruteforce_symmetric, (x, m, h), 0.5 * N * (N - 1)),
):
    t, c = bench(f, *a)
    print(f"  bruteforce {nm:18s} {t * 1e3:8.1f}ms  {pairs * 34 / t / 1e9:6.1f} GFLOP/s  {c:.2f} GHz")

# collateral: does forcing VF=8 hurt the tree walk?
pos, mm = plummer(200000)
hh = np.zeros(200000)
t, c = (
    bench(Accel, pos, mm, hh, reps=2, **{})
    if False
    else bench(lambda: Accel(pos, mm, hh, method="tree", parallel=True), reps=2)
)
print(f"  TREE walk N=200000  {t * 1e3:8.1f}ms  {c:.2f} GHz")
tb, _ = bench(lambda: ConstructTree(pos, mm, hh), reps=2)
print(f"  TREE build N=200000 {tb * 1e3:8.1f}ms")
