"""Run the checklist from bruteforce_avx.py's 'Before trusting this'."""

import re, time
import numpy as np
from numba import get_num_threads
from pytreegrav.bruteforce import Accel_bruteforce, Potential_bruteforce
from pytreegrav.bruteforce_symmetric import Accel_bruteforce_symmetric, Potential_bruteforce_symmetric
from bruteforce_avx import Accel_bruteforce_avx, Potential_bruteforce_avx, _accel_core, _potential_core

print(f"threads = {get_num_threads()}\n")

# ---- 1. correctness vs the exact serial reference ----------------------------------------
print("1. correctness vs exact serial reference")
for N in (100, 1000, 5000):
    rng = np.random.default_rng(1)
    x = rng.random((N, 3))
    m = rng.random(N) / N
    h = np.zeros(N)
    a_ref = Accel_bruteforce(x, m, h)
    p_ref = Potential_bruteforce(x, m, h)
    a = Accel_bruteforce_avx(x, m)
    p = Potential_bruteforce_avx(x, m)
    ea = np.max(np.abs(a - a_ref)) / np.max(np.abs(a_ref))
    ep = np.max(np.abs(p - p_ref)) / np.max(np.abs(p_ref))
    print(f"   N={N:5d}  accel rel err {ea:.2e}   potential rel err {ep:.2e}")

# ---- 2. codegen -------------------------------------------------------------------------
print("\n2. codegen (want packed > 0, vdivpd/vsqrtpd == 0)")
PD = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd|fmsub|fnmadd|max|min)\w*pd\b", re.M)
SD = re.compile(r"^\s+v?\w*(?:add|sub|mul|div|sqrt|fmadd)\w*sd\b", re.M)
for nm, f in (("accel core", _accel_core), ("potential core", _potential_core)):
    asm = "\n".join(f.inspect_asm().values())
    npk = len(PD.findall(asm))
    nsd = len(SD.findall(asm))
    ndiv = len(re.findall("vdivpd", asm))
    nsq = len(re.findall("vsqrtpd", asm))
    print(f"   {nm:15s} packed {npk:4d} | vdivpd {ndiv:3d} | vsqrtpd {nsq:3d} | scalar-sd {nsd:4d}")


# ---- 3. speed vs the symmetric kernel ----------------------------------------------------
def bench(f, *a, reps=5):
    """Best-of-repeats wall time; assumes the callable is already compiled."""
    f(*a)
    t = np.inf
    for _ in range(reps):
        s = time.perf_counter()
        f(*a)
        t = min(t, time.perf_counter() - s)
    return t


print("\n3. speed vs bruteforce_symmetric")
print(f"   {'N':>7s} | {'symmetric':>11s} {'avx':>10s} {'speedup':>8s}")
for N in (2000, 5000, 20000, 50000, 100000):
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.random((N, 3)))
    m = rng.random(N) / N
    h = np.zeros(N)
    ts = bench(Accel_bruteforce_symmetric, x, m, h, reps=3)
    ta = bench(Accel_bruteforce_avx, x, m, reps=3)
    print(f"   {N:7d} | {ts * 1e3:10.2f}ms {ta * 1e3:9.2f}ms {ts / ta:7.2f}x")

# ---- 4. momentum drift -------------------------------------------------------------------
print("\n4. momentum conservation (sum(m*a) / sum|m*a|)")
for N in (5000, 50000):
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.random((N, 3)))
    m = rng.random(N) / N
    h = np.zeros(N)
    for nm, a in (("symmetric", Accel_bruteforce_symmetric(x, m, h)), ("avx", Accel_bruteforce_avx(x, m))):
        net = np.abs((m[:, None] * a).sum(0)).max() / np.abs(m[:, None] * a).sum()
        print(f"   N={N:6d} {nm:10s} {net:.2e}")

# ---- 5. the softening guard --------------------------------------------------------------
print("\n5. softened input is rejected, not silently wrong")
try:
    Accel_bruteforce_avx(np.random.rand(10, 3), np.ones(10) / 10, np.full(10, 0.01))
    print("   FAIL - no error raised")
except ValueError as e:
    print(f"   ok - ValueError: {str(e)[:70]}...")
