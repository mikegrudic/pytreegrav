"""SUPERSEDED, kept for the negative result: FP32 brute-force potential as a chain of MLX array ops.

Superseded by bruteforce_metal.py, which does identical math in a hand-written Metal kernel and is
~29x faster.  This version runs at ~4 Gpair/s -- *slower* than the parallel CPU kernel (0.44-0.60x)
-- because mx.compile does not fuse the trailing sum(axis=1) into the tile body, so every tile
streams ~15 (TT, TS) float32 temporaries through unified memory.  The result is bandwidth-bound at
O(1) flop/byte, not an N-body kernel.  It is numerically correct (6e-8 vs the CPU), which is what
makes it dangerous: the number looks like a measurement of the idea and is a measurement of the
implementation.  See README.md.


Computes the same quantity as pytreegrav.bruteforce.Potential_bruteforce_parallel:

    phi_i = G * sum_{j != i}  m_j * PotentialKernel(r_ij, h_ij)   if r_ij <  h_ij
            G * sum_{j != i} -m_j / r_ij                          if r_ij >= h_ij, r_ij > 0

with h_ij = max(h_i, h_j) and the cubic-spline PotentialKernel from pytreegrav/kernel.py.

The GPU path is tiled over (target, source) blocks and mx.compile'd so the tile body fuses into
one kernel instead of materialising (TT, TS) temporaries per term.  All GPU math is float32; the
accumulator is float32 too, since Metal has no float64 at all.

Caveat worth remembering when reading the numbers: Apple silicon has unified memory, so there is
no host->device copy here.  A discrete GPU would pay PCIe on top of whatever this measures.
"""

import argparse
import time

import numpy as np
import mlx.core as mx

# --------------------------------------------------------------------------------------------------
# GPU kernel
# --------------------------------------------------------------------------------------------------

# spline coefficients, matching pytreegrav/kernel.py PotentialKernel
_C = dict(
    a0=-2.8,
    a1=5.33333333333333333,
    a2=-9.6,
    a3=6.4,
    b0=-3.2,
    b1=0.0666666666666666666,
    b2=10.6666666666666666,
    b3=-16.0,
    b4=9.6,
    b5=-2.13333333333333333,
)


@mx.compile
def _tile(xi, hi, xj, hj, mj, same):
    """Potential contribution of source tile (xj, hj, mj) to target tile (xi, hi).

    ``same`` is a (TT, TS) bool mask that is True where target and source are the SAME particle
    (self-interaction), which the CPU kernel skips via ``if i == j: continue``.  Returns (TT,).
    """
    dx = xi[:, 0:1] - xj[None, :, 0]
    dy = xi[:, 1:2] - xj[None, :, 1]
    dz = xi[:, 2:3] - xj[None, :, 2]
    r2 = dx * dx + dy * dy + dz * dz
    r = mx.sqrt(r2)

    h = mx.maximum(hi[:, None], hj[None, :])

    # far field: -m_j / r, guarded against r == 0 (masked out below)
    r_safe = mx.where(r > 0, r, mx.ones_like(r))
    phi_far = -mj[None, :] / r_safe

    # near field: m_j * PotentialKernel(r, h).  h > 0 wherever r < h, so hinv is finite there.
    h_safe = mx.where(h > 0, h, mx.ones_like(h))
    hinv = 1.0 / h_safe
    q = r * hinv
    q_safe = mx.where(q > 0, q, mx.ones_like(q))
    q2 = q * q
    k_inner = (_C["a0"] + q2 * (_C["a1"] + q2 * (_C["a3"] * q + _C["a2"]))) * hinv
    k_outer = (_C["b0"] + _C["b1"] / q_safe + q2 * (_C["b2"] + q * (_C["b3"] + q * (_C["b4"] + _C["b5"] * q)))) * hinv
    phi_near = mj[None, :] * mx.where(q <= 0.5, k_inner, k_outer)

    contrib = mx.where(r < h, phi_near, phi_far)
    # drop self-interaction and exact-coincidence-with-zero-softening (CPU falls through both)
    keep = mx.logical_and(mx.logical_not(same), mx.logical_or(r > 0, h > 0))
    return mx.where(keep, contrib, mx.zeros_like(contrib)).sum(axis=1)


def potential_gpu(x, m, softening, G=1.0, tt=1024, ts=8192):
    """FP32 brute-force potential on the GPU.  x (N,3), m (N,), softening (N,) numpy arrays."""
    n = x.shape[0]
    xg = mx.array(np.ascontiguousarray(x, dtype=np.float32))
    mg = mx.array(np.ascontiguousarray(m, dtype=np.float32))
    hg = mx.array(np.ascontiguousarray(softening, dtype=np.float32))
    idx = mx.arange(n)

    out = []
    for a in range(0, n, tt):
        b = min(a + tt, n)
        acc = mx.zeros((b - a,), dtype=mx.float32)
        for c in range(0, n, ts):
            d = min(c + ts, n)
            same = idx[a:b, None] == idx[None, c:d]
            acc = acc + _tile(xg[a:b], hg[a:b], xg[c:d], hg[c:d], mg[c:d], same)
        out.append(acc)
    res = mx.concatenate(out) * G
    mx.eval(res)
    return res


# --------------------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------------------


def plummer(n, seed=42):
    """Plummer sphere positions, unit total mass, matching the style of tests/test_plummer.py."""
    rng = np.random.default_rng(seed)
    r = 1.0 / np.sqrt(rng.random(n) ** (-2.0 / 3.0) - 1.0)
    u = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    s = np.sqrt(1.0 - u * u)
    x = np.column_stack([r * s * np.cos(phi), r * s * np.sin(phi), r * u])
    return np.ascontiguousarray(x), np.repeat(1.0 / n, n)


def bench(fn, *a, reps=3, **kw):
    """Return (best wall time, last result).  One untimed warm-up call absorbs JIT/kernel compile."""
    r = fn(*a, **kw)
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        r = fn(*a, **kw)
        best = min(best, time.perf_counter() - t)
    return best, r


def rel_err(a, b):
    """RMS relative error of a against reference b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--sizes", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 131072])
    p.add_argument("--softening", type=float, default=0.01, help="uniform softening; 0 disables")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tt", type=int, default=1024)
    p.add_argument("--ts", type=int, default=8192)
    args = p.parse_args()

    from numba import get_num_threads
    from pytreegrav.bruteforce import Potential_bruteforce_parallel
    from pytreegrav.bruteforce_symmetric import Potential_bruteforce_symmetric

    print(f"device={mx.default_device()}  numba_threads={get_num_threads()}  softening={args.softening}")
    print(f"tile=({args.tt},{args.ts})  reps={args.reps}  (best-of; one untimed warm-up)")
    print()
    hdr = f"{'N':>8} {'CPU par':>10} {'CPU sym':>10} {'GPU fp32':>10} {'vs par':>7} {'vs sym':>7} {'Gpair/s':>9} {'rel err':>9}"
    print(hdr)
    print("-" * len(hdr))

    for n in args.sizes:
        x, m = plummer(n)
        h = np.repeat(args.softening, n)

        t_par, phi_par = bench(Potential_bruteforce_parallel, x, m, h, reps=args.reps)
        t_sym, _ = bench(Potential_bruteforce_symmetric, x, m, h, reps=args.reps)
        t_gpu, phi_gpu = bench(potential_gpu, x, m, h, reps=args.reps, tt=args.tt, ts=args.ts)
        phi_gpu = np.array(phi_gpu, copy=False)

        gpair = n * n / t_gpu / 1e9
        print(
            f"{n:>8} {t_par:>9.4f}s {t_sym:>9.4f}s {t_gpu:>9.4f}s "
            f"{t_par / t_gpu:>6.2f}x {t_sym / t_gpu:>6.2f}x {gpair:>9.2f} {rel_err(phi_gpu, phi_par):>9.2e}"
        )


if __name__ == "__main__":
    main()
