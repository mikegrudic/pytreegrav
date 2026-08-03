"""FP32 brute-force potential as a real Metal N-body kernel, vs pytreegrav's parallel CPU kernels.

Computes the same quantity as pytreegrav.bruteforce.Potential_bruteforce_parallel:

    phi_i = G * sum_{j != i}  m_j * PotentialKernel(r_ij, h_ij)   if r_ij <  h_ij
            G * sum_{j != i} -m_j / r_ij                          if r_ij >= h_ij, r_ij > 0

with h_ij = max(h_i, h_j) and the cubic-spline PotentialKernel from pytreegrav/kernel.py.

This is the classic GPU N-body structure (Nyland/Harris): one thread per target, the target's
position and its potential accumulator live in registers for the whole kernel, and sources are
staged through threadgroup memory in tiles so each source load is reused by TG targets.  That
turns the arithmetic intensity from O(1) flop/byte -- which is what a chain of MLX array ops
gives you (see bruteforce_mlx_arrayops.py: same math, ~4 Gpair/s, *slower* than the CPU) --
into O(TG) flop/byte.

The far field uses Metal's hardware ``rsqrt``, which is the entire reason FP32 is fast on a GPU
and the operation that has no FP64 equivalent in silicon.  ``--precise`` switches to
``1/sqrt(r2)`` to price that choice.

Caveat when reading the numbers: Apple silicon has unified memory, so there is no host->device
copy here.  A discrete GPU would pay PCIe on top of this.
"""

import argparse
import time

import mlx.core as mx
import numpy as np

_SRC = """
    uint tid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint N = x_shape[0];

    threadgroup float sx[{TG}];
    threadgroup float sy[{TG}];
    threadgroup float sz[{TG}];
    threadgroup float sm[{TG}];
    threadgroup float sh[{TG}];

    // clamp rather than early-return: every thread must reach the barriers below
    uint i = tid < N ? tid : 0;
    float xi = x[3 * i + 0];
    float yi = x[3 * i + 1];
    float zi = x[3 * i + 2];
    float hi = hsml[i];
    float phi = 0.0f;

    for (uint base = 0; base < N; base += {TG}) {{
        uint j = base + lid;
        bool valid = j < N;
        uint jc = valid ? j : 0;
        sx[lid] = x[3 * jc + 0];
        sy[lid] = x[3 * jc + 1];
        sz[lid] = x[3 * jc + 2];
        sm[lid] = valid ? mass[jc] : 0.0f;   // zero mass makes padding lanes inert
        sh[lid] = hsml[jc];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kmax = metal::min((uint){TG}, N - base);
        for (uint k = 0; k < kmax; ++k) {{
            if (base + k == i) continue;             // CPU: `if i == j: continue`
            float dx = xi - sx[k];
            float dy = yi - sy[k];
            float dz = zi - sz[k];
            float r2 = dx * dx + dy * dy + dz * dz;
            float hh = metal::max(hi, sh[k]);
            float mj = sm[k];
            if (r2 < hh * hh) {{                     // CPU: `if r < h`
                float r = metal::sqrt(r2);
                float hinv = 1.0f / hh;
                float q = r * hinv;
                float q2 = q * q;
                float kk;
                if (q <= 0.5f) {{
                    kk = (-2.8f + q2 * (5.33333333333333f + q2 * (6.4f * q - 9.6f))) * hinv;
                }} else {{
                    kk = (-3.2f + 0.0666666666666667f / q
                          + q2 * (10.6666666666667f + q * (-16.0f + q * (9.6f - 2.13333333333333f * q)))) * hinv;
                }}
                phi += mj * kk;
            }} else if (r2 > 0.0f) {{                 // CPU: `elif r > 0`
                phi += -mj * {RSQRT};
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid < N) out[tid] = phi * G[0];
"""

_CACHE = {}


def _kernel(tg, precise):
    key = (tg, precise)
    if key not in _CACHE:
        src = _SRC.format(TG=tg, RSQRT="(1.0f / metal::sqrt(r2))" if precise else "metal::rsqrt(r2)")
        _CACHE[key] = mx.fast.metal_kernel(
            name=f"nbody_pot_{tg}_{int(precise)}",
            input_names=["x", "mass", "hsml", "G"],
            output_names=["out"],
            source=src,
        )
    return _CACHE[key]


def potential_gpu(x, m, softening, G=1.0, tg=256, precise=False):
    """FP32 brute-force potential on the GPU.  x (N,3), m (N,), softening (N,) numpy arrays."""
    n = x.shape[0]
    # keep the (N, 3) shape: the kernel reads the buffer flat as x[3*i+k] but needs x_shape[0] == N
    xg = mx.array(np.ascontiguousarray(x, dtype=np.float32))
    mg = mx.array(np.ascontiguousarray(m, dtype=np.float32))
    hg = mx.array(np.ascontiguousarray(softening, dtype=np.float32))
    Gg = mx.array(np.array([G], dtype=np.float32))
    nblk = (n + tg - 1) // tg
    (out,) = _kernel(tg, precise)(
        inputs=[xg, mg, hg, Gg],
        grid=(nblk * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out)
    return out


# --------------------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------------------


def plummer(n, seed=42):
    """Plummer sphere positions, unit total mass, in the style of tests/test_plummer.py."""
    rng = np.random.default_rng(seed)
    r = 1.0 / np.sqrt(rng.random(n) ** (-2.0 / 3.0) - 1.0)
    u = 2.0 * rng.random(n) - 1.0
    phi = 2.0 * np.pi * rng.random(n)
    s = np.sqrt(1.0 - u * u)
    x = np.column_stack([r * s * np.cos(phi), r * s * np.sin(phi), r * u])
    return np.ascontiguousarray(x), np.repeat(1.0 / n, n)


def bench(fn, *a, reps=3, **kw):
    """Return (best wall time, last result).  One untimed warm-up absorbs JIT/kernel compile."""
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
    p.add_argument("-n", "--sizes", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 131072, 262144])
    p.add_argument("--softening", type=float, default=0.01, help="uniform softening; 0 disables")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tg", type=int, default=256, help="threadgroup size / source tile")
    p.add_argument("--precise", action="store_true", help="use 1/sqrt instead of hardware rsqrt")
    p.add_argument("--skip-cpu-above", type=int, default=10**9)
    args = p.parse_args()

    from numba import get_num_threads

    from pytreegrav.bruteforce import Potential_bruteforce_parallel
    from pytreegrav.bruteforce_symmetric import Potential_bruteforce_symmetric

    print(f"device={mx.default_device()}  numba_threads={get_num_threads()}  softening={args.softening}")
    print(f"tg={args.tg}  rsqrt={'precise 1/sqrt' if args.precise else 'hardware'}  reps={args.reps} (best-of)")
    print()
    hdr = f"{'N':>8} {'CPU par':>10} {'CPU sym':>10} {'GPU fp32':>10} {'vs par':>8} {'vs sym':>8} {'Gpair/s':>9} {'rel err':>9}"
    print(hdr)
    print("-" * len(hdr))

    for n in args.sizes:
        x, m = plummer(n)
        h = np.repeat(args.softening, n)

        t_gpu, phi_gpu = bench(potential_gpu, x, m, h, reps=args.reps, tg=args.tg, precise=args.precise)
        phi_gpu = np.array(phi_gpu, copy=False)

        if n <= args.skip_cpu_above:
            t_par, phi_par = bench(Potential_bruteforce_parallel, x, m, h, reps=args.reps)
            t_sym, _ = bench(Potential_bruteforce_symmetric, x, m, h, reps=args.reps)
            sp, ss, err = f"{t_par / t_gpu:>7.1f}x", f"{t_sym / t_gpu:>7.1f}x", f"{rel_err(phi_gpu, phi_par):>9.2e}"
            cp, cs = f"{t_par:>9.4f}s", f"{t_sym:>9.4f}s"
        else:
            sp = ss = err = f"{'-':>8}"
            cp = cs = f"{'skipped':>10}"

        print(f"{n:>8} {cp} {cs} {t_gpu:>9.4f}s {sp} {ss} {n * n / t_gpu / 1e9:>9.1f} {err}")


if __name__ == "__main__":
    main()
