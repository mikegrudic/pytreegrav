"""One brute-force potential kernel body, dispatched to Metal or CUDA through a token shim.

Same math and same result as bruteforce_metal.py -- see that file for the algorithm.  The point
here is portability: MLX exposes ``mx.fast.metal_kernel`` and ``mx.fast.cuda_kernel`` with an
*identical* Python-side contract (same constructor args, same inputs/grid/threadgroup/output_shapes
call, and ``grid`` counted in threads rather than blocks on both).  So the whole harness is shared
and the only thing that differs is the dialect of the kernel body:

    concept              Metal                                    CUDA
    global thread id     thread_position_in_grid.x                blockIdx.x*blockDim.x+threadIdx.x
    local thread id      thread_position_in_threadgroup.x         threadIdx.x
    shared decl          threadgroup float s[N];                  __shared__ float s[N];
    barrier              threadgroup_barrier(mem_flags::...)      __syncthreads()
    rsqrt / sqrt / max   metal::rsqrt / metal::sqrt / metal::max  rsqrtf / sqrtf / fmaxf

That is eight tokens.  They are absorbed by a per-backend prelude of #defines passed via MLX's
``header=`` argument, leaving ONE copy of the actual physics.  ``--emit`` prints either variant.

STATUS: the Metal path is measured and validated (see README.md).  The CUDA path is
**written but unverified** -- this machine has no NVIDIA GPU, and the installed MLX is a
Metal-only build, so ``mx.fast.cuda_kernel`` exists in the API but cannot execute here.  Treat the
CUDA source as a strong first draft, not a measurement.  The two things most likely to need
adjustment on first contact with real hardware are flagged with CHECK: comments below.
"""

import argparse

import mlx.core as mx
import numpy as np

# --------------------------------------------------------------------------------------------------
# Per-backend prelude.  Everything platform-specific lives here and nowhere else.
# --------------------------------------------------------------------------------------------------

_PRELUDE = {
    "metal": """
#define TGSIZE {tg}
#define SHARED  threadgroup
#define BARRIER threadgroup_barrier(mem_flags::mem_threadgroup)
#define RSQRT(x)  metal::rsqrt(x)
#define SQRTF(x)  metal::sqrt(x)
#define FMAXF(a,b) metal::max(a,b)
#define UMIN(a,b)  metal::min(a,b)
""",
    "cuda": """
#define TGSIZE {tg}
#define SHARED  __shared__
#define BARRIER __syncthreads()
#define RSQRT(x)  rsqrtf(x)
#define SQRTF(x)  sqrtf(x)
#define FMAXF(a,b) fmaxf(a,b)
#define UMIN(a,b)  min(a,b)
""",
}

# The thread indices canNOT be macro-hidden.  MLX builds the kernel signature by scanning the
# `source` string for Metal attribute names, so if `thread_position_in_grid` only ever appears after
# macro expansion MLX never declares it and the Metal compiler fails on an undeclared identifier.
# Hence a two-line per-backend preamble prepended to the shared body -- the literal tokens stay in
# `source` where the scanner can see them.  This is the one genuine seam between the two targets.
_PREAMBLE = {
    "metal": """
    uint tid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
""",
    "cuda": """
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint lid = threadIdx.x;
""",
}

# --------------------------------------------------------------------------------------------------
# The kernel body -- one copy, both backends.
#
# N is passed as an explicit input rather than read from `x_shape[0]`.  Two reasons: MLX's shape
# injection is documented for the Metal target and only implied for CUDA (CHECK: if `<name>_shape`
# does exist on CUDA this could be dropped), and reading a shape is how the earlier version of this
# kernel acquired a buffer-overrun bug -- see the dead-ends section of README.md.
# --------------------------------------------------------------------------------------------------

_BODY = """
    uint N = (uint)nparticles[0];

    SHARED float sx[TGSIZE];
    SHARED float sy[TGSIZE];
    SHARED float sz[TGSIZE];
    SHARED float sm[TGSIZE];
    SHARED float sh[TGSIZE];

    // clamp rather than early-return: every thread must reach the barriers below
    uint i = tid < N ? tid : 0;
    float xi = x[3 * i + 0];
    float yi = x[3 * i + 1];
    float zi = x[3 * i + 2];
    float hi = hsml[i];
    float phi = 0.0f;

    for (uint base = 0; base < N; base += TGSIZE) {
        uint j = base + lid;
        bool valid = j < N;
        uint jc = valid ? j : 0;
        sx[lid] = x[3 * jc + 0];
        sy[lid] = x[3 * jc + 1];
        sz[lid] = x[3 * jc + 2];
        sm[lid] = valid ? mass[jc] : 0.0f;   // zero mass makes padding lanes inert
        sh[lid] = hsml[jc];
        BARRIER;

        uint kmax = UMIN((uint)TGSIZE, N - base);
        for (uint k = 0; k < kmax; ++k) {
            if (base + k == i) continue;             // CPU: `if i == j: continue`
            float dx = xi - sx[k];
            float dy = yi - sy[k];
            float dz = zi - sz[k];
            float r2 = dx * dx + dy * dy + dz * dz;
            float hh = FMAXF(hi, sh[k]);
            float mj = sm[k];
            if (r2 < hh * hh) {                      // CPU: `if r < h`
                float r = SQRTF(r2);
                float hinv = 1.0f / hh;
                float q = r * hinv;
                float q2 = q * q;
                float kk;
                if (q <= 0.5f) {
                    kk = (-2.8f + q2 * (5.33333333333333f + q2 * (6.4f * q - 9.6f))) * hinv;
                } else {
                    kk = (-3.2f + 0.0666666666666667f / q
                          + q2 * (10.6666666666667f + q * (-16.0f + q * (9.6f - 2.13333333333333f * q)))) * hinv;
                }
                phi += mj * kk;
            } else if (r2 > 0.0f) {                  // CPU: `elif r > 0`
                phi += -mj * RSQRT(r2);
            }
        }
        BARRIER;
    }
    if (tid < N) out[tid] = phi * G[0];
"""

_IN_NAMES = ["x", "mass", "hsml", "G", "nparticles"]
_CACHE = {}


def backend():
    """'metal' or 'cuda', from the MLX build actually installed."""
    if hasattr(mx, "metal") and mx.metal.is_available():
        return "metal"
    return "cuda"


def source_for(be, tg):
    """(header, source) for backend ``be``.  Only the 2-line preamble and the #defines differ."""
    return _PRELUDE[be].format(tg=tg), _PREAMBLE[be] + _BODY


def _kernel(be, tg):
    key = (be, tg)
    if key not in _CACHE:
        header, body = source_for(be, tg)
        # CHECK: identical signature on both targets, per the mx.fast.{metal,cuda}_kernel docs.
        factory = mx.fast.metal_kernel if be == "metal" else mx.fast.cuda_kernel
        _CACHE[key] = factory(
            name=f"nbody_pot_portable_{tg}",
            input_names=_IN_NAMES,
            output_names=["out"],
            source=body,
            header=header,
        )
    return _CACHE[key]


def potential_gpu(x, m, softening, G=1.0, tg=256, be=None):
    """FP32 brute-force potential.  x (N,3), m (N,), softening (N,) numpy arrays."""
    be = be or backend()
    n = x.shape[0]
    inputs = [
        mx.array(np.ascontiguousarray(x, dtype=np.float32)),
        mx.array(np.ascontiguousarray(m, dtype=np.float32)),
        mx.array(np.ascontiguousarray(softening, dtype=np.float32)),
        mx.array(np.array([G], dtype=np.float32)),
        mx.array(np.array([n], dtype=np.int32)),
    ]
    nblk = (n + tg - 1) // tg
    (out,) = _kernel(be, tg)(
        inputs=inputs,
        grid=(nblk * tg, 1, 1),  # threads, not blocks -- same convention on both targets
        threadgroup=(tg, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    mx.eval(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", choices=["metal", "cuda"], help="print the generated source and exit")
    p.add_argument("-n", "--sizes", type=int, nargs="+", default=[32768, 131072])
    p.add_argument("--softening", type=float, default=0.01)
    p.add_argument("--tg", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()

    if args.emit:
        header, body = source_for(args.emit, args.tg)
        print(f"// ==== {args.emit} ====\n{header}\n{body}")
        return

    from bruteforce_metal import bench, plummer, rel_err
    from bruteforce_metal import potential_gpu as potential_metal_direct

    from pytreegrav.bruteforce import Potential_bruteforce_parallel

    be = backend()
    print(f"backend={be}  device={mx.default_device()}  tg={args.tg}  reps={args.reps} (best-of)")
    print(f"{'N':>8} {'CPU par':>10} {'portable':>10} {'speedup':>8} {'vs CPU':>9} {'vs direct':>10}")
    print("-" * 60)
    for n in args.sizes:
        x, m = plummer(n)
        h = np.repeat(args.softening, n)
        t_cpu, phi_cpu = bench(Potential_bruteforce_parallel, x, m, h, reps=args.reps)
        t_gpu, phi_gpu = bench(potential_gpu, x, m, h, reps=args.reps, tg=args.tg)
        phi_direct = np.array(potential_metal_direct(x, m, h, tg=args.tg))
        phi_gpu = np.array(phi_gpu, copy=False)
        print(
            f"{n:>8} {t_cpu:>9.4f}s {t_gpu:>9.4f}s {t_cpu / t_gpu:>7.1f}x "
            f"{rel_err(phi_gpu, phi_cpu):>9.2e} {rel_err(phi_gpu, phi_direct):>10.2e}"
        )


if __name__ == "__main__":
    main()
