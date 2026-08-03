# experiments/gpu

Prototypes and measurements that are **not part of the package**. Nothing here is imported by
`pytreegrav`, covered by the test suite, or wired into the frontend. Kept because the negative
results are worth as much as the positive one.

Requires `mlx` (`pip install mlx`), which is Apple-silicon only and is deliberately *not* in
`requirements.txt`. All numbers below are from an Apple M3 Max (40-core GPU, 16 numba threads),
MLX 0.32.0, numba 0.66, macOS 26.5.1. **The machine was not idle**: a backup daemon held ~200% CPU
(2 of 16 cores) throughout, so the CPU times are inflated by perhaps 10% and every speedup quoted
here is correspondingly optimistic. The N≤16k rows are additionally noisy — CPU times there are a
few ms, and a repeat run moved the N=8192 speedup between 5.2× and 7.2×. Trust the N≥32k rows.

## The question

FP32 brute-force potential on the GPU, timed against the shipped parallel CPU kernels. This is
step 1 of the "measure before building" plan: does GPU FP32 buy enough to justify a second
implementation of anything?

## The answer

Yes for brute force — 8–16× unsoftened, up to 63× heavily softened — **and it does not matter,
because brute force is not the algorithm anyone runs above N≈1000.** Against the shipped CPU
*tree*, the GPU brute force is ahead only below N≈50k and loses by 14× at N=1e6. See "The
comparison that decides it".

Everything here is single precision. Metal has no `float64` at all, so on this hardware there is
no FP64 path to measure — not a slow one, none.

## Correctness first

`bruteforce_metal.py` reproduces `Potential_bruteforce_parallel` exactly in structure: cubic-spline
`PotentialKernel`, `h_ij = max(h_i, h_j)`, `if r < h` / `elif r > 0`, self-interaction skipped.
Verified against the serial `Potential_bruteforce` at N=3000:

| case | softened pairs | of those, `q<=0.5` | rel err (hw rsqrt) | rel err (`1/sqrt`) |
| --- | --- | --- | --- | --- |
| `h = 0` | 0% | — | 7.8e-07 | 7.8e-07 |
| `h = 0.01` | ~0% | 100% | 7.8e-07 | 7.8e-07 |
| `h = 2.0` | 44.8% | 28.1% | 8.3e-07 | 8.3e-07 |
| `h = 10` | 96.1% | 90.2% | 7.4e-07 | 7.4e-07 |
| `h` variable | 43.1% | 33.0% | 8.1e-07 | 8.1e-07 |

Also checked: coincident particles at `h=0` and `h=0.5` (4.5e-07 / 3.9e-07), tile-size invariance
over `tg` = 64…1024 (identical to 3 digits), stability across repeated calls on the same input,
and non-power-of-2 `N` (1, 2, 999, 1001, 4097).

## Throughput, unsoftened (`h = 0.01`, best-of-5)

| N | CPU parallel | CPU symmetric | GPU fp32 | vs par | vs sym | Gpair/s | rel err |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 14.3 ms | 5.4 ms | 2.0 ms | 7.2× | 2.7× | 34.0 | 1.3e-06 |
| 16384 | 34.0 ms | 20.0 ms | 3.5 ms | 9.8× | 5.7× | 77.1 | 1.9e-06 |
| 32768 | 145.4 ms | 73.9 ms | 12.5 ms | 11.6× | 5.9× | 86.0 | 2.6e-06 |
| 65536 | 509.9 ms | 266.7 ms | 41.6 ms | 12.3× | 6.4× | 103.2 | 3.7e-06 |
| 131072 | 2408 ms | 1286 ms | 151.4 ms | **15.9×** | **8.5×** | 113.5 | 5.2e-06 |
| 262144 | — | — | 592.5 ms | — | — | 116.0 | — |
| 524288 | — | — | 2361 ms | — | — | 116.4 | — |

Plateaus at ~116 Gpair/s. At ~14 flop/pair (counting `rsqrt` as one) that is ~1.6 TFLOP/s, or
**~11% of the M3 Max's ~14 TFLOP/s FP32 peak** — respectable for a first cut with two branches
in the inner loop, and clearly not the ceiling.

FP32 accumulation error grows as √N (1.3e-06 at 8k → 5.2e-06 at 131k, i.e. 3.9× for 16× N — a
textbook random walk). Still two decades below the θ=0.7 tree truncation error, and it would be
removed by pairwise or Kahan accumulation if it ever mattered.

## Throughput, heavily softened (`h = 2.0`, 45% of pairs softened)

| N | CPU parallel | CPU symmetric | GPU fp32 | vs par | vs sym |
| --- | --- | --- | --- | --- | --- |
| 32768 | 721 ms | 440 ms | 12.5 ms | 57.9× | 35.3× |
| 131072 | 12082 ms | 7405 ms | 191 ms | **63.2×** | **38.8×** |

**This is the most interesting result here.** Going from ~0% to 45% softened pairs costs the CPU
**5.0×** and the GPU **1.26×**. On the GPU the three-way spline is predicated, and a threadgroup's
softenings are similar enough that divergence is minimal.

That is precisely the regime where `../vectorized_bruteforce` dead-ended: SIMD gave only 1.32–1.38×
softened on AVX-512, and **0.42–0.47× (i.e. slower)** on this same ARM hardware. The GPU's margin
is largest exactly where CPU vectorization failed, which is the one genuinely new fact in this
directory.

## The comparison that decides it

GPU brute force is the wrong opponent. Against `Potential(..., method="tree", parallel=True)`,
θ=0.7, on the same 16 threads:

| N | CPU tree | GPU brute force fp32 | GPU/tree |
| --- | --- | --- | --- |
| 32768 | 19.5 ms | 12.6 ms | **0.6×** |
| 131072 | 65.0 ms | 161 ms | 2.5× slower |
| 262144 | 156 ms | 636 ms | 4.1× slower |
| 524288 | 342 ms | 2851 ms | 8.3× slower |
| 1048576 | 895 ms | 12271 ms | **13.7× slower** |

Crossover is around **N ≈ 50k**. A 15× constant factor against O(N²) buys roughly one octave of N
before O(N log N) takes all of it back. So the shipped CPU tree beats a GPU brute force at every
size anyone cares about, and the GPU brute-force path is only interesting for the `*Target` case
with few targets and many sources, or for heavily-softened work below ~50k.

## How you write the kernel is worth 30×

`bruteforce_mlx_arrayops.py` was the first attempt: the same math expressed as a chain of MLX
array ops over (target, source) tiles, `mx.compile`d. It is **numerically correct** — 6e-08
against the CPU — and it runs at **4 Gpair/s**:

| N | CPU parallel | GPU (array ops) | vs par | vs sym | Gpair/s |
| --- | --- | --- | --- | --- | --- |
| 32768 | 134.6 ms | 261.4 ms | 0.52× | 0.24× | 4.11 |
| 131072 | 2310 ms | 4487 ms | 0.51× | 0.27× | 3.83 |

**Slower than the CPU, and 29× slower than the Metal kernel doing identical arithmetic.**
`mx.compile` does not fuse the trailing `sum(axis=1)` into the tile body, so each tile streams
~15 `(1024, 8192)` float32 temporaries through memory: O(1) flop/byte, bandwidth-bound. The fix is
structural, not a tuning knob — one thread per target, accumulator in a register, sources staged
through threadgroup memory so each source load is reused `tg` times, giving O(tg) flop/byte.

Kept because a correct-but-slow prototype is the easiest way to get a wrong answer to the *design*
question. Any GPU number quoted for this library should say which of these two shapes produced it.

## One kernel for Metal *and* NVIDIA: cheap, and mostly a red herring

`bruteforce_gpu_portable.py` answers the portability question. MLX exposes `mx.fast.metal_kernel`
and `mx.fast.cuda_kernel` with an **identical Python-side contract** — same constructor args
(`name`/`input_names`/`output_names`/`source`/`header`), same call signature
(`inputs`/`template`/`grid`/`threadgroup`/`output_shapes`/`output_dtypes`), and `grid` counted in
*threads* rather than blocks on both. So the entire harness is shared and only the kernel dialect
differs:

| concept | Metal | CUDA |
| --- | --- | --- |
| global thread id | `thread_position_in_grid.x` | `blockIdx.x*blockDim.x + threadIdx.x` |
| local thread id | `thread_position_in_threadgroup.x` | `threadIdx.x` |
| shared declaration | `threadgroup float s[N];` | `__shared__ float s[N];` |
| barrier | `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads()` |
| rsqrt / sqrt / max | `metal::rsqrt` / `metal::sqrt` / `metal::max` | `rsqrtf` / `sqrtf` / `fmaxf` |

Eight `#define`s plus a two-line preamble absorb all of it. Of 62 generated lines, 18 differ
between the two backends and **16 of those are the shim itself** — the ~40-line physics inner loop
is byte-identical. On Metal the portable version produces results **bit-identical** to the
hand-written `bruteforce_metal.py` (rel err exactly 0.0) at 10.1×/14.3× over the CPU at
N=32768/131072, i.e. the abstraction costs nothing measurable.

**The CUDA path is written but unverified.** This machine has no NVIDIA GPU and no `nvcc`, and the
installed MLX is a Metal-only build, so `mx.fast.cuda_kernel` exists in the API but cannot execute
or even compile here. Treat that source as a strong first draft.

Three caveats before believing "works well on both":

- **The thread indices cannot be macro-hidden.** MLX builds the kernel signature by scanning the
  `source` string for Metal attribute names, so if `thread_position_in_grid` appears only after
  macro expansion, MLX never declares it and the Metal compiler dies on an undeclared identifier.
  That is why the shim has a two-line preamble instead of two more `#define`s. Found the hard way.
- **Portable source is not portable performance.** ~11% of FP32 peak on Apple came from one thread
  per target with `tg=256`. NVIDIA will want register blocking — several targets per thread, the
  i-loop unrolled — to approach peak, because its FMA-to-load ratio and register file are different.
  Expect one source but two tuning constants, and do not quote an Apple-tuned number as an NVIDIA
  result.
- **Portability forces FP32.** Metal has no `float64`, so the common denominator across both targets
  is single precision by construction. A portable kernel and an FP64 kernel are mutually exclusive
  here; CUDA-only is the price of FP64.

Alternatives considered for one-source-many-backend, none better for this: **Taichi** is the only
genuine write-once option covering CUDA + Metal + Vulkan, but it is FP32-only on Metal and project
momentum has visibly declined; **OpenCL** is the historically correct answer and runs on both, but
Apple deprecated it in 2018 and froze it at 1.2; **wgpu/WebGPU** is genuinely portable and FP32-only
but immature for HPC; **Triton** and **JAX Pallas** have no Metal backend at all. For a 40-line
kernel, an 8-macro shim beats adopting any of them.

**But note what this is portability *for*.** It is a brute-force kernel that loses to the shipped
CPU tree above N≈50k. The result that matters is not "brute force is portable", it is that the
harness and the shim are now proven, which de-risks the only port that could pay — the tree walk.
The shim will hurt there in a way it does not here: a warp-synchronous walk needs vote/ballot
intrinsics, and CUDA's `__ballot_sync`/`__any_sync` versus Metal's `simd_ballot`/`simd_any` are a
real semantic seam, not a token substitution. That is the next thing to prototype if anyone cares.

## Notes and dead ends

- **Hardware `rsqrt` is worth only ~1.25×**, not the order-of-magnitude the FP32-vs-FP64 argument
  might suggest: 116 vs 99 Gpair/s at N=262144, 116 vs 92 at N=524288 (`--precise`). Metal's
  `1.0f/sqrt(x)` evidently already lowers to rsqrt plus a refinement step. Accuracy is
  indistinguishable (7.8e-07 either way), so the "precise" variant costs 20% for nothing
  measurable — but it also means the rsqrt is not the load-bearing part of the FP32 story here.
- **Threadgroup size barely matters.** `tg` = 128/256/512/1024 are within noise; 64 is slightly
  worse. Left at 256.
- **No `float64` on Metal**, so this cannot be extended to price the FP64 path. That measurement
  needs CUDA hardware and `numba.cuda`.
- **Unified memory means no transfer is measured.** There is no host→device copy on Apple silicon.
  A discrete GPU pays PCIe on top of every number above — irrelevant for one-shot calls at these
  sizes (~25 ms for 1e7 particles on PCIe4 against seconds of compute), but it makes these numbers
  a mild over-estimate of what a 4090 would show for the same work.
- **A bug worth recording**, because it produced plausible-looking wrong answers: passing the
  positions raveled to `(3N,)` makes `x_shape[0]` equal `3N`, so the kernel loops over 3× too many
  sources and reads past the buffers. It validated at 7.8e-07 **on the first call** and then
  degraded (1e+06, 1e+13, 1e+18, 1e+24 over successive calls) as the Metal allocator pool got
  dirty — fresh buffers are zeroed, so the out-of-range sources initially contributed zero mass.
  A first-call-only correctness check would have passed it. Keep `x` as `(N, 3)`.

## Before any of this ships

- **The measured win does not justify shipping anything.** 8–16× on a kernel the frontend only
  dispatches to below N≈1000-4000, which loses to the shipped tree above N≈50k. The softened 63×
  is the only number that argues for a narrow GPU brute-force path, and it wants a real use case
  attached before it earns a dependency.
- **`mlx` is Apple-only and `numba.cuda` is NVIDIA-only**, so a GPU path means either two backends
  or a platform commitment. Neither belongs in `requirements.txt`; if anything ships it should be
  a separate optional package.
- **CI cannot test it.** `.github/workflows/tests.yml` runs `ubuntu-latest`; there are no free GPU
  runners. Given the class of bug this directory already contains — correct on the first call,
  garbage afterwards — untested GPU code is a poor bet.
- FP32 accumulation would need its own tolerance (~1e-5 at N~1e5, scaling as √N), not the suite's
  1e-12, plus a note that the accumulator is the limit rather than the pair math.

## Files

## The ray tracer: a better target than brute force

This started as a spike here and **graduated into the package** as `pytreegrav.cuda`, wired up as
`ColumnDensity(..., device="cuda")` and the `CudaColumnDensity` context, with an optional
`pip install pytreegrav[cuda]` extra. The prototype file has been removed so it cannot drift from the
shipped code; what stays here is the measurement record.

Ray-traced column density is the workload that actually justifies a GPU: unlike gravity it has no
FP64 story to lose (uniform spheres are already far cruder than FP32), and unlike brute force it does
not lose to the shipped CPU tree.

Three design points, all measured on the CPU side first:

- **Not a port of `ColumnDensity_grouped`.** Grouping won ~3x on CPU by amortizing gathers across 16
  targets, and paid for it by inflating the acceptance reach by the group extent -- leaf tests went
  from 24 to 1389 per target. A warp gets the same amortization 32-fold for free: 32 Morton-adjacent
  targets on one ray hold the same node index for nearly the whole descent, so the row fetch
  broadcasts, while each lane still tests its exact ray. So the kernel is the *simpler* per-target
  walk, and the port is smaller than it looks.
- **The packed row is where the performance is.** Seven scattered float64 SoA gathers per node visit
  become one 32 B contiguous read: `(NumNodes, 8) float32` plus an int32 link array. NumNodes = 1.5 N,
  so the tree goes 84 MB -> 48 MB at N=1e6. Slots are overloaded between leaves and nodes, and 1/h,
  1/h^2, the leaf prefactor and the squared node reach are all precomputed, leaving the kernel with
  **no divisions and no sqrt on the node path**.
- **No warp intrinsics.** One independent cursor per thread, so the `__ballot_sync` / `simd_ballot`
  seam flagged above never arises. This is the port the shim work de-risked.

### The float32 finding, which is the important one

The first GPU run was **wrong by 2%** at N=20000, against 1.98e-06 for the same float32 rows walked in
float64 arithmetic. Cause: the shipped walk gets the squared impact parameter as `r^2 - z^2`, which
cancels catastrophically for a nearly radial ray -- relative error ~ `eps (r/b)^2`. In float64 that is
a harmless 3.6e-12 (it is why `test_clustered_configuration` sits at 1e-9 rather than 1e-12). In
float32 the same `(r/h)^2 ~ 2000` at N=2e4 gives `eps (r/h)^2 = 2.4e-04`, which the chord derivative
then amplifies near the sphere edge.

Computing `b = d - (d.n) n` and `|b|^2` instead fixes it: **2.00e-02 -> 3.27e-06**, a 6000x
improvement, and the residual is now float32 *storage* error rather than arithmetic. That form was
measured at +12.8% on the CPU earlier and rejected as not worth digits 12-16 of an extinction
estimate. That judgement still holds for float64 -- but the well-conditioned form is a **prerequisite**
for any float32 or GPU path, not an optimization. On a latency-bound kernel its ~9 extra flops are
free.

### Validation

`--validate` reports both precisions, which separates two questions:

| rows | N=2000 | N=20000 | what it measures |
| --- | --- | --- | --- |
| float64 | 2.72e-14 | 8.86e-14 | limited by the *reference* -- the shipped walk's own `r^2-z^2` conditioning |
| float32 | 6.54e-07 | 1.98e-06 | what narrowing storage costs, on its own |
| cuda float32 | 7.98e-07 | 3.27e-06 | storage + float32 arithmetic + the launch path |

Zero-radius particles pack to a zeroed row and contribute nothing, matching the CPU walk; the packer
raises rather than shipping `inf` if `M/h^2` overflows float32 (checked: fires on `h=1e-22`).

### Results: RTX A6000 vs 32 Xeon Gold 6244 threads

ccalin030, driver 580.142, numba 0.66.0 + numba-cuda 0.30.4, CC 8.6, 84 SMs. CPU column is the shipped
`ColumnDensity(rays=6, parallel=True)` -- i.e. the grouped walk at `group_size=16` -- on the same node.
All JIT paths warmed first; `build` and `pack+upload` are one-off per tree.

| N | tree build | pack+upload | CPU 32t | CUDA | speedup | max rel |
| --- | --- | --- | --- | --- | --- | --- |
| 1e5 | 17 ms | 15 ms | 328 ms | **28 ms** | **11.8x** | 9.6e-06 |
| 1e6 | 160 ms | 179 ms | 7643 ms | **507 ms** | **15.1x** | 8.4e-06 |
| 3e6 | 486 ms | 511 ms | 40659 ms | **2230 ms** | **18.2x** | 8.0e-06 |

**Clears the 10x criterion**, and the margin grows with N -- there is more work to fill 84 SMs with,
while the CPU is increasingly cache-limited (its cost goes as N^(4/3)). Note the CPU number at N=3e6
is 41 s, which is the scaling this was meant to attack.

Secondary measurements:

- **`threads_x = 32` is best** -- 486 ms against 547 at 128, at N=1e6. One warp per block: no partly
  divergent warps at block boundaries, and 187k blocks for the scheduler to balance over 84 SMs.
- **Transfers are 2% of a call.** Kernel 554 ms vs 568 ms for the full call including target upload and
  result download. The 60 MB tree upload (136 ms) is one-off, which is what the context object is for.
- **Ray count scales linearly**: 92.6 / 102.6 / 104.1 / 108.3 ms per ray at 6 / 12 / 48 / 192 rays
  (N=1e6). So healpix nside=4 is ~21 s on the GPU against ~250 s on the CPU.
- Throughput is 9-12 M walks/s, roughly flat in ray count.

## Files

| file | what it is |
| --- | --- |
| `bruteforce_metal.py` | **The useful outcome.** Hand-written Metal kernel via `mx.fast.metal_kernel`; one thread per target, sources staged through threadgroup memory. 116 Gpair/s. `--precise` prices the hardware rsqrt, `--tg` the threadgroup size. |
| `bruteforce_gpu_portable.py` | Same kernel behind a Metal/CUDA shim: 8 `#define`s + a 2-line preamble, physics byte-identical. Metal path validated bit-identical to the above at no cost; **CUDA path unverified** (no NVIDIA hardware here). `--emit {metal,cuda}` prints either generated source. |
| `bruteforce_mlx_arrayops.py` | Superseded first attempt: identical math as MLX array ops. Correct to 6e-08 and *slower than the CPU*. Kept for the 29× implementation-shape result. |
