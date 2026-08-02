# experiments

Prototypes and measurements that are **not part of the package**. Nothing here is imported by
`pytreegrav`, covered by the test suite, or wired into the frontend. Kept because the negative
results are worth as much as the positive one — several obvious-looking ideas were tried and
measured and do not work.

All numbers below are from `ccalin030` (2× Intel Xeon Gold 6244, Cascade Lake, 16 physical
cores / 32 threads, AVX-512) on an idle machine, numba 0.66 / llvmlite 0.48, unless stated.

## The question

The shipped brute-force kernels are scalar. Can direct summation be vectorized?

## The answer

Yes, and it needs three things — each established separately:

1. **Remove the divide.** The inner loop carries an `fdiv`, and LLVM's loop vectorizer refuses
   any loop containing one. Delete it from an otherwise identical loop and it emits 24 packed
   ops and runs **5.1× faster** (53.6 → 10.5 µs / 20k interactions). That 5× is the whole
   prize. Replaced by a Newton-Raphson reciprocal square root built from multiplies only;
   four Newton steps give **7.6e-15** relative error, i.e. full double precision.
2. **Remove the softening branch.** `if r < h` is data-dependent and blocks vectorization on
   its own. It can be removed rather than avoided: evaluate all three spline pieces and blend
   them with selects, which LLVM if-converts. See "Softening works too" below.
3. **Avoid a runtime loop lower bound in a loop that also stores.** This one is subtle and was
   the last thing found — see below.

## The subtle one

`bruteforce_avx.py` was written believing symmetry and vectorization were mutually exclusive,
because adding the `out[j] -=` scatter took the row loop from 56 packed ops to 0. That was
wrong. What actually blocks numba's vectorizer is the *combination* of a runtime loop lower
bound and a store:

| loop form | packed ops |
| --- | --- |
| `for j in range(0, n)` + store | 60 — vectorizes |
| `for j in range(i0, n)` + store | **0 — does not** |
| `for j in range(i0, n)`, no store | 56 — vectorizes |
| slice so the loop starts at 0, + store | 60 — vectorizes |

Either ingredient alone is fine. The symmetric algorithm needs `range(i+1, n)`, which is
exactly the bad case. Passing the row kernel a *slice* starting at `i+1` so its loop runs from
zero restores vectorization at negligible cost (one array-struct construction per row).

`bruteforce_sym_vec.py` does that, and keeps the symmetry:

| N | shipped | unsoftened | speedup | shipped | softened h=0.05 | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 5000 | 5.9 ms | 1.6 ms | 3.65× | 4.1 ms | 3.1 ms | 1.32× |
| 20000 | 61.2 ms | 21.4 ms | 2.86× | 61.1 ms | 44.7 ms | 1.37× |
| 50000 | 402.3 ms | 166.0 ms | 2.42× | 412.4 ms | 298.5 ms | 1.38× |

Softening works too. `ForceKernel`'s `0.0667/q³` needs no divide because `1/q = h·rinv`, and
the three-way spline branch becomes selects — matching `ForceKernel` to 9.5e-15. It gains
less because every lane evaluates all three pieces regardless, and supporting `h == 0` in the
same kernel costs a further ~20%. Momentum is conserved to 1.1e-17, so the symmetry is real.

Accuracy/speed vs Newton steps (N=20000, unsoftened / softened):
`1 → 3.71×/2.31× at 3e-03`, `2 → 3.53×/2.03× at 6e-06`, `3 → 3.11×/1.89× at 4e-11`,
`4 → 2.88×/1.68× at 4e-15`. Three steps is the sweet spot unless a bit-level reference is
needed; one step is never worth it.

Single-threaded at N=20000 it runs in **415 ms against 390 ms for equivalent C++** — within
7%. So this was never an LLVM limitation, and not structural to numba either; just an
avoidable code pattern.

## Files

| file | what it is |
| --- | --- |
| `bruteforce_sym_vec.py` | **The useful outcome.** Symmetric *and* vectorized, softened *and* unsoftened. Supersedes `bruteforce_avx.py`. |
| `bruteforce_avx.py` | Superseded first attempt: vectorized by giving up symmetry, unsoftened only. 2.0–3.0× on x86, **0.68–0.76× (a loss) on ARM/NEON**. Kept for the ARM measurement. |
| `alias_test.cpp` | C++ control. Shows LLVM vectorizes the scatter loop with *or without* `__restrict`. |
| `stride_test.cpp` | C++ bisection: rules out runtime strides and branch-vs-ternary. |
| `i0_test.py` | Isolates the runtime-lower-bound + store blocker, and the slice fix. |
| `minimal.py` | Rules out loop *shape*: reduction, store, RMW, and all three combined vectorize fine. |
| `bodysize.py` | Rules out loop-body size. |
| `alias_probe.py` | Rules out alias-check budget (one scatter array is already enough to block). |
| `avx_ceiling.py` | Quantifies the headroom: divide ≈ 80% of runtime; float32 buys only 1.13×. |
| `bench_avx.py`, `verify_avx.py` | Benchmark / correctness-codegen-momentum checks for `bruteforce_avx`. |

## Dead ends, so nobody repeats them

- **SVML** — numba's documented route to vectorized div/sqrt. `libsvml.so` installs fine, but
  llvmlite 0.48 reports `has_svml: False` on both PyPI *and* conda-forge, and the `defaults`
  channel will not resolve. Unavailable in practice.
- **`NUMBA_SLP_VECTORIZE=1`** — numba disables SLP by default where clang enables it, so this
  looked promising. Moves the scatter loop from 0 to 1 packed op. Nothing.
- **Manual 4-wide unrolling** to trigger the SLP vectorizer — 0 `vdivpd`, no speedup.
- **numpy whole-array expressions inside `njit`** — 2.5× *slower* (temporaries).
- **`r2 ** -0.5`** — compiles to exactly the same sqrt+divide; no `vrsqrt14pd` even on
  AVX-512 hardware.
- **float32** — only 1.13×, because the divide blocks vectorization in single precision too.
- **`__restrict` in C++** — zero effect; LLVM already versions the loop.

## Before any of this ships

- **ARM/NEON is measured, and it does not transfer.** On an M-series Mac (16 threads),
  `bruteforce_sym_vec` gets only **1.05–1.28× unsoftened** and is **0.42–0.47× — i.e. ~2.2×
  slower — softened** (two independent runs agreeing to within 0.06×; an earlier pass on a
  busier machine reported a flatteringly high 1.18–1.33× / 0.50–0.57× because the background
  load was slowing the *shipped* kernel). Both paths still vectorize (99 packed `.2d` ops) and stay accurate to
  ~4e-15; NEON is simply 2-wide, so the ceiling is 2× before the Newton overhead, and the
  branchless spline's ~3× arithmetic cannot be covered by two lanes. (`bruteforce_avx` is
  worse still at 0.68–0.76×, the difference being the symmetry this module keeps.) **Any
  dispatch must gate on vector width, not just on whether softening is present.**
- `bruteforce_sym_vec` handles softening; `bruteforce_avx` raises on it rather than silently mis-handling it.
- Tests would need their own tolerance (~1e-14, not the suite's 1e-12) since the NR rsqrt is
  not bit-exact, plus a **codegen assertion** — if a future edit reintroduces a runtime loop
  bound or a branch, this silently reverts to scalar while the answers stay correct.
- Only ~43% of AVX2 peak, and LLVM never emits `zmm` (Cascade Lake defaults to
  `prefer-vector-width=256`), so there may be another ~2× in 512-bit vectors.
