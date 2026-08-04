#!/usr/bin/env python
"""Plummer-sphere scaling benchmark: time per particle vs N, for Potential/Acceleration x
Tree/BruteForce, producing the Serial and Parallel panels of the JOSS paper figure in one run.

This is a more careful version of the timing half of ``benchmark.py`` (which is the original
paper script, and additionally measures tree-vs-bruteforce accuracy).  The differences that
matter if you are comparing numbers between library versions:

  * the JIT is warmed on every specialization up front, so no timed point includes a compile
    -- otherwise the small-N end of the parallel curves measures numba startup, not the solver;
  * timings are min-of-repeats inside a wall-clock budget, so noisy machines do not inflate
    the result and large-N points still only run once;
  * both parallel and serial are measured in one process, rather than editing a flag and
    running twice;
  * raw timings are written to JSON so the figure can be restyled without re-running a sweep
    that costs ~20 minutes at N=1e7.

Brute force is O(N^2), so cap it well below the tree's range with --nmax-bf.

    python benchmark_scaling.py                                    # quick: tree to 1e6
    python benchmark_scaling.py --nmax-tree 1e7 --nmax-bf 1.5e5    # full paper range
"""

import argparse
import json
import time

import numpy as np


def plummer(n, seed=42):
    """Sample n particles from a Plummer sphere (M=1, a=1), repo/JOSS inverse-CDF convention."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    r = np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))
    d = rng.normal(size=(n, 3))
    pos = (d.T * r / np.sum(d**2, axis=1) ** 0.5).T
    return np.ascontiguousarray(np.float64(pos)), np.repeat(1.0 / n, n)


def bench(f, *args, budget=2.0, max_reps=50, **kwargs):
    """Best-of-repeats wall time within `budget` seconds.  Assumes f is already compiled."""
    t0 = time.perf_counter()
    f(*args, **kwargs)
    best = time.perf_counter() - t0
    n = 1
    while n < max_reps and (time.perf_counter() - t0) < budget:
        s = time.perf_counter()
        f(*args, **kwargs)
        best = min(best, time.perf_counter() - s)
        n += 1
    return best


def backends(cuda, nmax_bf, nmax_bf_cuda):
    """(tag, solver kwargs, brute-force cap) per measured backend, CUDA last and only if available."""
    out = [
        ("ser", {"parallel": False}, nmax_bf),
        ("par", {"parallel": True}, nmax_bf),
    ]
    if cuda:
        # device='cuda' is one-shot: each rep repacks and re-uploads the tree, exactly as the CPU
        # columns rebuild nothing -- so this is the honest single-call comparison, not the
        # tree-resident one you get from holding a pytreegrav.cuda.CudaPotential.
        out.append(("cuda", {"parallel": True, "device": "cuda"}, nmax_bf_cuda))
    return out


def run(nmin, nmax_tree, nmax_bf, nmax_bf_cuda, per_decade, theta, budget, cuda):
    from numba import get_num_threads
    from pytreegrav import Accel, Potential

    nthreads = get_num_threads()
    decades = np.log10(nmax_tree / nmin)
    npoints = int(round(decades * per_decade)) + 1
    Ns = np.unique(np.int64(np.round(np.logspace(np.log10(nmin), np.log10(nmax_tree), npoints))))
    bks = backends(cuda, nmax_bf, nmax_bf_cuda)

    # Warm every specialization once, cheaply, so no timed point pays a compile.  The CUDA kernels go
    # through NVRTC, which is seconds -- much worse to leave in a timed point than numba's own JIT.
    pos, m = plummer(256)
    h = np.zeros(256)
    for _, kw, _ in bks:
        for method in ("tree", "bruteforce"):
            Potential(pos, m, h, method=method, theta=theta, **kw)
            Accel(pos, m, h, method=method, theta=theta, **kw)

    res = {f"{q}_{method}_{tag}": {} for q in ("pot", "acc") for method in ("tree", "bruteforce") for tag, _, _ in bks}

    print(f"threads = {nthreads}, theta = {theta}" + (", cuda" if cuda else ""))
    header = [f"{tag} {q} {mn}" for tag, _, _ in bks for mn in ("tree", "bf") for q in ("pot", "acc")]
    print(f"{'N':>9s} | " + " ".join(f"{c:>12s}" for c in header) + "   (us/particle)")
    print("-" * (12 + 13 * len(header)))
    for N in Ns:
        N = int(N)
        pos, m = plummer(N)
        h = np.zeros(N)
        row = []
        for tag, kw, bf_cap in bks:
            for method in ("tree", "bruteforce"):
                if method == "bruteforce" and N > bf_cap:
                    row += ["--", "--"]
                    continue
                for q, fn in (("pot", Potential), ("acc", Accel)):
                    t = bench(fn, pos, m, h, budget=budget, method=method, theta=theta, **kw)
                    res[f"{q}_{method}_{tag}"][N] = t / N
                    row.append(f"{t / N * 1e6:.4f}")
        print(f"{N:9d} | " + " ".join(f"{v:>12s}" for v in row), flush=True)
    return res, nthreads


def plot(res, nthreads, theta, out, panels="both"):
    """panels: 'both' (Serial|Parallel), 'serial', or 'parallel'."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    series = [
        ("pot_tree", "Potential (Tree)", "#3f9e6d"),
        ("acc_tree", "Acceleration (Tree)", "#d1701c"),
        ("pot_bruteforce", "Potential (Brute Force)", "#6b6fc4"),
        ("acc_bruteforce", "Acceleration (Brute Force)", "#e0417f"),
    ]
    wanted = {
        "both": [("ser", "Serial"), ("par", "Parallel")],
        "serial": [("ser", "Serial")],
        "parallel": [("par", "Parallel")],
        "all": [("ser", "Serial"), ("par", "Parallel"), ("cuda", "CUDA")],
        "cuda": [("cuda", "CUDA")],
    }[panels]
    wanted = [w for w in wanted if any(res.get(f"{k}_{w[0]}") for k, _, _ in series)]
    fig, axes = plt.subplots(1, len(wanted), figsize=(5.9 * len(wanted), 5.4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (tag, title) in zip(axes, wanted):
        for key, label, color in series:
            d = res[f"{key}_{tag}"]
            if not d:
                continue
            ns = sorted(d)
            ax.plot(ns, [d[n] for n in ns], "-", color=color, lw=1.8, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of particles")
        ax.text(0.04, 0.93, title, transform=ax.transAxes, fontsize=17, va="top")
        ax.grid(alpha=0.15, which="both", lw=0.5)
        ax.set_box_aspect(1)
    axes[0].set_ylabel("Time per particle (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    # only mention the thread count when a CPU-parallel panel is actually shown
    tags = {t for t, _ in wanted}
    thread_note = f", {nthreads} threads" if "par" in tags else ("" if "cuda" in tags else ", single core")
    fig.suptitle(f"pytreegrav scaling - Plummer, theta={theta}{thread_note}", fontsize=10, y=0.98)
    fig.tight_layout()
    # One legend under the panels rather than per-axes, added after tight_layout so it does not collide
    # with the x labels: in the CUDA panel the tree curves flatten into exactly the corner an inset
    # legend wants, and no single corner is free in all panels.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmin", type=float, default=64)
    ap.add_argument("--nmax-tree", type=float, default=1e6)
    ap.add_argument("--nmax-bf", type=float, default=1e5, help="brute force is O(N^2); cap it")
    ap.add_argument(
        "--nmax-bf-cuda",
        type=float,
        default=1e6,
        help="separate brute-force cap for CUDA, which sustains ~40x the CPU pair rate",
    )
    ap.add_argument("--per-decade", type=int, default=4)
    ap.add_argument("--theta", type=float, default=0.7, help="library default")
    ap.add_argument("--budget", type=float, default=2.0, help="seconds per timing point")
    ap.add_argument("-o", "--out", default="benchmark_scaling.png")
    ap.add_argument("--data", default="benchmark_scaling.json", help="raw timings, for re-plotting")
    ap.add_argument("--panels", choices=("both", "serial", "parallel", "all", "cuda"), default="both")
    ap.add_argument("--cuda", action="store_true", help="also measure device='cuda' (needs pytreegrav[cuda])")
    ap.add_argument("--replot-from", metavar="JSON", help="skip the sweep; re-render from a saved --data file")
    args = ap.parse_args()

    if args.cuda:
        from pytreegrav.cuda import is_available

        if not is_available():
            ap.error("--cuda given but numba-cuda is not installed or no device is visible")
        if args.panels == "both":
            args.panels = "all"

    if args.replot_from:
        with open(args.replot_from) as fp:
            d = json.load(fp)
        res = {k: {int(n): v for n, v in sub.items()} for k, sub in d["res"].items()}
        plot(res, d["nthreads"], d["theta"], args.out, args.panels)
        print(f"wrote {args.out} (replotted from {args.replot_from})")
        return

    res, nthreads = run(
        args.nmin, args.nmax_tree, args.nmax_bf, args.nmax_bf_cuda, args.per_decade, args.theta, args.budget, args.cuda
    )

    with open(args.data, "w") as fp:
        json.dump(
            {
                "nthreads": int(nthreads),
                "theta": args.theta,
                "res": {k: {str(n): v for n, v in d.items()} for k, d in res.items()},
            },
            fp,
            indent=1,
        )
    plot(res, nthreads, args.theta, args.out, args.panels)
    print(f"\nwrote {args.out} and {args.data}")


if __name__ == "__main__":
    main()
