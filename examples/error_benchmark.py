#!/usr/bin/env python
"""Accuracy of the tree solver vs the opening angle theta, on a Plummer sphere.

Runs the tree solve in parallel at a range of theta and compares against the exact
(brute-force) field, plotting RMS and maximum force and potential error against theta.  Also records
runtime, so the accuracy/cost tradeoff is visible in one place -- the tree walk costs
roughly theta^-3, so the interesting question is what accuracy that buys.

Both RMS and maximum error are reported.  Errors are *relative*: the per-particle error
magnitude |a_tree - a_exact| is normalised by the RMS |a_exact| of the whole system (and the
potential error by std(phi), since phi has an arbitrary zero point).  Normalising per particle
instead would let the max be dominated by the handful of particles sitting near a field null,
where |a_exact| is nearly zero and any absolute error looks enormous.

The max error is typically 1-2 orders of magnitude above the RMS: the treecode error
distribution has a long tail, so RMS alone understates the worst case a particle can see.

    python error_benchmark.py                      # N=1e5, theta 0.1 .. 1.0
    python error_benchmark.py -N 30000 --quadrupole
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-N", type=int, default=10**5, help="number of particles (default 1e5)")
    ap.add_argument("--theta-min", type=float, default=0.1)
    ap.add_argument("--theta-max", type=float, default=1.0)
    ap.add_argument("--n-theta", type=int, default=10)
    ap.add_argument("--softening", type=float, default=0.0)
    ap.add_argument("--quadrupole", action="store_true", help="also run with quadrupole moments")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--out", default="error_benchmark.png")
    ap.add_argument("--data", default="error_benchmark.json")
    args = ap.parse_args()

    import json

    from numba import get_num_threads
    from pytreegrav import Accel, Potential

    pos, m = plummer(args.N, args.seed)
    h = np.repeat(args.softening, args.N)
    thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta)

    # warm the JIT so the reported runtimes are solve time, not compilation
    warm_p, warm_m = plummer(256)
    warm_h = np.zeros(256)
    for meth in ("tree", "bruteforce"):
        Accel(warm_p, warm_m, warm_h, method=meth, parallel=True)
        Potential(warm_p, warm_m, warm_h, method=meth, parallel=True)

    print(f"N = {args.N}, softening = {args.softening}, threads = {get_num_threads()}")
    t0 = time.perf_counter()
    accel_exact = Accel(pos, m, h, method="bruteforce", parallel=True)
    phi_exact = Potential(pos, m, h, method="bruteforce", parallel=True)
    print(f"exact (brute force) reference: {time.perf_counter() - t0:.1f}s")

    # normalisations: RMS field strength, and the spread of phi (its zero point is arbitrary)
    accel_scale = np.sqrt(np.mean(np.sum(accel_exact**2, axis=1)))
    phi_scale = np.std(phi_exact)

    runs = [("monopole", False)] + ([("quadrupole", True)] if args.quadrupole else [])
    res = {}
    for label, quad in runs:
        aerr, aerr_max, perr, perr_max, atime, ptime = [], [], [], [], [], []
        print(f"\n{label}:")
        print(
            f"{'theta':>7s} | {'accel RMS':>11s} {'accel max':>11s} | {'pot RMS':>11s} "
            f"{'pot max':>11s} | {'accel s':>8s} {'pot s':>7s}"
        )
        print("-" * 82)
        for th in thetas:
            t0 = time.perf_counter()
            at = Accel(pos, m, h, method="tree", parallel=True, theta=th, quadrupole=quad)
            ta = time.perf_counter() - t0
            t0 = time.perf_counter()
            pt = Potential(pos, m, h, method="tree", parallel=True, theta=th, quadrupole=quad)
            tp = time.perf_counter() - t0
            da = np.sqrt(np.sum((at - accel_exact) ** 2, axis=1))  # per-particle error magnitude
            dp = np.abs((pt - phi_exact) - np.mean(pt - phi_exact))  # phi zero point is arbitrary
            ea, ea_max = np.sqrt(np.mean(da**2)) / accel_scale, da.max() / accel_scale
            ep, ep_max = np.std(pt - phi_exact) / phi_scale, dp.max() / phi_scale
            aerr.append(ea)
            aerr_max.append(ea_max)
            perr.append(ep)
            perr_max.append(ep_max)
            atime.append(ta)
            ptime.append(tp)
            print(
                f"{th:7.3f} | {ea:11.4e} {ea_max:11.4e} | {ep:11.4e} {ep_max:11.4e} | {ta:8.3f} {tp:7.3f}", flush=True
            )
        res[label] = {
            "theta": thetas.tolist(),
            "accel_err": aerr,
            "accel_err_max": aerr_max,
            "pot_err": perr,
            "pot_err_max": perr_max,
            "accel_time": atime,
            "pot_time": ptime,
        }

    with open(args.data, "w") as fp:
        json.dump(
            {"N": args.N, "softening": args.softening, "nthreads": int(get_num_threads()), "results": res}, fp, indent=1
        )
    plot(res, args.N, args.out)
    print(f"\nwrote {args.out} and {args.data}")


def plot(res, N, out):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    styles = {"monopole": ("-", "o"), "quadrupole": ("--", "s")}
    for label, d in res.items():
        ls, mk = styles.get(label, ("-", "o"))
        sfx = f" ({label})" if len(res) > 1 else ""
        for key, kmax, color, name in (
            ("accel_err", "accel_err_max", "#e0417f", "force"),
            ("pot_err", "pot_err_max", "#6b6fc4", "potential"),
        ):
            axes[0].plot(d["theta"], d[key], ls, marker=mk, ms=4, color=color, label=f"{name} RMS{sfx}")
            axes[0].plot(
                d["theta"], d[kmax], ls, marker=mk, ms=4, mfc="none", color=color, alpha=0.55, label=f"{name} max{sfx}"
            )
        axes[1].plot(d["accel_time"], d["accel_err"], ls, marker=mk, ms=4, color="#e0417f", label=f"force RMS{sfx}")
        axes[1].plot(
            d["accel_time"],
            d["accel_err_max"],
            ls,
            marker=mk,
            ms=4,
            mfc="none",
            color="#e0417f",
            alpha=0.55,
            label=f"force max{sfx}",
        )
    axes[0].set_xlabel(r"opening angle $\theta$")
    axes[0].set_ylabel("relative error (RMS filled, max open)")
    axes[0].set_yscale("log")
    axes[0].axvline(0.7, color="0.5", lw=1.0, ls=":")
    axes[0].text(0.7, axes[0].get_ylim()[1], " default", color="0.4", fontsize=8, va="top")
    axes[1].set_xlabel("acceleration solve time (s)")
    axes[1].set_ylabel("relative force error (RMS filled, max open)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_box_aspect(1)  # square plotting box, independent of the data limits
        ax.grid(alpha=0.15, which="both", lw=0.5)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"pytreegrav tree accuracy vs opening angle - Plummer sphere, N={N:,}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
