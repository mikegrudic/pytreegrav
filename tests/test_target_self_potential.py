"""Targets coincident with sources must contribute no self-potential.

Potential_bruteforce excludes self by construction (j starts at i+1), AccelTarget_bruteforce
skips r2==0 explicitly, and the treewalk skips r==0.  PotentialTarget_bruteforce used to be the
lone exception: at r==0 with h>0 it took the r<h branch and added m*PotentialKernel(0,h).  That
made tree and bruteforce disagree, which surfaced through method='adaptive' as a silent jump in
the answer as soon as n_target*n_source crossed its internal 1e6 threshold.
"""

import numpy as np

from pytreegrav.frontend import Potential, PotentialTarget


def _data(n=400, seed=11):
    rng = np.random.default_rng(seed)
    x = np.asarray(rng.normal(size=(n, 3)), np.float64)
    m = np.asarray(rng.random(n) / n, np.float64)
    h = np.asarray(np.repeat(0.1, n), np.float64)
    return x, m, h


def test_target_bruteforce_matches_tree_on_coincident_points():
    x, m, h = _data()
    xt, ht = x[:16].copy(), h[:16].copy()

    bf = PotentialTarget(xt, x, m, ht, h, method="bruteforce")
    tr = PotentialTarget(xt, x, m, ht, h, method="tree", theta=0.1)

    assert np.allclose(bf, tr, rtol=2e-3)


def test_adaptive_threshold_does_not_change_the_answer():
    """Same particle, same sources -- crossing adaptive's n_target*n_source>1e6 switch must not
    move the answer by more than treecode truncation error."""
    x, m, h = _data(n=4000)

    below = PotentialTarget(x[:200], x, m, h[:200], h, theta=0.1)  # -> bruteforce
    above = PotentialTarget(x[:400], x, m, h[:400], h, theta=0.1)  # -> tree

    assert np.allclose(below[:200], above[:200], rtol=2e-3)


def test_potential_target_reproduces_potential():
    """PotentialTarget(x, x, ...) is the self-potential, so it must agree with Potential(x, ...)
    for both summation methods."""
    x, m, h = _data()
    ref = Potential(x, m, h, method="bruteforce")

    assert np.allclose(PotentialTarget(x, x, m, h, h, method="bruteforce"), ref, rtol=1e-12)
    assert np.allclose(PotentialTarget(x, x, m, h, h, method="tree", theta=0.1), ref, rtol=2e-3)
