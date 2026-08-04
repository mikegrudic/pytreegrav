"""Tests for the ``tree=`` contract on Potential/Accel: target ordering must not matter.

The frontend used to permute ``pos`` by ``tree.TreewalkIndices``, a fixed sigma over whatever built
the tree. A larger supplied tree then raised IndexError, and since sigma is not an involution, a caller
who had already tree-ordered ``pos`` got X[sigma^2] -- the right answer, but with grouping's acceptance
padding inflated ~94x in a STARFORGE snapshot's densest regions, 15 s -> over 285 s at N=2.5e7. Sorting
is idempotent and indexes ``pos`` by construction, so it covers both.
"""

import numpy as np
import pytest

from pytreegrav import (
    Accel,
    ConstructTree,
    DensityCorrFunc,
    Potential,
    VelocityCorrFunc,
    VelocityStructFunc,
)
from pytreegrav.grouped_treewalk import _morton_order


def cloud(n, seed=42):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, 3)))
    return x, np.ones(n) / n, np.full(n, 2.0 * n ** (-1 / 3))


def cloud_with_velocity(n, seed=42):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, 3)))
    v = np.ascontiguousarray(rng.normal(size=(n, 3)))
    return x, v, np.ones(n) / n


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def test_morton_order_is_idempotent():
    """The property the fix relies on: sorting already-sorted data is the identity."""
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    xs = np.ascontiguousarray(np.take(x, tree.TreewalkIndices, axis=0))
    assert np.array_equal(_morton_order(np.float64(xs)), np.arange(len(xs)))
    # and the reason applying the stored permutation twice is not a no-op
    sig = tree.TreewalkIndices
    assert not np.array_equal(sig[sig], np.arange(len(sig)))


@pytest.mark.parametrize("fn", [Potential, Accel])
def test_supplied_tree_of_the_same_particles(fn):
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    assert rel(fn(x, m, h, tree=tree, parallel=True), fn(x, m, h, parallel=True)) < 1e-14


@pytest.mark.parametrize("fn", [Potential, Accel])
def test_pre_sorted_targets_give_the_same_answer(fn):
    """Pre-ordering targets by the tree's walk order is a natural thing to try for cache reasons; it
    must not change the result (it used to be correct but ~20x slower)."""
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    idx = tree.TreewalkIndices
    xs = np.ascontiguousarray(np.take(x, idx, axis=0))
    hs = np.ascontiguousarray(h[idx])
    ref = fn(x, m, h, tree=tree, parallel=True)
    got = fn(xs, m, hs, tree=tree, parallel=True)
    assert rel(got, np.take(ref, idx, axis=0)) < 1e-14


@pytest.mark.parametrize("fn", [Potential, Accel])
def test_supplied_tree_larger_than_the_target_array(fn):
    """The case that raised IndexError: fewer targets than the tree has particles."""
    x, m, h = cloud(5000)
    tree = ConstructTree(x, m, h)
    sub = np.ascontiguousarray(x[:200])
    got = fn(sub, m[:200], np.ascontiguousarray(h[:200]), tree=tree, parallel=True)
    assert len(got) == 200
    assert np.all(np.isfinite(got))
    # Compare with grouping off. With it on, the acceptance radius is padded by each group's extent, so
    # which nodes get opened depends on the *target set* -- 200 targets group differently from 5000, and
    # the two answers legitimately differ at the theta-truncation level rather than at 1e-14.
    sub_ref = fn(sub, m[:200], np.ascontiguousarray(h[:200]), tree=tree, parallel=True, group_size=1)
    full_ref = fn(x, m, h, tree=tree, parallel=True, group_size=1)
    assert rel(sub_ref, full_ref[:200]) < 1e-14


@pytest.mark.parametrize("fn", [Potential, Accel])
def test_target_order_independence(fn):
    """Any permutation of the targets must give the correspondingly permuted answer."""
    x, m, h = cloud(8000)
    tree = ConstructTree(x, m, h)
    perm = np.random.default_rng(0).permutation(len(x))
    ref = fn(x, m, h, tree=tree, parallel=True)
    got = fn(np.ascontiguousarray(x[perm]), m, np.ascontiguousarray(h[perm]), tree=tree, parallel=True)
    assert rel(got, np.take(ref, perm, axis=0)) < 1e-14


# --------------------------------------------------------------------------------------------------
# The correlation functions applied tree.TreewalkIndices to pos unconditionally too, but the defect
# there was narrower: they return binned aggregates, not per-particle arrays, so X[sigma^2] is still a
# valid sample and only the IndexError on a larger supplied tree was a real failure.
#
# They are NOT order-invariant, by design: a node is binned wholesale once it is small relative to the
# bin, and target grouping decides which nodes reach that test.  Below, order-dependence is 4e-3 at
# the default max_bin_size_ratio=100 and falls to 3e-5 at 0.1 -- which is what identifies it as that
# approximation rather than a permutation bug, and is why no tight tolerance belongs here.
# --------------------------------------------------------------------------------------------------


def corrfunc_cases(n):
    """(cloud, [(callable, kwargs)]) for each correlation function, on one shared cloud."""
    x, v, m = cloud_with_velocity(n)
    rbins = np.geomspace(0.05, 4.0, 8)
    return x, v, m, [
        (DensityCorrFunc, {"rbins": rbins}),
        (VelocityCorrFunc, {"rbins": rbins, "v": v}),
        (VelocityStructFunc, {"rbins": rbins, "v": v}),
    ]


@pytest.mark.parametrize("i", range(3))
def test_corrfunc_supplied_tree_larger_than_pos(i):
    """The IndexError case: a tree built on more particles than are passed as pos."""
    x, v, m, cases = corrfunc_cases(3000)
    fn, kw = cases[i]
    tree = ConstructTree(x, m, np.zeros_like(m), vel=v)
    sub = slice(None, 400)
    kw = dict(kw)
    if "v" in kw:
        kw["v"] = np.ascontiguousarray(v[sub])
    bins, vals = fn(np.ascontiguousarray(x[sub]), np.ascontiguousarray(m[sub]), tree=tree, **kw)
    assert len(vals) == len(bins) - 1
    assert np.all(np.isfinite(vals))


@pytest.mark.parametrize("i", range(3))
def test_corrfunc_order_dependence_is_the_binning_approximation(i):
    """Pre-sorted input gives a different answer, and it must converge as the binning tightens.

    A permutation bug would not care about max_bin_size_ratio; the node-acceptance approximation does.
    """
    x, v, m, cases = corrfunc_cases(3000)
    fn, kw = cases[i]
    tree = ConstructTree(x, m, np.zeros_like(m), vel=v)
    idx = tree.TreewalkIndices
    kw_s = dict(kw)
    if "v" in kw_s:
        kw_s["v"] = np.ascontiguousarray(v[idx])
    xs, ms = np.ascontiguousarray(x[idx]), np.ascontiguousarray(m[idx])

    def spread(ratio):
        _, a = fn(x, m, tree=tree, max_bin_size_ratio=ratio, **kw)
        _, b = fn(xs, ms, tree=tree, max_bin_size_ratio=ratio, **kw_s)
        return np.abs(a - b).max() / max(np.abs(a).max(), 1e-300)

    # No absolute bound: VelocityCorrFunc normalizes by a correlation that is ~0 for a random cloud,
    # so its relative spread is meaningless in isolation.  Convergence is what discriminates.
    loose, tight = spread(100), spread(0.1)
    assert tight < loose / 10, f"did not converge with the binning: {loose:.1e} -> {tight:.1e}"
