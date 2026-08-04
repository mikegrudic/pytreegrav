"""Tests for the ``tree=`` contract on Potential/Accel: target ordering must not matter.

The frontend used to permute ``pos`` by ``tree.TreewalkIndices``, a fixed sigma over whatever built
the tree. A larger supplied tree then raised IndexError, and since sigma is not an involution, a caller
who had already tree-ordered ``pos`` got X[sigma^2] -- the right answer, but with grouping's acceptance
padding inflated ~94x in a STARFORGE snapshot's densest regions, 15 s -> over 285 s at N=2.5e7. Sorting
is idempotent and indexes ``pos`` by construction, so it covers both.
"""

import numpy as np
import pytest

from pytreegrav import Accel, ConstructTree, Potential
from pytreegrav.grouped_treewalk import _morton_order


def cloud(n, seed=42):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, 3)))
    return x, np.ones(n) / n, np.full(n, 2.0 * n ** (-1 / 3))


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
