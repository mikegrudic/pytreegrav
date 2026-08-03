"""Tests for the ``no=`` start-node argument shared by every treewalk kernel.

``no`` is documented as restricting the sum to that subtree, but it did not: ``NextBranch`` leads
*out* of the subtree and carries on in depth-first order, so starting at the root's first child
returned the root's own answer (ratio exactly 1.0000) and summing a node's children over-counted by
roughly the number of children.  The fix halts at ``subtree_end``.

Tested via the invariant the argument exists for: every one of these quantities is a plain sum over
source elements, so splitting the tree into the root's children and summing the parts must reproduce
the whole.
"""

import numpy as np
import pytest

from pytreegrav import ConstructTree
from pytreegrav.treewalk import (
    AccelWalk,
    AccelWalk_quad,
    ColumnDensityWalk_binned,
    ColumnDensityWalk_multiray,
    ColumnDensityWalk_singleray,
    PotentialWalk,
    PotentialWalk_quad,
)

RTOL = 1e-12
SIX_RAYS = np.vstack([np.eye(3), -np.eye(3)])


def cloud(N=2000, seed=42, quadrupole=False):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(N, 3)))
    m = rng.uniform(0.5, 1.5, N) / N
    h = np.full(N, 2.0 * N ** (-1 / 3))
    return x, m, h, ConstructTree(x, m, h, quadrupole=quadrupole)


def root_children(tree):
    """Indices of the root's direct children, in depth-first order.

    Children may be particles as well as nodes: a leaf child is a valid start point, and the walk
    should then contribute exactly that one particle.
    """
    kids = []
    no = tree.FirstSubnode[tree.NumParticles]
    while no > -1:
        kids.append(no)
        no = tree.NextBranch[no]
    return kids


def test_root_has_several_children():
    """Guard against the whole suite going vacuous if the tree shape changes."""
    _, _, _, tree = cloud()
    kids = root_children(tree)
    assert 2 <= len(kids) <= 8, f"root has {len(kids)} children"


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.abs(b).max()
    return np.abs(a - b).max() / (scale if scale > 0 else 1.0)


# --------------------------------------------------------------------------------------------------
# Column density
# --------------------------------------------------------------------------------------------------


DIAG = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def test_singleray_subtrees_sum_to_whole():
    x, _, _, tree = cloud()
    ray = np.array([0.3, -0.5, 0.81])
    ray /= np.linalg.norm(ray)
    # in-cloud targets, plus one outside shooting along the body diagonal so the ray traverses
    # several octants rather than terminating inside the first one it enters
    cases = [(x[0], ray), (x[501], ray), (x[1999], ray), (np.full(3, -3.0), DIAG)]
    for pos, r in cases:
        whole = ColumnDensityWalk_singleray(pos, r, tree)
        parts = sum(ColumnDensityWalk_singleray(pos, r, tree, k) for k in root_children(tree))
        assert whole > 0, "vacuous unless the ray actually hits something"
        assert rel(parts, whole) < RTOL


def test_no_single_subtree_reproduces_the_whole():
    """Directly targets the pre-fix failure mode: starting at a subnode returned the *whole* tree's
    answer, because NextBranch led back out of the subtree and the walk carried on.

    Note that ``part <= whole`` would not catch this -- the buggy walk satisfied it too. The test has
    to require that no single subtree accounts for everything, with several genuinely contributing.
    """
    _, _, _, tree = cloud()
    # shoot from outside along the body diagonal, so the ray crosses several octants -- an in-cloud
    # target with an axis-aligned ray can legitimately hit only one, which would make this vacuous
    pos = np.full(3, -3.0)
    whole = ColumnDensityWalk_singleray(pos, DIAG, tree)
    parts = [ColumnDensityWalk_singleray(pos, DIAG, tree, k) for k in root_children(tree)]
    assert sum(p > 0 for p in parts) >= 2, "vacuous unless several subtrees contribute"
    assert max(parts) < whole * (1 - 1e-9), f"a subtree returned the whole answer: {max(parts)} vs {whole}"


def test_multiray_subtrees_sum_to_whole():
    x, _, _, tree = cloud()
    whole = ColumnDensityWalk_multiray(x[7], SIX_RAYS, tree)
    parts = sum(ColumnDensityWalk_multiray(x[7], SIX_RAYS, tree, k) for k in root_children(tree))
    assert np.all(whole > 0)
    assert rel(parts, whole) < RTOL


def test_binned_subtrees_sum_to_whole():
    x, _, _, tree = cloud()
    whole = ColumnDensityWalk_binned(x[13], tree)
    parts = sum(ColumnDensityWalk_binned(x[13], tree, 0.5, k) for k in root_children(tree))
    assert np.all(whole > 0)
    assert rel(parts, whole) < RTOL


# --------------------------------------------------------------------------------------------------
# Gravity.  The multipole acceptance criterion depends only on the node under test, so the set of
# accepted elements is the same however the tree is partitioned -- the split is exact, not approximate.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("walk", [PotentialWalk, AccelWalk])
def test_gravity_subtrees_sum_to_whole(walk):
    x, _, h, tree = cloud()
    for i in (0, 1234):
        whole = walk(x[i], tree, h[i], -1, 0.7)
        parts = sum(walk(x[i], tree, h[i], k, 0.7) for k in root_children(tree))
        assert np.any(np.abs(whole) > 0)
        assert rel(parts, whole) < RTOL


@pytest.mark.parametrize("walk", [PotentialWalk_quad, AccelWalk_quad])
def test_gravity_quad_subtrees_sum_to_whole(walk):
    x, _, h, tree = cloud(quadrupole=True)
    for i in (0, 1234):
        whole = walk(x[i], tree, h[i], -1, 0.7)
        parts = sum(walk(x[i], tree, h[i], k, 0.7) for k in root_children(tree))
        assert np.any(np.abs(whole) > 0)
        assert rel(parts, whole) < RTOL


# --------------------------------------------------------------------------------------------------
# The default path must be untouched: NextBranch[root] is -1, so the bound is a no-op there.
# --------------------------------------------------------------------------------------------------


def test_explicit_root_matches_default():
    x, _, h, tree = cloud()
    root = tree.NumParticles
    ray = np.array([0.0, 1.0, 0.0])
    assert ColumnDensityWalk_singleray(x[3], ray, tree, root) == ColumnDensityWalk_singleray(x[3], ray, tree)
    assert PotentialWalk(x[3], tree, h[3], root, 0.7) == PotentialWalk(x[3], tree, h[3], -1, 0.7)
    assert np.array_equal(
        ColumnDensityWalk_multiray(x[3], SIX_RAYS, tree, root),
        ColumnDensityWalk_multiray(x[3], SIX_RAYS, tree),
    )
