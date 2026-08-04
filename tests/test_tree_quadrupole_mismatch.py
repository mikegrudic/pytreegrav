"""Guard against walking a quadrupole-less tree with quadrupole=True.

The walk kernels index tree.Quadrupoles based only on the walk's quadrupole flag, but that
array is allocated only when the tree itself was built with quadrupole=True.  numba does not
bounds-check, so before the guard in checkTreeQuadrupoles these calls read out of bounds and
segfaulted the interpreter rather than raising.
"""

import numpy as np
import pytest

from pytreegrav.frontend import ConstructTree, Potential, PotentialTarget, Accel, AccelTarget


def _data(n=500, seed=7):
    rng = np.random.default_rng(seed)
    x = np.asarray(rng.normal(size=(n, 3)), np.float64)
    m = np.asarray(rng.random(n) / n, np.float64)
    h = np.asarray(np.repeat(0.05, n), np.float64)
    return x, m, h


def test_monopole_tree_with_quadrupole_walk_raises():
    """All four entry points must raise, not segfault, on a quadrupole-less tree."""
    x, m, h = _data()
    xt, ht = x[:50], h[:50]
    tree = ConstructTree(x, m, h)  # quadrupole=False by default
    assert not tree.HasQuads

    with pytest.raises(ValueError, match="quadrupole"):
        Potential(x, m, h, tree=tree, quadrupole=True, method="tree")
    with pytest.raises(ValueError, match="quadrupole"):
        Accel(x, m, h, tree=tree, quadrupole=True, method="tree")
    with pytest.raises(ValueError, match="quadrupole"):
        PotentialTarget(xt, None, None, ht, tree=tree, quadrupole=True, method="tree")
    with pytest.raises(ValueError, match="quadrupole"):
        AccelTarget(xt, None, None, ht, tree=tree, quadrupole=True, method="tree")


def test_quadrupole_tree_with_quadrupole_walk_works():
    """The guard must not fire when the tree really does carry quadrupole moments."""
    x, m, h = _data()
    xt, ht = x[:50], h[:50]
    qtree = ConstructTree(x, m, h, quadrupole=True)
    assert qtree.HasQuads

    kw = dict(quadrupole=True, method="tree")
    assert np.all(np.isfinite(Potential(x, m, h, tree=qtree, **kw)))
    assert np.all(np.isfinite(Accel(x, m, h, tree=qtree, **kw)))
    assert np.all(np.isfinite(PotentialTarget(xt, None, None, ht, tree=qtree, **kw)))
    assert np.all(np.isfinite(AccelTarget(xt, None, None, ht, tree=qtree, **kw)))


def test_monopole_tree_with_monopole_walk_works():
    """The common case -- no quadrupoles anywhere -- must be unaffected by the guard."""
    x, m, h = _data()
    xt, ht = x[:50], h[:50]
    tree = ConstructTree(x, m, h)

    assert np.all(np.isfinite(Potential(x, m, h, tree=tree, method="tree")))
    assert np.all(np.isfinite(Accel(x, m, h, tree=tree, method="tree")))
    assert np.all(np.isfinite(PotentialTarget(xt, None, None, ht, tree=tree, method="tree")))
    assert np.all(np.isfinite(AccelTarget(xt, None, None, ht, tree=tree, method="tree")))


def test_internally_built_tree_honours_quadrupole():
    """When pytreegrav builds the tree itself the flags cannot diverge."""
    x, m, h = _data()
    assert np.all(np.isfinite(Potential(x, m, h, quadrupole=True, method="tree")))
    assert np.all(np.isfinite(Accel(x, m, h, quadrupole=True, method="tree")))
