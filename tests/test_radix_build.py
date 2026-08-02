"""Tests for the radix-sort octree build (drop-in replacement for the insertion build)."""

import numpy as np
from pytreegrav.frontend import ConstructTree, AccelTarget, PotentialTarget


def _accel(x, m, h, tree, theta=0.7):
    return AccelTarget(x, None, None, softening_target=h, method="tree", theta=theta, tree=tree)


def _phi(x, m, h, tree, theta=0.7):
    return PotentialTarget(x, None, None, softening_target=h, method="tree", theta=theta, tree=tree)


def test_radix_matches_insertion():
    """Radix and insertion builds give matching accel/potential on generic (random) data.

    Both algorithms subdivide the same dyadic cell iff it contains >=2 points, so for a generic
    point set they produce isomorphic trees and hence bit-identical fields.  This near-exact
    agreement is NOT guaranteed in general and relies on the RANDOM coordinates used here:
      - Points sitting exactly on a dyadic cell boundary (pos == a cell mid-plane) are assigned to
        opposite octants by the two builds (insertion uses strict '>', the radix build's integer
        quantization rounds onto the upper side).  Grid/lattice-aligned data can trigger this and
        would legitimately give differing-but-both-valid trees (agreeing only to ~theta accuracy).
      - Exactly-coincident points are separated differently (insertion perturbs randomly; the radix
        build buckets them deterministically).
    So do not tighten this into a universal "the two builds are identical" assertion.
    """
    np.random.seed(3)
    N = 5000
    x = np.random.rand(N, 3)
    m = np.ones(N) / N
    h = np.repeat(0.01, N)

    tr = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=True)
    ti = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=False)

    ar, ai = _accel(x, m, h, tr), _accel(x, m, h, ti)
    pr, pi = _phi(x, m, h, tr), _phi(x, m, h, ti)

    a_rel = np.sqrt(np.mean(np.sum((ar - ai) ** 2, 1))) / np.sqrt(np.mean(np.sum(ar**2, 1)))
    p_rel = np.std(pr - pi) / np.std(pr)
    # isomorphic trees for this random data -> agreement is essentially exact (see docstring caveats)
    assert a_rel < 1e-10
    assert p_rel < 1e-10


def test_radix_clustered_growth():
    """A clustered distribution forces node-array growth and deep nesting; must stay consistent."""
    np.random.seed(5)
    N = 5000
    x = np.vstack([np.random.normal(size=(N, 3)) * 1e-3, np.random.rand(N, 3)])
    m = np.ones(len(x)) / len(x)
    h = np.repeat(1e-4, len(x))

    tr = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=True)
    ti = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=False)
    ar, ai = _accel(x, m, h, tr, theta=0.5), _accel(x, m, h, ti, theta=0.5)
    a_rel = np.sqrt(np.mean(np.sum((ar - ai) ** 2, 1))) / np.sqrt(np.mean(np.sum(ar**2, 1)))
    assert a_rel < 1e-10
    assert np.isclose(tr.Masses[tr.NumParticles], m.sum())


def test_radix_coincident_points():
    """Exactly-coincident duplicates must not be dropped (mass/particle count conserved)."""
    np.random.seed(11)
    x = np.random.rand(1000, 3)
    x = np.vstack([x, x[:50]])  # 50 exact duplicates
    m = np.ones(len(x)) / len(x)
    h = np.repeat(0.02, len(x))

    tr = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=True)
    assert np.isclose(tr.Masses[tr.NumParticles], m.sum())
    assert np.array_equal(np.sort(tr.TreewalkIndices), np.arange(len(x)))


def test_radix_rekeying_separation():
    """Points sharing a 63-bit Morton key but distinct in float must be fully separated by re-keying."""
    np.random.seed(7)
    corners = np.array([[0.0, 0, 0], [1, 1, 1]])  # force root cube of side 1
    cluster = 0.5 + 1e-8 * np.random.rand(300, 3)  # 1e-8 << 2^-21 ~ 4.8e-7, so they collide initially
    x = np.vstack([corners, cluster])
    m = np.ones(len(x)) / len(x)
    h = np.repeat(1e-3, len(x))

    tr = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=True)
    ti = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=False)
    assert np.isclose(tr.Sizes[tr.NumParticles], 1.0)  # unit root cube
    assert np.isclose(tr.Masses[tr.NumParticles], m.sum())
    assert np.array_equal(np.sort(tr.TreewalkIndices), np.arange(len(x)))
    ar, ai = _accel(x, m, h, tr, theta=0.5), _accel(x, m, h, ti, theta=0.5)
    a_rel = np.sqrt(np.mean(np.sum((ar - ai) ** 2, 1))) / np.sqrt(np.mean(np.sum(ar**2, 1)))
    assert a_rel < 1e-10


def test_radix_tiny_N():
    """Trees with very few particles must build without error and conserve mass."""
    np.random.seed(13)
    for N in (1, 2, 3):
        x = np.random.rand(N, 3)
        m = np.ones(N) / N
        h = np.repeat(0.1, N)
        t = ConstructTree(np.float64(x), np.float64(m), np.float64(h), radix=True)
        assert np.isclose(t.Masses[t.NumParticles], m.sum())
        assert np.array_equal(np.sort(t.TreewalkIndices), np.arange(N))


def _walk_particles(tree):
    """Every particle reached by fully opening the tree, via FirstSubnode/NextBranch only.

    Mirrors how the treewalk traverses: descend into FirstSubnode, and on reaching a leaf take
    NextBranch. Returns the particle indices in visit order.
    """
    seen = []
    no = tree.NumParticles  # root
    # hard cap: a mislinked chain can cycle forever, and a hung test is useless in CI
    for _ in range(4 * tree.NumNodes + 16):
        if no <= -1:
            return np.array(seen)
        if no < tree.NumParticles:  # a particle
            seen.append(no)
            no = tree.NextBranch[no]
        else:
            no = tree.FirstSubnode[no]
    raise AssertionError("treewalk did not terminate: FirstSubnode/NextBranch chain has a cycle")


def test_treewalk_links_reach_every_particle_once():
    """The radix build now writes FirstSubnode/NextBranch inline instead of deriving them in a
    separate SetupTreewalk pass over a `children` table. A mislinked sibling chain would silently
    drop or revisit particles while still producing plausible-looking forces, so check the
    traversal directly: fully opening the tree must visit each particle exactly once.
    """
    for n, gen in ((1, "single"), (2, "pair"), (997, "random"), (5000, "random")):
        rng = np.random.default_rng(n)
        x = rng.random((n, 3))
        m = np.repeat(1.0 / n, n)
        h = np.zeros(n)
        tree = ConstructTree(np.float64(x), np.float64(m), np.float64(h))
        got = _walk_particles(tree)
        assert np.array_equal(np.sort(got), np.arange(n)), f"n={n} ({gen}): traversal is not a permutation"


def test_treewalk_links_reach_every_particle_with_duplicates():
    """Coincident particles go down BuildBucket's chained-node path, which does its own linking."""
    import warnings

    rng = np.random.default_rng(11)
    base = rng.random((800, 3))
    x = np.vstack([base, base[:200]])  # 200 exact duplicates
    n = len(x)
    m = np.repeat(1.0 / n, n)
    h = np.repeat(0.01, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ConstructTree(np.float64(x), np.float64(m), np.float64(h))
    assert np.array_equal(np.sort(_walk_particles(tree)), np.arange(n))


def test_node_masses_equal_sum_of_children():
    """Each internal node's mass must equal the sum over its child chain.

    ComputeMoments now iterates children by following NextBranch from FirstSubnode until it
    reaches NextBranch[node], rather than reading a child table. If that chain terminates early
    the node's moments are computed from a subset of its children -- mass is the sharpest probe.
    """
    n = 4000
    rng = np.random.default_rng(5)
    x = rng.random((n, 3))
    m = rng.random(n) / n
    h = np.zeros(n)
    tree = ConstructTree(np.float64(x), np.float64(m), np.float64(h))
    for no in range(tree.NumParticles, tree.NumNodes):
        if tree.FirstSubnode[no] < 0:
            continue  # not an allocated internal node
        stop = tree.NextBranch[no]
        c = tree.FirstSubnode[no]
        total = 0.0
        nkids = 0
        while c != stop and nkids <= 8:
            total += tree.Masses[c]
            c = tree.NextBranch[c]
            nkids += 1
        assert nkids <= 8, f"node {no} has a runaway child chain"
        assert np.isclose(total, tree.Masses[no], rtol=1e-12), f"node {no}: children sum {total} != {tree.Masses[no]}"
