"""Tests for the radix-sort octree build (drop-in replacement for the insertion build)."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pytreegrav.frontend import Accel, ConstructTree, AccelTarget, Potential, PotentialTarget


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

    tr = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=True)
    ti = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=False)

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

    tr = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=True)
    ti = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=False)
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

    tr = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=True)
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

    tr = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=True)
    ti = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=False)
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
        t = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64), radix=True)
        assert np.isclose(t.Masses[t.NumParticles], m.sum())
        assert np.array_equal(np.sort(t.TreewalkIndices), np.arange(N))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("method", ["tree", "bruteforce"])
def test_single_particle_through_the_frontend(n, method):
    """N=1 must work end to end, and be the analytic answer for a lone softened particle.

    This was broken on numpy < 2.4 (github issue #31): the frontend cast inputs with ``np.float64(a)``,
    whose *scalar constructor* path triggers for any size-1 array, so a one-particle ``m``/``softening``
    reached the jitclass as a scalar and the build died in numba type inference. Passing plain lists
    exercises the coercion too, since a list has no dtype at all.
    """
    rng = np.random.default_rng(n)
    x = rng.random((n, 3))
    m = np.repeat(1.0 / n, n)
    h = np.repeat(0.1, n)
    phi = Potential(x, m, h, method=method)
    acc = Accel(x, m, h, method=method)
    assert phi.shape == (n,) and acc.shape == (n, 3)
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(acc))
    if n == 1:
        # a lone particle feels no force, and its self-potential is excluded
        assert phi[0] == 0.0
        assert np.array_equal(acc[0], np.zeros(3))
    # and again from lists, which have no dtype for np.float64 to preserve
    assert np.allclose(Potential(x.tolist(), m.tolist(), h.tolist(), method=method), phi)


def test_single_particle_tree_is_usable():
    """ConstructTree must also accept a single particle, including already-collapsed scalar inputs."""
    tree = ConstructTree(np.array([[0.2, 0.3, 0.4]]), np.array([2.0]), np.array([0.1]))
    assert tree.NumParticles == 1
    assert np.isclose(tree.Masses[tree.NumParticles], 2.0)
    # what np.float64() hands you on numpy < 2.4 for a length-1 array: bare scalars
    collapsed = ConstructTree(np.array([[0.2, 0.3, 0.4]]), np.float64(2.0), np.float64(0.1))
    assert collapsed.NumParticles == 1
    assert np.isclose(collapsed.Masses[collapsed.NumParticles], 2.0)


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
        tree = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64))
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
        tree = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64))
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
    tree = ConstructTree(np.asarray(x, np.float64), np.asarray(m, np.float64), np.asarray(h, np.float64))
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


def _build_and_dump(nmin, n, seed, dup):
    """Build in a subprocess with PARALLEL_SPLIT_NMIN forced, and dump the tree's fingerprint.

    The constant has to be set before the first ConstructTree call: numba freezes module globals
    at compile time, so assigning it afterwards is silently ignored (a mistake that made an
    earlier serial-vs-parallel benchmark report a spurious 1.00x, because both runs were serial).
    Hence a fresh interpreter per configuration.
    """
    src = textwrap.dedent(f"""
        import warnings, numpy as np
        warnings.simplefilter("ignore")
        import pytreegrav.octree as om
        om.PARALLEL_SPLIT_NMIN = {nmin}      # baked in at first compile, below
        from pytreegrav import ConstructTree
        rng = np.random.default_rng({seed})
        x = rng.random(({n}, 3))
        if {dup}:
            x[:{dup}] = x[0]                 # force the deferred re-keying path
        m = rng.random({n}) / {n}
        h = rng.random({n}) * 0.01
        t = ConstructTree(x, m, h)
        nn = t.NumParticles
        First, Next = np.asarray(t.FirstSubnode), np.asarray(t.NextBranch)
        M, C, D, S = (np.asarray(a) for a in (t.Masses, t.Coordinates, t.Deltas, t.Softenings))
        # Walk the links and fingerprint what the traversal sees. Node *indices* legitimately
        # differ between the two paths -- serial numbers nodes in DFS order, parallel gives each
        # frontier subtree its own block -- so anything index-keyed would compare unequal even
        # when the trees are identical.
        order, nodes, st = [], [], [nn]
        # Bounded: a mis-budgeted parallel run produces overlapping node blocks and hence a
        # cyclic sibling chain, which an unbounded walk would spin on forever instead of
        # reporting. Ask me how I know.
        budget = 8 * t.NumNodes + 64
        while st:
            budget -= 1
            if budget < 0:
                raise AssertionError("link chain does not terminate: node blocks overlap")
            no = st.pop()
            if no < nn:
                order.append(no)
                continue
            nodes.append((M[no], C[no, 0], C[no, 1], C[no, 2], D[no], S[no]))
            kids, c, stop = [], First[no], Next[no]
            while c != -1 and c != stop:
                kids.append(c)
                c = Next[c]
                if len(kids) > 8:
                    raise AssertionError("more than 8 children: sibling chain is corrupt")
            st.extend(reversed(kids))
        print(int(om.PARALLEL_SPLIT_NMIN <= {n}))
        print(",".join(str(v) for v in order))
        print(",".join(repr(float(v)) for row in nodes for v in row))
        print(f"{{len(nodes)}} {{repr(float(M[nn]))}}")
    """)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"build failed (rc={r.returncode}): {r.stderr[-2000:]}"
    used_parallel, order, nodes, summary = r.stdout.strip().split("\n")
    return used_parallel == "1", order, nodes, summary


@pytest.mark.parametrize("dup", [0, 40])
def test_parallel_split_matches_serial(dup):
    """The parallel split must produce exactly the tree the serial split produces.

    Each worker builds one frontier subtree into a private, exactly-sized block of node indices,
    so the node numbering and hence the whole structure should be reproduced bit-for-bit -- there
    is no reordering or floating-point reassociation anywhere in the split.

    dup=40 forces the deferred path: groups whose Morton key is exhausted cannot be budgeted for
    (re-keying permutes particle data and its node count is not predictable), so workers hand
    them back and the serial loop finishes them afterwards.
    """
    n = 60_000  # above PARALLEL_SPLIT_NMIN, so the parallel path is taken
    par, order_p, nodes_p, summary_p = _build_and_dump(1, n, 5, dup)
    ser, order_s, nodes_s, summary_s = _build_and_dump(10**12, n, 5, dup)
    assert par and not ser, "the two runs did not exercise different split paths"
    assert summary_p == summary_s, f"node count or total mass differs: {summary_p} vs {summary_s}"
    assert order_p == order_s, "parallel split changed the treewalk order"
    assert nodes_p == nodes_s, "parallel split changed node moments or geometry"
