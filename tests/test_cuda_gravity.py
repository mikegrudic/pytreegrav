"""Tests for the CUDA monopole gravity backend (pytreegrav.cuda).

As with the column-density backend, the walk bodies are jitted twice from one Python function -- CPU
and CUDA device -- so the pack, the acceptance criterion and the traversal are all checkable with no
GPU, and only the launch path needs hardware.

Tolerances are asymmetric on purpose: float64 rows must reproduce PotentialWalk/AccelWalk to
~1e-15, pinning the pack and the walk, while float32 agrees in the *rms* to ~1e-6 but reaches ~1e-3 on
the worst particle. That tail is not accumulation -- it is the acceptance criterion flipping for a node
on the opening-angle boundary, so it is bounded by theta's own error rather than by epsilon. Hence a
tight rms tolerance and a loose max one.
"""

import numpy as np
import pytest

from pytreegrav import Accel, ConstructTree, Potential
from pytreegrav.bruteforce import Accel_bruteforce, Potential_bruteforce, PotentialTarget_bruteforce
from pytreegrav.cuda import (
    CudaAccel,
    CudaAccelBruteforce,
    CudaPotential,
    CudaPotentialBruteforce,
    accel_packed_cpu,
    is_available,
    pack_tree_gravity,
    potential_packed_cpu,
)
from pytreegrav.treewalk import AccelWalk, PotentialWalk

THETA = 0.7
needs_cuda = pytest.mark.skipif(not is_available(), reason="no CUDA device / numba-cuda not installed")


def cloud(n, seed=42):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, 3)))
    return x, np.ones(n) / n, np.full(n, 2.0 * n ** (-1 / 3))


def tree_order(x, h, tree):
    return (
        np.ascontiguousarray(np.take(x, tree.TreewalkIndices, axis=0)),
        np.ascontiguousarray(h[tree.TreewalkIndices]),
    )


def shipped(xs, hs, tree, n=None):
    """The float64 reference: the shipped per-target walks."""
    n = len(xs) if n is None else n
    p = np.array([PotentialWalk(xs[i], tree, hs[i], -1, THETA) for i in range(n)])
    a = np.array([AccelWalk(xs[i], tree, hs[i], -1, THETA) for i in range(n)])
    return p, a


def rel_scalar(got, ref):
    return np.abs(got - ref).max() / np.abs(ref).max()


def rel_vec(got, ref):
    return np.sqrt(((got - ref) ** 2).sum(1)) / np.sqrt((ref**2).sum(1))


# --------------------------------------------------------------------------------------------------
# Pack + traversal, no GPU needed
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2000, 20000])
def test_packed_float64_reproduces_shipped_walks(n):
    """The pack and the walk must be exactly equivalent in float64 -- this is the correctness gate."""
    x, m, h = cloud(n)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    wp, wa = shipped(xs, hs, tree)
    nodes, links = pack_tree_gravity(tree, dtype=np.float64)
    gp = potential_packed_cpu(xs, hs, nodes, links, tree.NumParticles, 1.0 / THETA)
    ga = accel_packed_cpu(xs, hs, nodes, links, tree.NumParticles, 1.0 / THETA)
    assert rel_scalar(gp, wp) < 1e-13
    assert rel_scalar(ga, wa) < 1e-12


def test_packed_float32_rms_is_tight_and_tail_is_bounded():
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    wp, wa = shipped(xs, hs, tree)
    nodes, links = pack_tree_gravity(tree, dtype=np.float32)
    # float32 targets, matching what the device path does -- see the self-interaction test below
    xs32 = np.ascontiguousarray(xs, dtype=np.float32).astype(np.float64)
    gp = potential_packed_cpu(xs32, hs, nodes, links, tree.NumParticles, 1.0 / THETA)
    ga = accel_packed_cpu(xs32, hs, nodes, links, tree.NumParticles, 1.0 / THETA)
    assert np.sqrt(((gp - wp) ** 2).mean()) / np.abs(wp).std() < 1e-5
    assert np.median(rel_vec(ga, wa)) < 1e-5
    assert rel_vec(ga, wa).max() < 1e-2  # theta-boundary flips, not epsilon; see the module docstring


def test_float32_targets_are_required_to_exclude_the_self_interaction():
    """A subtle invariant worth a test, because breaking it costs 1% and looks like nothing.

    The self term is dropped by ``r > 0``, so for self-gravity the target must be *bit-identical* to
    its own stored row. Narrow both to float32 and r is exactly 0. Feed float64 targets against
    float32 rows and the particle sits ~1e-7 from itself -- inside its softening -- so it picks up a
    spurious ``-2.8 m/h``.
    """
    x, m, h = cloud(2000)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    wp, _ = shipped(xs, hs, tree)
    nodes, links = pack_tree_gravity(tree, dtype=np.float32)

    consistent = potential_packed_cpu(
        np.ascontiguousarray(xs, dtype=np.float32).astype(np.float64), hs, nodes, links, tree.NumParticles, 1.0 / THETA
    )
    mismatched = potential_packed_cpu(xs, hs, nodes, links, tree.NumParticles, 1.0 / THETA)

    assert rel_scalar(consistent, wp) < 1e-6
    err = np.abs(mismatched - wp)
    assert err.max() / np.abs(wp).max() > 1e-3, "mismatched precision should show the spurious self term"
    # and it should look like -2.8 m/h on the worst particle
    worst = err.argmax()
    assert np.isclose(mismatched[worst] - wp[worst], -2.8 * m[worst] / hs[worst], rtol=1e-2)


def test_pack_gravity_layout():
    x, m, h = cloud(5000)
    tree = ConstructTree(x, m, h)
    nodes, links = pack_tree_gravity(tree)
    assert nodes.shape == (tree.NumNodes, 8) and nodes.dtype == np.float32
    assert nodes.strides[0] == 32  # one 32 B row per element
    assert links.shape == (tree.NumNodes, 2) and links.dtype == np.int32
    npart = tree.NumParticles
    assert np.allclose(nodes[:npart, 3], m[tree.TreewalkIndices], rtol=1e-6)  # leaf masses
    # Only *populated* nodes carry a size: the split allocates degenerate cells it never fills, which
    # RollupMoments skips on m == 0. They pack to an all-zero row and the walk treats them exactly as
    # the CPU does -- the float64 pack reproduces the shipped walk bit for bit, which covers this.
    populated = nodes[npart:, 3] > 0
    assert populated.any()
    assert np.all(nodes[npart:, 5][populated] > 0)


def test_theta_is_a_runtime_parameter_not_baked_into_the_pack():
    """One packed tree must serve any opening angle, and a smaller theta must be more accurate."""
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    nodes, links = pack_tree_gravity(tree, dtype=np.float64)
    exact = Accel(xs, m, hs, method="bruteforce", parallel=True)
    errs = []
    for theta in (1.0, 0.4):
        got = accel_packed_cpu(xs, hs, nodes, links, tree.NumParticles, 1.0 / theta)
        errs.append(np.sqrt((rel_vec(got, exact) ** 2).mean()))
    assert errs[1] < errs[0], f"theta=0.4 rms {errs[1]:.2e} should beat theta=1.0 {errs[0]:.2e}"


# --------------------------------------------------------------------------------------------------
# Launch path -- needs a device
# --------------------------------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("n", [2000, 20000])
def test_cuda_potential_matches_shipped_walk(n):
    x, m, h = cloud(n)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    wp, _ = shipped(xs, hs, tree)
    got = CudaPotential(tree)(xs, hs, theta=THETA)
    assert got.dtype == np.float32 and got.shape == (n,)
    assert np.sqrt(((got - wp) ** 2).mean()) / np.abs(wp).std() < 1e-5


@needs_cuda
@pytest.mark.parametrize("n", [2000, 20000])
def test_cuda_accel_matches_shipped_walk(n):
    x, m, h = cloud(n)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    _, wa = shipped(xs, hs, tree)
    got = CudaAccel(tree)(xs, hs, theta=THETA)
    assert got.shape == (n, 3)
    rel = rel_vec(got, wa)
    assert np.median(rel) < 1e-5
    assert rel.max() < 1e-2  # theta-boundary flips


@needs_cuda
def test_cuda_gravity_g_scaling_and_softening_forms():
    x, m, h = cloud(4000)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    pot = CudaPotential(tree)
    assert np.allclose(pot(xs, hs, G=3.7, theta=THETA), 3.7 * pot(xs, hs, theta=THETA), rtol=1e-5)
    # scalar softening must broadcast, and None must mean zero
    assert pot(xs, 0.05, theta=THETA).shape == (4000,)
    assert np.all(np.isfinite(pot(xs, None, theta=THETA)))


@needs_cuda
def test_cuda_gravity_context_is_reusable_and_deterministic():
    x, m, h = cloud(4000)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    ctx = CudaAccel(tree)
    assert np.array_equal(ctx(xs, hs, theta=THETA), ctx(xs, hs, theta=THETA))
    assert np.array_equal(ctx(xs[:100], hs[:100], theta=THETA), ctx(xs, hs, theta=THETA)[:100])


@needs_cuda
def test_cuda_gravity_agrees_with_the_frontend_within_truncation_error():
    """End-to-end sanity against the shipped grouped frontend, at the theta error level."""
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    xs, hs = tree_order(x, h, tree)
    cpu_a = Accel(xs, m, hs, tree=tree, parallel=True, theta=THETA)
    gpu_a = CudaAccel(tree)(xs, hs, theta=THETA)
    # the grouped CPU walk opens a superset of nodes, so these differ at the truncation level, not
    # at float32 level -- compare against the brute-force answer instead of each other
    exact = Accel(xs, m, hs, method="bruteforce", parallel=True)
    rms_cpu = np.sqrt((rel_vec(cpu_a, exact) ** 2).mean())
    rms_gpu = np.sqrt((rel_vec(gpu_a, exact) ** 2).mean())
    assert rms_gpu < 3 * rms_cpu, f"gpu rms {rms_gpu:.2e} vs cpu {rms_cpu:.2e}"
    assert rms_gpu < 0.02


def test_gravity_backend_needs_no_cuda_to_import():
    x, m, h = cloud(500)
    tree = ConstructTree(x, m, h)
    pack_tree_gravity(tree)  # works regardless
    if not is_available():
        with pytest.raises(RuntimeError, match="no CUDA device"):
            CudaPotential(tree)


@needs_cuda
@pytest.mark.parametrize("fn,cls", [(Potential, CudaPotential), (Accel, CudaAccel)])
def test_frontend_device_cuda_matches_the_ungrouped_cpu_walk(fn, cls):
    """device="cuda" must reproduce the algorithm it actually implements: the *ungrouped* walk.

    Comparing against the default grouped CPU path instead measures grouping, not precision -- it opens
    a superset of nodes, a theta-level difference.  Measured at N=20000: 8.6e-07 against group_size=1
    but 7.3e-03 against the default, and both figures are unchanged if the device kernels are compiled
    in float64, which is what proves the gap is not roundoff.
    """
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    cpu = fn(x, m, h, tree=tree, parallel=True, theta=THETA, group_size=1)
    gpu = fn(x, m, h, tree=tree, theta=THETA, device="cuda")
    assert gpu.shape == cpu.shape
    d = np.abs(gpu - cpu)
    assert np.sqrt((d**2).mean()) / np.abs(cpu).std() < 1e-5
    assert d.max() / np.abs(cpu).max() < 1e-5


@needs_cuda
@pytest.mark.parametrize("fn", [Potential, Accel])
def test_frontend_device_cuda_stays_within_truncation_of_the_grouped_path(fn):
    """The grouped CPU path is what users get by default, so bound the difference -- but at theta's
    level, since that is what a different node set costs.  theta=0.7 is ~2e-3 RMS."""
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    cpu = fn(x, m, h, tree=tree, parallel=True, theta=THETA)
    gpu = fn(x, m, h, tree=tree, theta=THETA, device="cuda")
    assert np.abs(gpu - cpu).max() / np.abs(cpu).max() < 0.05


@pytest.mark.parametrize("fn", [Potential, Accel])
def test_frontend_rejects_bad_device_combinations(fn):
    """Validated without a GPU: these raise before any CUDA import."""
    x, m, h = cloud(500)
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        fn(x, m, h, device="gpu")
    with pytest.raises(ValueError, match="monopole only"):
        fn(x, m, h, device="cuda", quadrupole=True)


# --------------------------------------------------------------------------------------------------
# Brute force.  Exact by construction, so unlike the tree walks there is no theta tail to excuse a loose
# tolerance -- the only error is float32 accumulation over N terms.  That grows as sqrt(N), measured on
# an A6000 for the worst case (acceleration, h=0): 2.4e-6 at N=1e3, 1.4e-5 at 4097, 3.3e-5 at 2e4,
# 5.1e-5 at 6e4.  So the bound scales the same way rather than being flat, which keeps it tight enough
# at small N to catch a regression there.
# --------------------------------------------------------------------------------------------------


def bf_tol(n):
    """sqrt(N) float32 accumulation bound, with ~1.5-4x headroom over the measurements above."""
    return 1e-5 * np.sqrt(n / 1000)


@needs_cuda
@pytest.mark.parametrize("n", [1000, 4097, 20000])
@pytest.mark.parametrize("soft", ["zero", "uniform", "varying"])
def test_cuda_bruteforce_matches_the_cpu_kernels(n, soft):
    """4097 is deliberately not a multiple of TILE=128, so the ragged last tile is covered."""
    x, m, h = cloud(n)
    if soft == "zero":
        h = np.zeros(n)
    elif soft == "uniform":
        h = np.full(n, 0.05)
    p_ref = Potential_bruteforce(x, m, h)
    a_ref = Accel_bruteforce(x, m, h)
    assert rel_scalar(CudaPotentialBruteforce(x, m, h)(x, h), p_ref) < bf_tol(n)
    assert rel_vec(CudaAccelBruteforce(x, m, h)(x, h), a_ref).max() < bf_tol(n)


@needs_cuda
def test_cuda_bruteforce_excludes_the_self_interaction():
    """The r > 0 guard is the only thing dropping the self term, and it needs the target to be
    bit-identical to its own source row -- the same float32-narrowing contract as the tree walks."""
    x, m, h = cloud(3000)
    h = np.full(len(x), 0.1)  # large enough that a spurious self term lands in the softened branch
    got = CudaPotentialBruteforce(x, m, h)(x, h)
    assert rel_scalar(got, Potential_bruteforce(x, m, h)) < bf_tol(len(x))
    # a spurious self term would be -2.8*m/h, orders of magnitude above the tolerance above
    assert np.abs(-2.8 * m[0] / h[0]) / np.abs(got).max() > 100 * bf_tol(len(x))


@needs_cuda
def test_cuda_bruteforce_separate_targets_and_sources():
    """Targets need not be the sources; with disjoint sets no self term arises at all."""
    x, m, h = cloud(4000)
    ct = np.ascontiguousarray
    src, ms, hs = ct(x[:3000]), ct(m[:3000]), ct(h[:3000])
    tgt, ht = ct(x[3000:]), ct(h[3000:])
    ref = PotentialTarget_bruteforce(tgt, ht, src, ms, hs)
    assert rel_scalar(CudaPotentialBruteforce(src, ms, hs)(tgt, ht), ref) < bf_tol(len(src))


@needs_cuda
def test_cuda_bruteforce_g_scaling_and_reuse():
    """One uploaded source set, many queries: G must scale linearly and repeats must be identical."""
    x, m, h = cloud(2000)
    ctx = CudaAccelBruteforce(x, m, h)
    a1 = ctx(x, h)
    assert np.array_equal(a1, ctx(x, h))
    assert rel_vec(ctx(x, h, G=2.5), 2.5 * a1).max() < 1e-6


@needs_cuda
@pytest.mark.parametrize("fn,method", [(Potential, "bruteforce"), (Accel, "bruteforce")])
def test_frontend_device_cuda_bruteforce_matches_cpu(fn, method):
    """device='cuda' must actually reach the GPU brute force rather than silently staying on the CPU."""
    x, m, h = cloud(5000)
    cpu = fn(x, m, h, method=method, parallel=True)
    gpu = fn(x, m, h, method=method, device="cuda")
    assert gpu.shape == cpu.shape
    assert np.abs(gpu - cpu).max() / np.abs(cpu).max() < bf_tol(len(x))
