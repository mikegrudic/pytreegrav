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
from pytreegrav.cuda import (
    CudaAccel,
    CudaPotential,
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
def test_frontend_device_cuda_matches_cpu(fn, cls):
    """The device="cuda" flag must agree with the CPU tree path to within the float32 tail."""
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    cpu = fn(x, m, h, tree=tree, parallel=True, theta=THETA)
    gpu = fn(x, m, h, tree=tree, theta=THETA, device="cuda")
    assert gpu.shape == cpu.shape
    d = np.abs(gpu - cpu)
    assert np.sqrt((d**2).mean()) / np.abs(cpu).std() < 1e-4


@pytest.mark.parametrize("fn", [Potential, Accel])
def test_frontend_rejects_bad_device_combinations(fn):
    """Validated without a GPU: these raise before any CUDA import."""
    x, m, h = cloud(500)
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        fn(x, m, h, device="gpu")
    with pytest.raises(ValueError, match="monopole only"):
        fn(x, m, h, device="cuda", quadrupole=True)
