"""Tests for the optional CUDA column-density backend (pytreegrav.cuda).

Most of this runs with no GPU. The walk body is jitted twice from one Python function -- CPU and CUDA
device function -- so the packed tree and the traversal can be checked on any machine, and only the
launch path needs hardware. Tests that do need a device are marked and skip cleanly.

Two precisions are checked separately, because they answer different questions:

    float64 rows   must agree with treewalk.ColumnDensityWalk_singleray to ~1e-13, which pins the
                   repack, the overloaded slots and the traversal as equivalent
    float32 rows   ~2e-06, i.e. what narrowing storage costs on its own

The float64 figure is limited by the *reference*, not by this walk: the shipped walk gets the impact
parameter as ``r^2 - z^2``, which is the less well-conditioned form (see the module docstring).
"""

import numpy as np
import pytest

from pytreegrav import ColumnDensity, ConstructTree
from pytreegrav.cuda import CudaColumnDensity, column_density_packed_cpu, is_available, pack_tree
from pytreegrav.treewalk import ColumnDensityWalk_singleray

SIX_RAYS = np.vstack([np.eye(3), -np.eye(3)])
needs_cuda = pytest.mark.skipif(not is_available(), reason="no CUDA device / numba-cuda not installed")


def cloud(n, seed=42, radius_scale=3.0):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, 3)))
    return x, np.ones(n) / n, np.full(n, radius_scale * n ** (-1 / 3))


def morton_targets(x, tree):
    return np.ascontiguousarray(np.take(x, tree.TreewalkIndices, axis=0))


def reference(pos, rays, tree):
    return np.array([[ColumnDensityWalk_singleray(p, r, tree) for r in rays] for p in pos])


def rel_err(a, b):
    scale = np.abs(b).max()
    return np.abs(a - b).max() / (scale if scale > 0 else 1.0)


# --------------------------------------------------------------------------------------------------
# Repack + traversal, no GPU needed
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2000, 20000])
@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-13), (np.float32, 1e-5)])
def test_packed_walk_matches_shipped_walk(n, dtype, tol):
    x, m, h = cloud(n)
    tree = ConstructTree(x, m, h)
    xs = morton_targets(x, tree)
    nodes, links = pack_tree(tree, dtype=dtype)
    got = column_density_packed_cpu(xs, np.ascontiguousarray(SIX_RAYS), nodes, links, tree.NumParticles)
    assert rel_err(got, reference(xs, SIX_RAYS, tree)) < tol


def test_packed_row_shape_and_footprint():
    """The layout is load-bearing: one 32 B float32 row per element, half the float64 SoA footprint."""
    x, m, h = cloud(5000)
    tree = ConstructTree(x, m, h)
    nodes, links = pack_tree(tree)
    assert nodes.shape == (tree.NumNodes, 8)
    assert nodes.dtype == np.float32 and nodes.strides[0] == 32
    assert links.shape == (tree.NumNodes, 2) and links.dtype == np.int32
    assert nodes.nbytes < tree.NumNodes * 7 * 8  # beats the 7-array float64 SoA it replaces


def test_packed_walk_handles_zero_radii():
    """Zero-radius particles pack to a zeroed row and must contribute nothing, as on the CPU path."""
    x, m, h = cloud(3000)
    h = h.copy()
    h[:30] = 0.0
    tree = ConstructTree(x, m, h)
    xs = morton_targets(x, tree)
    nodes, links = pack_tree(tree)
    got = column_density_packed_cpu(xs, np.ascontiguousarray(SIX_RAYS), nodes, links, tree.NumParticles)
    assert np.isfinite(got).all()
    assert rel_err(got, reference(xs, SIX_RAYS, tree)) < 1e-5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_packed_row_values(dtype):
    """Pin every slot against the formulas directly, since the pack is a hand-written parallel kernel.

    Includes zero-radius leaves (row stays zeroed past the COM) and the node reach, which is the one
    slot computed rather than copied.
    """
    x, m, h = cloud(4000)
    h = h.copy()
    h[:40] = 0.0
    tree = ConstructTree(x, m, h)
    n, npart = tree.NumNodes, tree.NumParticles
    nodes, links = pack_tree(tree, dtype=dtype)
    hs = np.asarray(tree.Softenings[:n], np.float64)
    ms = np.asarray(tree.Masses[:n], np.float64)

    assert np.array_equal(nodes[:, 0:3], dtype(tree.Coordinates[:n]))
    assert np.array_equal(links[:, 0], tree.NextBranch[:n])
    assert np.array_equal(links[:, 1], tree.FirstSubnode[:n])
    assert np.all(nodes[:, 7] == 0)  # pad

    leaf = np.zeros(n, bool)
    leaf[:npart] = True
    live = leaf & (hs > 0)
    assert live.any() and (leaf & (hs == 0)).any()
    for slot, want in ((3, hs**2), (4, 1.0 / hs), (6, 1.0 / hs**2)):
        with np.errstate(divide="ignore"):
            assert np.array_equal(nodes[live, slot], dtype(want)[live])
    assert np.array_equal(nodes[live, 5], dtype(3.0 / (4.0 * np.pi) * ms / hs**2)[live])
    assert np.all(nodes[leaf & (hs == 0), 3:7] == 0)  # no cross-section, so nothing to contribute

    node = ~leaf
    reach = hs + np.sqrt(3) / 2 * np.asarray(tree.Sizes[:n], np.float64) + np.asarray(tree.Deltas[:n], np.float64)
    assert np.array_equal(nodes[node, 3], dtype(reach**2)[node])


def test_pack_raises_rather_than_uploading_inf():
    """The leaf prefactor goes as M/h^2, so a tiny radius can exceed float32; that must not be silent."""
    x, m, h = cloud(500)
    h = h.copy()
    h[0] = 1e-22
    with pytest.raises(ValueError, match="overflowed"):
        pack_tree(ConstructTree(x, m, h))
    pack_tree(ConstructTree(x, m, h), dtype=np.float64)  # fine with the wider type


def test_stable_impact_parameter_beats_the_subtractive_form_in_float32():
    """The reason the kernel computes |d - zn|^2 rather than r^2 - z^2.

    The subtraction cancels for a nearly radial ray with relative error ~eps (r/b)^2, which float64
    absorbs and float32 does not -- measured 2.0e-02 before this was fixed.  Reproduced here directly
    on the geometry, so the rationale cannot quietly stop being true.
    """
    d = np.array([0.0, 0.11, 5.0], dtype=np.float32)  # r/b ~ 45, as at N=2e4
    n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    z = np.dot(d, n)
    subtractive = np.float32(np.dot(d, d)) - np.float32(z * z)
    b = d - z * n
    stable = np.dot(b, b)
    exact = np.float64(0.11) ** 2
    assert abs(stable - exact) / exact < 1e-6
    assert abs(subtractive - exact) / exact > 100 * abs(stable - exact) / exact


# --------------------------------------------------------------------------------------------------
# Launch path -- needs a device
# --------------------------------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("n", [2000, 20000])
def test_cuda_matches_shipped_walk(n):
    x, m, h = cloud(n)
    tree = ConstructTree(x, m, h)
    xs = morton_targets(x, tree)
    got = CudaColumnDensity(tree)(xs, SIX_RAYS)
    assert got.dtype == np.float32
    assert rel_err(got, reference(xs, SIX_RAYS, tree)) < 1e-5


@needs_cuda
def test_cuda_context_is_reusable_and_deterministic():
    x, m, h = cloud(4000)
    tree = ConstructTree(x, m, h)
    xs = morton_targets(x, tree)
    ctx = CudaColumnDensity(tree)
    assert np.array_equal(ctx(xs, SIX_RAYS), ctx(xs, SIX_RAYS))
    assert np.array_equal(ctx(xs[:100], SIX_RAYS), ctx(xs, SIX_RAYS)[:100])  # target subset


@needs_cuda
def test_cuda_normalizes_rays_and_accepts_a_single_ray():
    x, m, h = cloud(3000)
    tree = ConstructTree(x, m, h)
    xs = morton_targets(x, tree)
    ctx = CudaColumnDensity(tree)
    unit = np.array([[0.0, 0.0, 1.0]])
    assert rel_err(ctx(xs, 7.3 * unit), ctx(xs, unit)) < 1e-6  # scale must not matter
    assert ctx(xs, unit).shape == (len(xs), 1)


@needs_cuda
def test_cuda_order_independence():
    """Morton order is for coherence only; any target order must give the same answer."""
    x, m, h = cloud(3000)
    tree = ConstructTree(x, m, h)
    xs = morton_targets(x, tree)
    perm = np.random.default_rng(0).permutation(len(xs))
    ctx = CudaColumnDensity(tree)
    assert np.array_equal(ctx(np.ascontiguousarray(xs[perm]), SIX_RAYS), ctx(xs, SIX_RAYS)[perm])


@needs_cuda
def test_frontend_device_cuda_matches_cpu():
    x, m, h = cloud(20000)
    tree = ConstructTree(x, m, h)
    cpu = ColumnDensity(x, m, h, rays=SIX_RAYS, tree=tree, parallel=True)
    gpu = ColumnDensity(x, m, h, rays=SIX_RAYS, tree=tree, device="cuda")
    assert rel_err(gpu, cpu) < 1e-5


def test_frontend_rejects_unsupported_device_combinations():
    """Validated without a GPU: these raise before any CUDA import."""
    x, m, h = cloud(200)
    with pytest.raises(ValueError, match="must be 'cpu' or 'cuda'"):
        ColumnDensity(x, m, h, rays=SIX_RAYS, device="gpu")
    with pytest.raises(ValueError, match="ray-traced path"):
        ColumnDensity(x, m, h, rays=None, device="cuda")
    with pytest.raises(ValueError, match="ray-traced path"):
        ColumnDensity(x, m, h, rays=SIX_RAYS, randomize_rays=True, device="cuda")


def test_importing_pytreegrav_does_not_require_cuda():
    """pytreegrav.cuda must stay out of the package __init__, so the core import never needs a GPU.

    Checked in a subprocess: this test module imports pytreegrav.cuda itself (and is_available() does
    ``from numba import cuda``), so sys.modules in-process cannot answer the question.
    """
    import subprocess
    import sys
    import textwrap

    src = textwrap.dedent("""
        import sys
        import pytreegrav
        from pytreegrav import ColumnDensity, Accel, Potential
        assert "pytreegrav.cuda" not in sys.modules, "cuda backend imported by the package __init__"
        assert "numba.cuda" not in sys.modules, "numba.cuda pulled in by a plain import"
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=300, check=False)
    assert r.returncode == 0, f"plain import of pytreegrav touched CUDA: {r.stdout}{r.stderr[-2000:]}"
    assert "OK" in r.stdout
