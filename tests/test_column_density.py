"""Tests for the column-density estimators.

The ray-traced walks are *exact*: a node opens whenever its bounding sphere (inflated by the node's
max particle radius) could intersect the ray, and only leaf particles contribute.  So the tree must
agree with direct summation to machine precision -- there is no opening-angle truncation error to
absorb, unlike the gravity walks.  That makes tree-vs-direct the sharp test it is here.

The physics is one closed form.  A particle of mass M and radius h is a uniform sphere of density
rho = 3M/(4 pi h^3).  A ray from ``pos`` along unit ``n`` sees it at separation ``d = x_p - pos``,
with along-ray coordinate ``z = d.n`` and impact parameter ``b = sqrt(|d|^2 - z^2)``; it misses
unless ``b < h``.  Outside the sphere the whole chord is traversed, inside only the forward half:

    outside:  rho * 2 sqrt(h^2 - b^2)      = 3M/(4 pi h^2) * 2 sqrt(1 - (b/h)^2)
    inside:   rho * (z + sqrt(h^2 - b^2))  = 3M/(4 pi h^2) * (z/h + sqrt(1 - (b/h)^2))
"""

import os

import numpy as np
import pytest

from pytreegrav import ColumnDensity, ConstructTree
from pytreegrav.grouped_treewalk import _morton_order
from pytreegrav.misc import random_rotation
from pytreegrav.treewalk import (
    COLUMN_GROUP_MIN_TARGETS,
    MULTIRAY_MAX_RAYS,
    ColumnDensity_grouped,
    ColumnDensity_grouped_parallel,
    ColumnDensityBinned_grouped,
    ColumnDensityBinned_grouped_parallel,
    ColumnDensityWalk_binned,
    ColumnDensityWalk_multiray,
    ColumnDensityWalk_singleray,
)

RTOL = 1e-12
SIX_RAYS = np.vstack([np.eye(3), -np.eye(3)])


# --------------------------------------------------------------------------------------------------
# Independent reference implementation: direct summation over all sources, no tree.
# --------------------------------------------------------------------------------------------------


def column_reference(pos_target, rays, x, m, h):
    """Direct-summation column density, shape (N_target, N_rays).  Pure numpy, no tree."""
    pos_target = np.atleast_2d(pos_target)
    rays = np.atleast_2d(rays)
    d = x[None, :, :] - pos_target[:, None, :]  # (Nt, Ns, 3)
    r2 = (d * d).sum(-1)  # (Nt, Ns)
    z = np.einsum("tsk,rk->tsr", d, rays)  # (Nt, Ns, Nr)
    b2 = np.maximum(r2[:, :, None] - z * z, 0.0)  # (Nt, Ns, Nr)

    h2 = (h * h)[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        fac = np.where(h > 0, 3 * m / (4 * np.pi * np.where(h > 0, h, 1) ** 2), 0.0)[None, :, None]
        chord = np.sqrt(np.maximum(1 - b2 / np.where(h2 > 0, h2, 1), 0.0))
        zoverh = z / np.where(h > 0, h, 1)[None, :, None]

    hits = (b2 < h2) & (h > 0)[None, :, None]
    origin_outside = (r2 > (h * h)[None, :])[:, :, None]
    full = hits & origin_outside & (z > 0)
    partial = hits & ~origin_outside

    contrib = np.where(full, fac * 2 * chord, 0.0) + np.where(partial, fac * (zoverh + chord), 0.0)
    return contrib.sum(axis=1)


def rel_err(a, b):
    scale = np.abs(b).max()
    return np.abs(a - b).max() / (scale if scale > 0 else 1.0)


def cloud(N, seed=42, radius_scale=3.0):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(N, 3)))
    m = np.ones(N) / N
    h = np.full(N, radius_scale * N ** (-1 / 3))
    return x, m, h


GLASS_CBRT = 64
GLASS_FILE = f"glass_{GLASS_CBRT}.hdf5"
GLASS_URL = f"https://users.flatironinstitute.org/~mgrudic/glass/{GLASS_FILE}"
# Same cache directory makecloud uses, so a machine that has ever run makecloud pays nothing.
GLASS_CACHE = os.path.join(os.path.expanduser("~"), ".makecloud_glass", GLASS_FILE)


def glass_coords():
    """Positions of a pre-relaxed periodic glass in [0,1]^3, as used by makecloud.

    A glass rather than a random (Poisson) realization is what makes the central-column test sharp:
    density noise falls off far faster than N^-1/2, so the discretized column converges to rho*R
    tightly.  Unlike a lattice, a glass has no preferred directions, so axis-aligned rays are not
    special.  Skips rather than fails when h5py or the network is unavailable.
    """
    h5py = pytest.importorskip("h5py", reason="glass-file tests need h5py")
    if not os.path.exists(GLASS_CACHE):
        import urllib.request

        try:
            os.makedirs(os.path.dirname(GLASS_CACHE), exist_ok=True)
            urllib.request.urlretrieve(GLASS_URL, GLASS_CACHE)
        except OSError as e:  # offline CI, DNS failure, 404 -- URLError/HTTPError subclass OSError
            pytest.skip(f"could not fetch {GLASS_URL}: {e}")
    with h5py.File(GLASS_CACHE, "r") as f:
        return np.array(f["Coordinates"][:], dtype=np.float64)


def glass_sphere_ic(R=0.45, M=1.0, eta=3.0):
    """Cut a uniform sphere of radius R out of the glass cube, centred on the box centre.

    Returns (x, m, h, centre, expected_column) where the sphere is centred on the origin and
    ``expected_column = rho * R`` is the exact column density from the centre of a uniform sphere
    of density rho out to infinity, along any direction.
    """
    pos = glass_coords()
    centre = np.full(3, 0.5)
    keep = np.linalg.norm(pos - centre, axis=1) < R
    x = np.ascontiguousarray(pos[keep] - centre)  # recentre so the sphere centre is the origin
    N = len(x)
    m = np.full(N, M / N)
    h = np.full(N, eta / GLASS_CBRT)  # glass spacing in the unit box is 1/cbrt
    rho = 3 * M / (4 * np.pi * R**3)
    return x, m, h, rho * R


def isotropic_rays(n, seed=0):
    rng = np.random.default_rng(seed)
    rays = rng.normal(size=(n, 3))
    return rays / np.linalg.norm(rays, axis=1)[:, None]


def spy(record, name, func):
    """Wrap ``func`` so calling it appends ``name`` to ``record``, for asserting which walk ran."""

    def wrapper(*args, **kwargs):
        record.append(name)
        return func(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------------------------
# Analytic single-particle chord integral -- exercises both branches of the kernel directly.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target, ray, label",
    [
        ([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], "head-on from outside"),
        ([-5.0, 0.4, 0.0], [1.0, 0.0, 0.0], "offset chord from outside"),
        ([5.0, 0.0, 0.0], [1.0, 0.0, 0.0], "pointing away -- sphere behind"),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], "origin at the centre (self-column)"),
        ([0.3, 0.2, -0.1], [0.0, 0.0, 1.0], "origin inside, off-centre"),
        ([-0.6, 0.0, 0.0], [1.0, 0.0, 0.0], "origin inside, near the edge"),
        ([-5.0, 1.5, 0.0], [1.0, 0.0, 0.0], "clean miss"),
    ],
)
def test_single_particle_chord(target, ray, label):
    """One uniform sphere: the walk must reproduce the closed-form chord integral."""
    x = np.array([[0.0, 0.0, 0.0]])
    m, h = np.array([2.0]), np.array([1.0])
    tree = ConstructTree(x, m, h)
    target, ray = np.array(target), np.array(ray, dtype=float)

    got = ColumnDensityWalk_singleray(target, ray, tree)
    want = column_reference(target, ray, x, m, h)[0, 0]
    assert abs(got - want) <= RTOL * max(abs(want), 1e-30), f"{label}: got {got}, want {want}"


def test_single_particle_analytic_values():
    """Spot-check the reference itself against hand-computed values, so the two can't drift together."""
    x = np.array([[0.0, 0.0, 0.0]])
    m, h = np.array([2.0]), np.array([1.0])
    tree = ConstructTree(x, m, h)
    rho = 3 * m[0] / (4 * np.pi * h[0] ** 3)

    # head-on from well outside: the full diameter
    got = ColumnDensityWalk_singleray(np.array([-5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), tree)
    assert np.isclose(got, rho * 2 * h[0], rtol=RTOL)

    # from the centre outward: one radius
    got = ColumnDensityWalk_singleray(np.zeros(3), np.array([1.0, 0.0, 0.0]), tree)
    assert np.isclose(got, rho * h[0], rtol=RTOL)

    # impact parameter b: chord 2*sqrt(h^2 - b^2)
    b = 0.6
    got = ColumnDensityWalk_singleray(np.array([-5.0, b, 0.0]), np.array([1.0, 0.0, 0.0]), tree)
    assert np.isclose(got, rho * 2 * np.sqrt(h[0] ** 2 - b**2), rtol=RTOL)

    # a clean miss contributes nothing
    got = ColumnDensityWalk_singleray(np.array([-5.0, 1.01, 0.0]), np.array([1.0, 0.0, 0.0]), tree)
    assert got == 0.0


def test_superposition_two_particles():
    """Column is additive in the sources: two spheres must give the sum of the singles."""
    m, h = np.array([1.0, 3.0]), np.array([0.5, 0.8])
    xa = np.array([[0.0, 0.0, 0.0]])
    xb = np.array([[2.0, 0.1, 0.0]])
    target, ray = np.array([-4.0, 0.05, 0.0]), np.array([1.0, 0.0, 0.0])

    both = ColumnDensityWalk_singleray(target, ray, ConstructTree(np.vstack([xa, xb]), m, h))
    a = ColumnDensityWalk_singleray(target, ray, ConstructTree(xa, m[:1], h[:1]))
    b = ColumnDensityWalk_singleray(target, ray, ConstructTree(xb, m[1:], h[1:]))
    assert a > 0 and b > 0, "test is vacuous unless both spheres are actually hit"
    assert np.isclose(both, a + b, rtol=RTOL)


# --------------------------------------------------------------------------------------------------
# The workhorse: the tree walk is exact, so it must match direct summation to machine precision.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("N", [50, 500])
@pytest.mark.parametrize("parallel", [False, True])
def test_tree_matches_direct_sum_six_rays(N, parallel):
    x, m, h = cloud(N)
    got = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=parallel)
    want = column_reference(x, SIX_RAYS, x, m, h)
    assert not np.isnan(got).any()
    assert rel_err(got, want) < RTOL


def test_tree_matches_direct_sum_random_rays():
    x, m, h = cloud(300, seed=7)
    rng = np.random.default_rng(3)
    rays = rng.normal(size=(17, 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    got = ColumnDensity(x, m, h, rays=rays, parallel=True)
    want = column_reference(x, rays, x, m, h)
    assert rel_err(got, want) < RTOL


def test_tree_matches_direct_sum_wide_radius_range():
    """Mixed particle sizes stress the node-opening radius (Softenings[no] is a per-node max)."""
    rng = np.random.default_rng(11)
    N = 400
    x = np.ascontiguousarray(rng.normal(size=(N, 3)))
    m = rng.uniform(0.5, 2.0, N)
    h = 10 ** rng.uniform(-2, 0, N)  # two decades of radii
    got = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    want = column_reference(x, SIX_RAYS, x, m, h)
    assert rel_err(got, want) < RTOL


def test_clustered_configuration():
    """Two well-separated clumps: deep trees, long empty stretches, and -- deliberately -- a large
    separation-to-radius ratio, where the impact parameter goes ill-conditioned.

    Walk and reference both compute ``b^2 = r^2 - z^2``.  For a nearly radial ray at large distance
    r ~ z, so that subtraction cancels: at separation/radius = 200 it loses ~4-5 digits, and
    ``sqrt(1 - q^2)`` amplifies the loss for edge-grazing rays.  Both lose the same digits and differ
    only in rounding -- a property of the formulation, not a defect, and harmless since the
    amplification peaks exactly where the contribution vanishes.  Hence 1e-9 rather than 1e-12.
    The cancellation-free ``b = |d - (d.n) n|`` fixes it but costs ~13% on the hot loop.
    """
    rng = np.random.default_rng(5)
    a = rng.normal(scale=0.1, size=(200, 3))
    b = rng.normal(scale=0.1, size=(200, 3)) + np.array([10.0, 0.0, 0.0])
    x = np.ascontiguousarray(np.vstack([a, b]))
    m = np.ones(len(x)) / len(x)
    h = np.full(len(x), 0.05)
    got = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    want = column_reference(x, SIX_RAYS, x, m, h)
    assert rel_err(got, want) < 1e-9


def test_optical_depth_is_linear_in_mass():
    """The documented 'pass sigma = opacity * mass' usage requires exact linearity in m."""
    x, m, h = cloud(200, seed=9)
    c1 = ColumnDensity(x, m, h, rays=SIX_RAYS)
    c2 = ColumnDensity(x, 3.7 * m, h, rays=SIX_RAYS)
    assert rel_err(c2, 3.7 * c1) < RTOL


# --------------------------------------------------------------------------------------------------
# The multiray walk must agree with N_rays single-ray walks.  This invariant is what lets the
# randomize_rays path be restructured to loop single-ray walks (the multiray walk costs
# O(N_rays^2) because it opens the union of all rays' node sets).
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nrays", [6, 13, 48])
def test_multiray_matches_singleray(nrays):
    x, m, h = cloud(400, seed=13)
    tree = ConstructTree(x, m, h)
    rng = np.random.default_rng(nrays)
    rays = rng.normal(size=(nrays, 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]

    for i in (0, 37, 199):
        multi = ColumnDensityWalk_multiray(x[i], rays, tree)
        single = np.array([ColumnDensityWalk_singleray(x[i], r, tree) for r in rays])
        assert rel_err(multi, single) < RTOL


# --------------------------------------------------------------------------------------------------
# A supplied tree may hold a different particle set than the targets, as ColumnDensity documents.  It
# used to permute targets by the *tree's* TreewalkIndices, which do not index pos: a larger tree
# raised IndexError, a smaller one silently returned too few rows.  Targets are now sorted on their own.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True])
def test_prebuilt_tree_with_different_particles(parallel):
    """Sources and targets are disjoint sets of different sizes -- the documented use of ``tree=``."""
    xs, ms, hs = cloud(400, seed=61)  # sources
    tree = ConstructTree(xs, ms, hs)
    rng = np.random.default_rng(62)
    targets = np.ascontiguousarray(rng.normal(size=(37, 3)) * 1.5)  # far fewer, and elsewhere

    got = ColumnDensity(targets, ms, hs, rays=SIX_RAYS, tree=tree, parallel=parallel)
    assert got.shape == (len(targets), 6)
    want = column_reference(targets, SIX_RAYS, xs, ms, hs)
    assert rel_err(got, want) < RTOL


def test_prebuilt_tree_with_more_particles_than_targets():
    """The case that used to raise IndexError: the tree is larger than the target array."""
    xs, ms, hs = cloud(1000, seed=63)
    tree = ConstructTree(xs, ms, hs)
    targets = np.ascontiguousarray(xs[:10])  # 10 targets against a 1000-particle tree
    got = ColumnDensity(targets, ms, hs, rays=SIX_RAYS, tree=tree)
    assert got.shape == (10, 6)
    assert rel_err(got, column_reference(targets, SIX_RAYS, xs, ms, hs)) < RTOL


def test_prebuilt_tree_of_the_same_particles_still_works():
    """Regression on the common path: passing the tree back in must match building it internally."""
    x, m, h = cloud(300, seed=64)
    tree = ConstructTree(x, m, h)
    with_tree = ColumnDensity(x, m, h, rays=SIX_RAYS, tree=tree, parallel=True)
    without = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    assert rel_err(with_tree, without) < RTOL
    assert rel_err(with_tree, column_reference(x, SIX_RAYS, x, m, h)) < RTOL


def test_return_tree_round_trips():
    """The tree handed back by return_tree must be reusable for the same targets."""
    x, m, h = cloud(300, seed=65)
    first, tree = ColumnDensity(x, m, h, rays=SIX_RAYS, return_tree=True)
    second = ColumnDensity(x, m, h, rays=SIX_RAYS, tree=tree)
    assert rel_err(second, first) < RTOL


@pytest.mark.parametrize("nrays", [6, MULTIRAY_MAX_RAYS, MULTIRAY_MAX_RAYS + 1, 48])
@pytest.mark.parametrize("parallel", [False, True])
def test_randomize_rays_matches_direct_sum(nrays, parallel):
    """``randomize_rays`` rotates the ray grid per target; both dispatch branches must be exact.

    At or below MULTIRAY_MAX_RAYS this takes the bundled multiray walk, above it one single-ray walk
    per direction; the parametrization straddles the threshold so neither branch can go untested.

    The reference must reproduce the seeding exactly: ColumnDensity reorders targets before walking,
    so the one at position ``k`` of the reordered array -- original index ``idx[k]`` -- is rotated by
    ``random_rotation(k)``, not ``random_rotation(idx[k])``.
    """
    x, m, h = cloud(200, seed=51)
    rays = isotropic_rays(nrays, seed=nrays)

    got = ColumnDensity(x, m, h, rays=rays, randomize_rays=True, parallel=parallel)

    # ColumnDensity built its tree from (x, m, h); rebuild the identical one for its walk order
    idx = ConstructTree(np.float64(x), np.float64(m), np.float64(h)).TreewalkIndices
    want = np.empty_like(got)
    for k, orig in enumerate(idx):
        want[orig] = column_reference(x[orig], rays @ random_rotation(k), x, m, h)[0]
    assert rel_err(got, want) < RTOL


@pytest.mark.parametrize("nrays", [MULTIRAY_MAX_RAYS, MULTIRAY_MAX_RAYS + 1])
def test_randomize_rays_with_prebuilt_tree(nrays):
    """Same check on the supplied-tree path, where the targets are Morton-sorted on their own."""
    x, m, h = cloud(200, seed=52)
    tree = ConstructTree(x, m, h)
    rays = isotropic_rays(nrays, seed=nrays)

    got = ColumnDensity(x, m, h, rays=rays, randomize_rays=True, tree=tree, parallel=True)

    idx = _morton_order(np.float64(x))
    want = np.empty_like(got)
    for k, orig in enumerate(idx):
        want[orig] = column_reference(x[orig], rays @ random_rotation(k), x, m, h)[0]
    assert rel_err(got, want) < RTOL


def test_randomize_rays_branches_agree():
    """The two dispatch branches are the same calculation, so straddling the threshold with the same
    ray directions must not change the answer beyond the extra rays themselves."""
    x, m, h = cloud(200, seed=53)
    tree = ConstructTree(x, m, h)
    rays = isotropic_rays(MULTIRAY_MAX_RAYS + 4, seed=1)

    # the first MULTIRAY_MAX_RAYS directions go through the bundled walk on their own ...
    small = ColumnDensity(x, m, h, rays=rays[:MULTIRAY_MAX_RAYS], randomize_rays=True, tree=tree)
    # ... and through the per-ray walk when the bundle is one over the threshold
    large = ColumnDensity(x, m, h, rays=rays, randomize_rays=True, tree=tree)
    assert rel_err(large[:, :MULTIRAY_MAX_RAYS], small) < RTOL


def test_multiray_matches_direct_sum():
    x, m, h = cloud(250, seed=17)
    tree = ConstructTree(x, m, h)
    got = np.array([ColumnDensityWalk_multiray(p, SIX_RAYS, tree) for p in x])
    want = column_reference(x, SIX_RAYS, x, m, h)
    assert rel_err(got, want) < RTOL


# --------------------------------------------------------------------------------------------------
# Angle-binned estimator (rays=None).  This one *is* approximate -- it buckets whole nodes into 6
# sky bins -- so it is checked for physical sanity and mass conservation, not machine precision.
# --------------------------------------------------------------------------------------------------


def test_binned_estimator_sane():
    x, m, h = cloud(500, seed=23)
    got = ColumnDensity(x, m, h, parallel=True)
    assert got.shape == (len(x), 6)
    assert not np.isnan(got).any()
    assert (got >= 0).all()


def test_binned_estimator_matches_raytrace_in_the_mean():
    """A spherical cloud seen from its own particles: the 6-bin average should track the ray-traced
    average to within the binning error, and must not be biased by a large factor."""
    x, m, h = cloud(2000, seed=29)
    binned = ColumnDensity(x, m, h, parallel=True).mean()
    traced = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True).mean()
    assert 0.5 < binned / traced < 2.0, f"binned {binned:.4g} vs traced {traced:.4g}"


def test_binned_estimator_linear_in_mass():
    x, m, h = cloud(300, seed=31)
    c1 = ColumnDensity(x, m, h)
    c2 = ColumnDensity(x, 2.5 * m, h)
    assert rel_err(c2, 2.5 * c1) < RTOL


# --------------------------------------------------------------------------------------------------
# Grouped walk: one traversal per (group, ray) instead of per (target, ray).  The reach is padded by
# the group's bbox half-diagonal, so it opens a superset -- but the extra leaves contribute nothing and
# each target accumulates in the same order, so the result must be *bit-identical*, not merely close.
# That is the sharpest assertion available, so it is the main one here.
# --------------------------------------------------------------------------------------------------


def per_target_walk(pos, rays, tree):
    """Reference: the per-target single-ray walk, looped in Python."""
    return np.array([[ColumnDensityWalk_singleray(p, r, tree) for r in rays] for p in pos])


@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16])
def test_grouped_is_bit_identical_to_per_target(group_size):
    x, m, h = cloud(1500, seed=71)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)  # grouping needs Morton order
    rays = isotropic_rays(5, seed=71)

    got = ColumnDensity_grouped(xs, rays, tree, group_size)
    want = per_target_walk(xs, rays, tree)
    assert np.array_equal(got, want), f"group_size={group_size}: max diff {np.abs(got - want).max()}"


def test_grouped_parallel_matches_serial():
    """The parallel variant is what the frontend actually calls, so check it explicitly rather than
    relying on the serial one.  Each group owns its own output rows, so there is no reduction and the
    result should be bit-identical, not merely close."""
    x, m, h = cloud(1500, seed=78)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    rays = isotropic_rays(5, seed=78)
    serial = ColumnDensity_grouped(xs, rays, tree, 16)
    par = ColumnDensity_grouped_parallel(xs, rays, tree, 16)
    assert np.array_equal(serial, par), f"max diff {np.abs(serial - par).max()}"


def test_grouped_matches_direct_sum():
    """Independent check against direct summation, not just against the other walk."""
    x, m, h = cloud(1100, seed=72)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    rays = isotropic_rays(3, seed=72)
    got = ColumnDensity_grouped(xs, rays, tree, 8)
    assert rel_err(got, column_reference(xs, rays, xs, m, h)) < RTOL


def test_grouped_handles_zero_radii():
    """The zero-radius guard is duplicated in the grouped kernel, so it needs its own check."""
    x, m, h = cloud(1200, seed=73)
    h = h.copy()
    h[:20] = 0.0
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    got = ColumnDensity_grouped(xs, SIX_RAYS, tree, 8)
    assert np.isfinite(got).all()
    assert np.array_equal(got, per_target_walk(xs, SIX_RAYS, tree))


def test_grouped_tail_group():
    """N not divisible by group_size: the final short group must still be handled."""
    x, m, h = cloud(1003, seed=74)  # 1003 = 8*125 + 3
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    rays = isotropic_rays(3, seed=74)
    assert np.array_equal(ColumnDensity_grouped(xs, rays, tree, 8), per_target_walk(xs, rays, tree))


@pytest.mark.parametrize("ntarget", [COLUMN_GROUP_MIN_TARGETS - 1, COLUMN_GROUP_MIN_TARGETS])
def test_frontend_dispatches_across_the_grouping_threshold(ntarget, monkeypatch):
    """Both sides of COLUMN_GROUP_MIN_TARGETS must be exercised and must agree with direct summation."""
    import pytreegrav.frontend as fe

    called = []
    for name in (
        "ColumnDensity_grouped",
        "ColumnDensity_grouped_parallel",
        "ColumnDensity_tree",
        "ColumnDensity_tree_parallel",
    ):
        real = getattr(fe, name)
        monkeypatch.setattr(fe, name, spy(called, name, real))

    x, m, h = cloud(ntarget, seed=75)
    got = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    expect = "grouped" if ntarget >= COLUMN_GROUP_MIN_TARGETS else "tree"
    assert any(expect in c for c in called), f"expected the {expect} walk, got {called}"
    assert rel_err(got, column_reference(x, SIX_RAYS, x, m, h)) < RTOL


def test_randomize_rays_never_uses_the_grouped_walk(monkeypatch):
    """Grouping needs a shared ray grid, which randomize_rays breaks by construction."""
    import pytreegrav.frontend as fe

    called = []
    for name in ("ColumnDensity_grouped", "ColumnDensity_grouped_parallel"):
        real = getattr(fe, name)
        monkeypatch.setattr(fe, name, spy(called, name, real))

    x, m, h = cloud(1500, seed=76)
    ColumnDensity(x, m, h, rays=SIX_RAYS, randomize_rays=True, parallel=True)
    assert called == [], f"grouped walk used with randomize_rays: {called}"


def test_grouped_group_size_one_matches_default_path():
    """group_size=1 degenerates to the per-target walk, so it must match the ungrouped frontend."""
    x, m, h = cloud(1500, seed=77)
    a = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True, group_size=1)
    b = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True, group_size=8)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------------------------------
# Grouped angular-binned estimator (rays=None).  Acceptance uses r_min, the nearest distance from the
# node to the group's bounding box, so the group opens a superset of nodes.  Nodes contribute here, so
# unlike the ray-traced grouped walk the answer *changes* -- for the better, since each node's mass is
# split across the sky bins more finely.  group_size=1 makes r_min the exact per-target distance.
# --------------------------------------------------------------------------------------------------


def per_target_binned(pos, tree, theta=0.5):
    return np.array([ColumnDensityWalk_binned(p, tree, theta) for p in pos])


def test_grouped_binned_group_size_one_matches_per_target():
    """group_size=1 collapses the bounding box to a point, so r_min is the per-target distance."""
    x, m, h = cloud(1500, seed=81)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    got = ColumnDensityBinned_grouped(xs, tree, 0.5, 1)
    assert rel_err(got, per_target_binned(xs, tree)) < RTOL


@pytest.mark.parametrize("group_size", [1, 16])
def test_grouped_binned_parallel_matches_serial(group_size):
    """Not bit-exact, unlike the ray-traced kernel: this one accumulates over the 6 bins in an inner
    loop (``for k in range(n_bins): out[t, k] += col_isotropic``), which fastmath reassociates
    differently under parallel=True.  Measured at one ulp -- 2.4e-16 relative -- so 1e-14 is tight."""
    x, m, h = cloud(1500, seed=82)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    a = ColumnDensityBinned_grouped(xs, tree, 0.5, group_size)
    b = ColumnDensityBinned_grouped_parallel(xs, tree, 0.5, group_size)
    assert rel_err(a, b) < 1e-14


def test_grouped_binned_conserves_mass_and_stays_sane():
    """Grouping must not create or destroy column: all bins positive, and the sky-averaged total
    stays within a few percent of the per-target walk (it shifts, but only at the theta level)."""
    x, m, h = cloud(2000, seed=83)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    ref = per_target_binned(xs, tree)
    for group_size in (4, 16):
        got = ColumnDensityBinned_grouped(xs, tree, 0.5, group_size)
        assert np.isfinite(got).all()
        assert (got >= 0).all()
        assert abs(got.mean() / ref.mean() - 1) < 0.05, f"group_size={group_size} shifted the sky mean"


def test_grouped_binned_handles_zero_radii():
    x, m, h = cloud(1200, seed=84)
    h = h.copy()
    h[:20] = 0.0
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    assert np.isfinite(ColumnDensityBinned_grouped(xs, tree, 0.5, 16)).all()


def test_grouped_binned_tail_group():
    x, m, h = cloud(1003, seed=85)
    tree = ConstructTree(x, m, h)
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    got = ColumnDensityBinned_grouped(xs, tree, 0.5, 16)
    assert got.shape == (1003, 6)
    assert np.isfinite(got).all() and (got >= 0).all()


@pytest.mark.parametrize("ntarget", [COLUMN_GROUP_MIN_TARGETS - 1, COLUMN_GROUP_MIN_TARGETS])
def test_frontend_dispatches_binned_across_the_threshold(ntarget, monkeypatch):
    import pytreegrav.frontend as fe

    called = []
    for name in (
        "ColumnDensityBinned_grouped",
        "ColumnDensityBinned_grouped_parallel",
        "ColumnDensity_tree",
        "ColumnDensity_tree_parallel",
    ):
        monkeypatch.setattr(fe, name, spy(called, name, getattr(fe, name)))

    x, m, h = cloud(ntarget, seed=86)
    got = ColumnDensity(x, m, h, parallel=True)  # rays=None -> binned
    expect = "Binned_grouped" if ntarget >= COLUMN_GROUP_MIN_TARGETS else "ColumnDensity_tree"
    assert any(expect in c for c in called), f"expected {expect}, got {called}"
    assert got.shape == (ntarget, 6)
    assert np.isfinite(got).all() and (got >= 0).all()


def test_grouped_binned_beats_per_target_on_the_glass_sphere():
    """The accuracy claim, against a known truth: the central column of a uniform sphere is rho*R in
    every direction, so bin-to-bin spread is pure error.  Grouping must shrink it."""
    x, m, h, expected = glass_sphere_ic(eta=3.0)
    tree = ConstructTree(x, m, h)
    centre = int(np.linalg.norm(x, axis=1).argmin())
    xs = np.take(x, tree.TreewalkIndices, axis=0)
    inv = np.empty(len(x), dtype=np.int64)
    inv[tree.TreewalkIndices] = np.arange(len(x))

    ref = ColumnDensityWalk_binned(x[centre], tree, 0.5) / expected
    got = ColumnDensityBinned_grouped_parallel(xs, tree, 0.5, 16)[inv[centre]] / expected
    spread_ref, spread_got = np.ptp(ref), np.ptp(got)
    assert spread_got < spread_ref, f"grouped spread {spread_got:.3f} should beat per-target {spread_ref:.3f}"
    assert abs(got.mean() - 1) <= abs(ref.mean() - 1) + 1e-3, f"grouped mean {got.mean():.4f} vs {ref.mean():.4f}"


# --------------------------------------------------------------------------------------------------
# Glass sphere: the column from the centre of a uniform sphere is exactly rho*R in every direction.
# The end-to-end physics check -- geometry, kernel mass normalization and traversal isotropy at once,
# against a closed form independent of the implementation.
#
# Calibration at eta = 3 (h = 3 * glass spacing), R = 0.45, N ~ 1.0e5, 200 isotropic rays:
#   ray-traced   mean/expected = 0.9979, per-ray scatter 0.0027, worst single ray within 1.0%
#   6-ray grid   mean/expected = 0.9970
#   binned       mean/expected = 0.9401, individual bins spanning 0.79 - 1.10
# The small ray-traced deficit is real and physical: smoothing carries a little mass outside R, and
# the angle-averaged column weights the interior by 1/r^2.  It shrinks with eta (0.9992 at eta = 2).
# --------------------------------------------------------------------------------------------------


def test_glass_sphere_central_column_raytraced():
    """Ray-traced column from the centre of a glass sphere must equal rho*R in every direction."""
    x, m, h, expected = glass_sphere_ic(eta=3.0)
    tree = ConstructTree(x, m, h)
    centre = np.zeros(3)
    rays = isotropic_rays(200)

    single = np.array([ColumnDensityWalk_singleray(centre, r, tree) for r in rays])
    assert abs(single.mean() / expected - 1) < 0.01, f"mean {single.mean():.5f} vs rho*R {expected:.5f}"
    # isotropy: no individual direction may stray far, not just the mean
    assert np.abs(single / expected - 1).max() < 0.02, f"worst ray {np.abs(single / expected - 1).max():.4f}"


def test_glass_sphere_central_column_multiray():
    """The multiray walk is the same calculation, so it must hit rho*R identically."""
    x, m, h, expected = glass_sphere_ic(eta=3.0)
    tree = ConstructTree(x, m, h)
    centre = np.zeros(3)
    rays = isotropic_rays(200)

    multi = ColumnDensityWalk_multiray(centre, rays, tree)
    single = np.array([ColumnDensityWalk_singleray(centre, r, tree) for r in rays])
    assert rel_err(multi, single) < RTOL
    assert abs(multi.mean() / expected - 1) < 0.01
    assert np.abs(multi / expected - 1).max() < 0.02


def test_glass_sphere_central_column_six_ray_grid():
    """The axis-aligned 6-ray grid is not a special direction for a glass (unlike a lattice)."""
    x, m, h, expected = glass_sphere_ic(eta=3.0)
    tree = ConstructTree(x, m, h)
    six = np.array([ColumnDensityWalk_singleray(np.zeros(3), r, tree) for r in SIX_RAYS])
    assert abs(six.mean() / expected - 1) < 0.01
    assert np.abs(six / expected - 1).max() < 0.02


def test_glass_sphere_central_column_binned():
    """The 6-bin estimator buckets whole nodes by dominant axis, so it is much cruder: it runs a few
    percent low overall and individual bins scatter by tens of percent.  Checked loosely, on purpose."""
    x, m, h, expected = glass_sphere_ic(eta=3.0)
    tree = ConstructTree(x, m, h)
    binned = ColumnDensityWalk_binned(np.zeros(3), tree)
    assert binned.shape == (6,)
    assert abs(binned.mean() / expected - 1) < 0.15, f"mean {binned.mean():.5f} vs rho*R {expected:.5f}"
    assert np.abs(binned / expected - 1).max() < 0.35


def test_glass_sphere_central_column_via_frontend():
    """End-to-end through ColumnDensity, which also exercises the Morton permute/un-permute.

    The glass has exactly one particle at the box centre, so after recentring there is a particle at
    the origin -- its own row is the central column, with no need to insert a duplicate.
    """
    x, m, h, expected = glass_sphere_ic(eta=3.0)
    centre_idx = int(np.linalg.norm(x, axis=1).argmin())
    assert np.linalg.norm(x[centre_idx]) == 0.0, "expected a glass particle exactly at the centre"

    columns = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    assert not np.isnan(columns).any()
    centre_row = columns[centre_idx]
    # includes the centre particle's own self-column, ~0.1% of the total
    assert abs(centre_row.mean() / expected - 1) < 0.02, f"{centre_row.mean():.5f} vs {expected:.5f}"


def test_glass_sphere_converges_with_smoothing_length():
    """Smaller particle radii -> less mass smoothed beyond R -> closer to rho*R."""
    errs = []
    for eta in (4.0, 2.0):
        x, m, h, expected = glass_sphere_ic(eta=eta)
        tree = ConstructTree(x, m, h)
        single = np.array([ColumnDensityWalk_singleray(np.zeros(3), r, tree) for r in isotropic_rays(60)])
        errs.append(abs(single.mean() / expected - 1))
    assert errs[1] < errs[0], f"eta=2 error {errs[1]:.4f} should beat eta=4 error {errs[0]:.4f}"
    assert errs[1] < 0.005


# --------------------------------------------------------------------------------------------------
# Zero / degenerate particle radii.  A zero-radius sphere has zero geometric cross-section, so its
# contribution along any ray is zero (the h -> 0 limit is a delta function, and a ray hits it with
# probability zero).  Before the fix, ``1.0 / h_no`` was evaluated for *every* visited element
# before the leaf/node branch, so these inputs raised ZeroDivisionError in the serial walks and
# produced silent NaNs in the parallel ones.
# --------------------------------------------------------------------------------------------------


def _with_zero_radii(N=300, nzero=5, seed=37):
    x, m, h = cloud(N, seed=seed)
    h = h.copy()
    h[:nzero] = 0.0
    return x, m, h


@pytest.mark.parametrize("parallel", [False, True])
def test_zero_radius_raytrace(parallel):
    """Zero-radius sources must neither crash nor poison the result with NaNs."""
    x, m, h = _with_zero_radii()
    got = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=parallel)
    assert np.isfinite(got).all(), f"{np.isnan(got).sum()} NaN, {np.isinf(got).sum()} inf"
    want = column_reference(x, SIX_RAYS, x, m, h)
    assert rel_err(got, want) < RTOL


@pytest.mark.parametrize("parallel", [False, True])
def test_zero_radius_binned(parallel):
    x, m, h = _with_zero_radii()
    got = ColumnDensity(x, m, h, parallel=parallel)
    assert np.isfinite(got).all()


def test_zero_radius_multiray_walk():
    x, m, h = _with_zero_radii()
    tree = ConstructTree(x, m, h)
    for i in (0, 3, 150):
        assert np.isfinite(ColumnDensityWalk_multiray(x[i], SIX_RAYS, tree)).all()
        assert np.isfinite(ColumnDensityWalk_binned(x[i], tree)).all()


def test_all_radii_zero_gives_zero_column():
    """The degenerate limit: nothing has any cross-section, so nothing is obscured."""
    x, m, _ = cloud(100, seed=41)
    h = np.zeros(len(x))
    got = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    assert np.isfinite(got).all()
    assert np.abs(got).max() == 0.0


def test_zero_radius_does_not_perturb_other_particles():
    """Removing the cross-section of a few particles must leave the rest of the column intact."""
    x, m, h = cloud(300, seed=37)
    h_zero = h.copy()
    h_zero[:5] = 0.0
    full = ColumnDensity(x, m, h, rays=SIX_RAYS, parallel=True)
    holed = ColumnDensity(x, m, h_zero, rays=SIX_RAYS, parallel=True)
    want = column_reference(x, SIX_RAYS, x, m, h_zero)
    assert rel_err(holed, want) < RTOL
    # sanity: zeroing a radius should actually have changed something
    assert not np.allclose(full, holed)
