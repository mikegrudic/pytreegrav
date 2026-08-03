"""Tests for the column-density estimators.

The ray-traced walks are *exact*, not approximate: a node is opened whenever its bounding sphere
(inflated by the node's max particle radius) could intersect the ray, and only leaf particles ever
contribute.  So the tree result must agree with direct summation to machine precision -- unlike the
gravity walks, there is no opening-angle truncation error to absorb.  That makes tree-vs-direct a
very sharp test, and it is the main workhorse here.

The underlying physics is one closed form.  A particle of mass M and radius h is a uniform sphere of
density rho = 3M/(4 pi h^3).  A ray from ``pos`` in direction ``n`` (a unit vector) sees the sphere
at separation ``d = x_p - pos``, with along-ray coordinate ``z = d.n`` and impact parameter
``b = sqrt(|d|^2 - z^2)``.  The ray misses unless ``b < h``.  The full chord is
``2 sqrt(h^2 - b^2)``, so

    column = rho * 2 sqrt(h^2 - b^2) = 3M/(4 pi h^2) * 2 sqrt(1 - (b/h)^2)      [origin outside]

and when the origin lies *inside* the sphere the ray only traverses the forward half-chord,

    column = rho * (z + sqrt(h^2 - b^2)) = 3M/(4 pi h^2) * (z/h + sqrt(1 - (b/h)^2))
"""

import os

import numpy as np
import pytest

from pytreegrav import ColumnDensity, ConstructTree
from pytreegrav.treewalk import (
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
    """Two well-separated clumps: deep trees, long empty stretches along rays, and -- deliberately --
    a large separation-to-radius ratio, which is where the impact parameter becomes ill-conditioned.

    Both the walk and the reference compute ``b^2 = r^2 - z^2``.  For a nearly radial ray at large
    distance, r ~ z, so that subtraction cancels: at separation/radius = 200 it loses ~4-5 digits,
    and ``sqrt(1 - q^2)`` then amplifies the loss for rays grazing a sphere's edge (its derivative
    diverges as the chord goes to zero).  The two implementations lose the same digits and differ
    only in rounding, so this is a conditioning property of the formulation, not a defect -- and it
    is harmless, because the amplification is largest exactly where the contribution is smallest.
    Hence 1e-9 here rather than the 1e-12 used everywhere else.

    The cancellation-free form ``b = |d - (d.n) n|`` removes it entirely but costs ~13% on the hot
    loop, which is not worth digits 12-16 of an extinction estimate built on uniform spheres.
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
# Glass sphere: the column from the centre of a uniform sphere out to infinity is exactly rho*R in
# every direction.  This is the end-to-end physics check -- it exercises the geometry, the mass
# normalization of the uniform-sphere kernel, and the isotropy of the traversal all at once, against
# a closed form that does not depend on the implementation.
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
