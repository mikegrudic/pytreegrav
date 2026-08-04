"""Tidal tensor T_ij = dg_i/dx_j = -d^2 phi/(dx_i dx_j).

Comparing the tree against brute force only shows the two share a kernel, so the tests that
actually pin the arithmetic are the ones with an independent reference:

  * ``tr(T) = -4 pi G rho`` -- zero outside every softening kernel, the summed spline density
    inside one.  Fixes the normalization and the sign convention at once, and constrains the
    relative coefficient between the dx_i dx_j and delta_ij terms, which is the easy thing to
    get wrong in the softened kernel.  Also traceless term-by-term for the quadrupole, which is
    what pins its diagonal coefficients.
  * central differences of ``AccelTarget``, tying the tensor to the acceleration the package
    already returns.
  * the exact multipole expansion of a symmetric two-body node, the only test sharp enough to
    constrain the quadrupole *off-diagonals*.

Each of those was checked to fail under a 0.1% perturbation of every coefficient it covers; the
docstrings record the margins, so read them before adjusting a tolerance.
"""

import numpy as np
import pytest

from pytreegrav import AccelTarget, ConstructTree, TidalTensor, TidalTensorTarget
from pytreegrav.kernel import ForceKernel, TidalKernel

# Machine-precision comparisons still need a hair of room: fastmath reassociates the sums, so
# the tree and bruteforce paths accumulate in different orders.
TIGHT = 1e-12


def random_cloud(n, seed=0, h=0.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.random(n) / n, np.full(n, h)


def spline_density(r, h, m):
    """Mass density of the M4 cubic spline of total mass m and compact support radius h."""
    q = r / h
    w = np.where(q <= 0.5, 1 - 6 * q**2 + 6 * q**3, np.where(q <= 1, 2 * (1 - q) ** 3, 0.0))
    return m * 8 / (np.pi * h**3) * w


def fd_tensor(pos_source, m, h, targets, eps):
    """T_ij = dg_i/dx_j by central differences of the brute-force acceleration."""
    out = np.zeros((len(targets), 3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        gp = AccelTarget(targets + d, pos_source, m, softening_source=h, method="bruteforce")
        gm = AccelTarget(targets - d, pos_source, m, softening_source=h, method="bruteforce")
        out[:, :, j] = (gp - gm) / (2 * eps)
    return out


def rms_rel(a, ref):
    return np.sqrt(np.mean((a - ref) ** 2)) / np.sqrt(np.mean(ref**2))


def test_point_mass_analytic():
    """A point mass M at distance r along z gives diag(-1, -1, 2) GM/r^3: stretched along the
    separation, compressed transverse to it.  Gets the sign convention exactly."""
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    m = np.array([1.0, 0.0])
    T = TidalTensor(x, m, method="bruteforce")[1]
    assert np.allclose(T, np.diag([-0.125, -0.125, 0.25]), rtol=TIGHT, atol=0)


@pytest.mark.parametrize("method,quadrupole", [("bruteforce", False), ("tree", False), ("tree", True)])
def test_traceless_without_softening(method, quadrupole):
    """Unsoftened point masses source no density at the evaluation point, so tr(T) = 0 exactly
    -- independent of theta, since every accepted element contributes a traceless tensor.

    The quadrupole=True case is the sharpest available check on the quadrupole coefficients:
    that term's trace vanishes only via -10 - (5/2)*3 + 35/2 = 0, so perturbing any one of the
    three breaks tracelessness outright rather than by some small truncation-sized amount.
    """
    x, m, h = random_cloud(400, seed=1)
    T = TidalTensor(x, m, h, method=method, theta=0.7, quadrupole=quadrupole)
    assert np.abs(np.trace(T, axis1=1, axis2=2)).max() / np.abs(T).max() < 1e-13


def summed_spline_density(x, m, h):
    """Spline density at each particle, from every particle but itself."""
    n = len(x)
    keep = ~np.eye(n, dtype=bool)
    return np.array(
        [np.sum(spline_density(np.linalg.norm(x - x[i], axis=1)[keep[i]], h[i], m[keep[i]])) for i in range(n)]
    )


def test_trace_is_minus_4pi_rho_overlapping():
    """tr(T) = -4 pi G rho, pinned where float64 can fully resolve it.

    Sizing matters twice over, so don't rescale this cloud casually:

      * particles must sit well inside each other's kernels, or the trace becomes a small
        residual of large traceless far-pair tensors and cancellation, not the implementation,
        sets the accuracy (that regime is covered by the _general test below);
      * but not *too* far inside -- the TidalKernel term enters the trace as Q r^2 against 3 K,
        so at q -> 0 it vanishes and the test stops constraining Q at all.  At this scale
        (median q = 0.52) it carries ~37% of the trace.

    Measured 2.3e-12, so 1e-10 has ~50x margin while still catching a 0.1% error in any
    TidalKernel coefficient by a factor of ~4e6.
    """
    n = 50
    x = np.random.default_rng(2).normal(size=(n, 3)) * 0.25
    m, h = np.full(n, 1.0 / n), np.full(n, 1.0)
    T = TidalTensor(x, m, h, method="bruteforce")
    rho = summed_spline_density(x, m, h)
    assert np.abs(np.trace(T, axis1=1, axis2=2) / (-4 * np.pi * rho) - 1).max() < 1e-10


def test_trace_is_minus_4pi_rho_general():
    """The same identity on a realistic cloud, where the trace is a small residual.

    Distant pairs each contribute an exactly traceless tensor, so their O(|T|) components must
    cancel in the trace; that cancellation, not the implementation, sets the achievable
    accuracy.  The tolerance is therefore an absolute one scaled by |T| -- the magnitude the
    trace is a difference of -- rather than relative to the trace's own small value.  Measured
    max absolute error 7.6e-13 against max|T| = 0.41.
    """
    n = 300
    x, m, h = random_cloud(n, seed=2, h=0.5)
    T = TidalTensor(x, m, h, method="bruteforce")
    rho = summed_spline_density(x, m, h)
    err = np.abs(np.trace(T, axis1=1, axis2=2) + 4 * np.pi * rho).max()
    assert err < 1e-10 * np.abs(T).max()


@pytest.mark.parametrize("h", [0.0, 0.6])
def test_matches_finite_difference_of_accel(h):
    """The tensor must be the gradient of the acceleration the package already returns."""
    pos, m, hs = random_cloud(60, seed=3, h=h)
    targets = np.random.default_rng(4).normal(size=(12, 3)) * 1.5
    T = TidalTensorTarget(targets, pos, m, softening_source=hs, method="bruteforce")
    T_fd = fd_tensor(pos, m, hs, targets, 1e-5)
    # 1e-6 is the central-difference truncation floor here, not the implementation's error
    assert np.abs(T - T_fd).max() / np.abs(T_fd).max() < 1e-6


@pytest.mark.parametrize("method", ["bruteforce", "tree"])
@pytest.mark.parametrize("quadrupole", [False, True])
def test_symmetric(method, quadrupole):
    """T is a Hessian, so it must come out symmetric -- and the kernels are written to make the
    off-diagonals bit-identical rather than merely close."""
    x, m, h = random_cloud(600, seed=5, h=0.1)
    kw = {"quadrupole": quadrupole} if method == "tree" else {}
    T = TidalTensor(x, m, h, method=method, **kw)
    assert np.array_equal(T, T.transpose(0, 2, 1))


@pytest.mark.parametrize("quadrupole", [False, True])
def test_tree_matches_bruteforce(quadrupole):
    x, m, h = random_cloud(3000, seed=6, h=0.05)
    T_bf = TidalTensor(x, m, h, method="bruteforce", parallel=True)
    T_tree = TidalTensor(x, m, h, method="tree", theta=0.2, quadrupole=quadrupole, parallel=True)
    assert rms_rel(T_tree, T_bf) < 1e-4


@pytest.mark.parametrize("roll", [0, 1, 2])
def test_quadrupole_term_matches_exact_expansion(roll):
    """Pins all six components of the quadrupole tidal term against an exactly-known node.

    Two *equal* masses straddling their centre of mass along a skew axis: equal masses make the
    configuration parity-symmetric so every odd multipole vanishes, leaving the octupole zero
    and the monopole+quadrupole expansion accurate to O((d/r)^4) -- i.e. the residual is
    O((d/r)^2) *relative to the quadrupole term*, measured 1.7e-6 at this d/r.  A skew axis puts
    most of that term in the off-diagonals, which is what makes this the only test constraining
    them; tree-vs-bruteforce on a random cloud is orders of magnitude too loose.

    Two things here are load-bearing, so don't simplify them away:

      * the masses must stay equal.  Unequal masses reintroduce the octupole and the residual
        degrades to O(d/r), no longer below a coefficient error worth catching.
      * the three ``roll`` orientations are cyclic permutations of the same geometry.  How much
        of a given off-diagonal comes from its Q_ij/r^5 piece as opposed to the two contracted
        pieces is strongly orientation-dependent (8% for xy at roll=0, 331% for yz), so a single
        orientation leaves some individual coefficients effectively untested.
    """
    axis = np.roll(np.array([0.3, 0.5, 0.8]), roll)
    axis /= np.linalg.norm(axis)
    d = 0.001  # target sits at |r| = 1, so this is d/r
    x = np.array([axis * d, -axis * d])
    m = np.array([0.5, 0.5])
    tgt = np.roll(np.array([[0.6, -0.4, 0.5]]), roll, axis=1)
    tgt = tgt / np.linalg.norm(tgt)

    tree = ConstructTree(x, m, np.zeros(2), quadrupole=True)
    kw = dict(tree=tree, method="tree", theta=0.7)  # theta large enough that the root is accepted
    exact = TidalTensorTarget(tgt, x, m, method="bruteforce")
    mono = TidalTensorTarget(tgt, None, None, quadrupole=False, **kw)
    quad = TidalTensorTarget(tgt, None, None, quadrupole=True, **kw)

    quad_term = np.abs(quad - mono).max()
    # the quadrupole must be what closes the monopole's error, not a small perturbation on it
    assert np.abs(mono - exact).max() > 0.5 * quad_term
    # ...and adding it must leave only the O((d/r)^2) tail.  1e-5 sits ~6x above the true
    # residual and ~5x below the smallest single-coefficient error this geometry can resolve.
    assert np.abs(quad - exact).max() < 1e-5 * quad_term


def test_theta_convergence_and_quadrupole_helps():
    """Error must fall with theta, and the quadrupole must beat the monopole at fixed theta."""
    x, m, h = random_cloud(3000, seed=7, h=0.05)
    T_bf = TidalTensor(x, m, h, method="bruteforce", parallel=True)
    err = {}
    for quad in (False, True):
        for theta in (0.7, 0.4):
            T = TidalTensor(x, m, h, method="tree", theta=theta, quadrupole=quad, parallel=True)
            err[quad, theta] = rms_rel(T, T_bf)
    assert err[False, 0.4] < err[False, 0.7]
    assert err[True, 0.4] < err[True, 0.7]
    assert err[True, 0.7] < err[False, 0.7]
    assert err[True, 0.4] < err[False, 0.4]


def test_parallel_matches_serial():
    """Groups accumulate into private buffers, so threading must not change a single bit."""
    x, m, h = random_cloud(3000, seed=8, h=0.05)
    kw = dict(method="tree", theta=0.5, group_size=8)
    assert np.array_equal(TidalTensor(x, m, h, **kw), TidalTensor(x, m, h, parallel=True, **kw))


def test_target_form_matches_mutual_form():
    """TidalTensorTarget with the sources as its own targets must reproduce TidalTensor, on both
    paths: same r==0 skip, same softening symmetrization."""
    x, m, h = random_cloud(2000, seed=9, h=0.05)
    for method in ("bruteforce", "tree"):
        T = TidalTensor(x, m, h, method=method, theta=0.5)
        T_t = TidalTensorTarget(x, x, m, softening_target=h, softening_source=h, method=method, theta=0.5)
        assert np.abs(T - T_t).max() / np.abs(T).max() < TIGHT


def test_group_size_1_is_the_per_target_walk():
    """group_size only changes which nodes open (a superset), so at theta small enough that the
    truncation error is negligible the two must agree closely."""
    x, m, h = random_cloud(3000, seed=10, h=0.05)
    kw = dict(method="tree", theta=0.2)
    assert rms_rel(TidalTensor(x, m, h, group_size=1, **kw), TidalTensor(x, m, h, group_size=8, **kw)) < 1e-4


def test_kernel_continuous_at_breakpoints():
    """TidalKernel is piecewise over q <= 1/2, q <= 1, q > 1; a wrong coefficient in any branch
    shows up as a jump at a breakpoint.  Checked independently of any tree or sum."""
    for r0 in (0.5, 1.0):
        lo = TidalKernel(r0 - 1e-9, 1.0)
        hi = TidalKernel(r0 + 1e-9, 1.0)
        assert abs(hi - lo) / abs(lo) < 1e-7
    # and it must join the unsoftened limit at q = 1
    assert abs(TidalKernel(1.0, 1.0) - 3.0) < 1e-12
    # finite at the centre, unlike 3/r^5
    assert np.isfinite(TidalKernel(0.0, 1.0))


def test_softened_kernel_is_minus_dK_dr_over_r():
    """TidalKernel(r,h) == -(1/r) d/dr ForceKernel(r,h), the relation the tensor is built on."""
    h = 1.0
    for r in (0.1, 0.3, 0.49, 0.6, 0.8, 0.99, 1.5):
        eps = 1e-6
        deriv = (ForceKernel(r + eps, h) - ForceKernel(r - eps, h)) / (2 * eps)
        assert abs(TidalKernel(r, h) + deriv / r) / abs(TidalKernel(r, h)) < 1e-6


def test_tree_reuse_and_quadrupole_guard():
    """A supplied tree is honoured, and asking for quadrupoles on a monopole tree raises rather
    than reading out of bounds (which numba does not check -- it segfaults)."""
    x, m, h = random_cloud(2000, seed=11, h=0.05)
    tree = ConstructTree(x, m, h, quadrupole=True)
    T_ref = TidalTensor(x, m, h, method="tree", theta=0.5, quadrupole=True)
    assert (
        np.abs(TidalTensor(x, m, h, tree=tree, theta=0.5, quadrupole=True) - T_ref).max() / np.abs(T_ref).max() < TIGHT
    )

    mono_tree = ConstructTree(x, m, h, quadrupole=False)
    with pytest.raises(ValueError):
        TidalTensor(x, m, h, tree=mono_tree, quadrupole=True)


@pytest.mark.parametrize("method", ["bruteforce", "tree"])
def test_single_particle(method):
    """N=1 has no self-interaction, so the tensor is zero -- and the scalar-collapse trap from
    github issue #31 must not bite the new entry points either."""
    T = TidalTensor([[0.0, 0.0, 0.0]], [1.0], method=method)
    assert T.shape == (1, 3, 3)
    assert np.all(T == 0)


def test_invalid_method_raises():
    x, m, h = random_cloud(10, seed=12)
    with pytest.raises(ValueError):
        TidalTensor(x, m, h, method="nonsense")
    with pytest.raises(ValueError):
        TidalTensorTarget(x, x, m, method="nonsense")


def test_scales_with_G_and_mass():
    """T is linear in G and in the source masses."""
    x, m, h = random_cloud(500, seed=13, h=0.1)
    T = TidalTensor(x, m, h, method="bruteforce")
    assert np.allclose(TidalTensor(x, m, h, G=3.0, method="bruteforce"), 3.0 * T, rtol=TIGHT, atol=0)
    assert np.allclose(TidalTensor(x, 2 * m, h, method="bruteforce"), 2.0 * T, rtol=TIGHT, atol=0)
