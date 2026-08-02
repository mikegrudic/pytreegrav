"""Treecode behaviour when particle positions are not unique.

Exactly-coincident particles are a real thing in practice (restarted sims, projected data,
particles sitting on a grid), and they interact badly with a spatial tree: the build must
subdivide until it separates them, which it cannot do, so node sizes collapse toward zero.

What this module pins down:

  * the build survives duplicates without hanging or losing mass (also covered from the
    build side in test_radix_build.py);
  * with softening, the *acceleration* is trustworthy -- a coincident source exerts a zero
    force vector, so dropping it changes nothing;
  * with softening, the *potential* is NOT exact: both the plain and grouped walks exclude
    self by position (``if r > 0``) rather than by index, so a genuinely distinct particle
    sitting on top of the target is silently dropped along with the target itself. Brute
    force keeps it (the r < h branch), so the two disagree by exactly one softened
    self-pair, m*|PotentialKernel(0, h)|. Marked xfail below rather than papered over with a
    loose tolerance;
  * without softening the treecode output is *garbage* (~1e28), which the library warns
    about. The warning is the contract, so the warning is what we test.
"""

import warnings

import numpy as np
import pytest

from pytreegrav import Accel, ConstructTree, Potential

N_BASE, N_DUP = 2000, 200
SOFT = 0.02


def duplicated_particles(softening):
    """N_BASE random points with the first N_DUP repeated exactly."""
    rng = np.random.default_rng(3)
    base = rng.random((N_BASE, 3))
    x = np.vstack([base, base[:N_DUP]])
    m = np.repeat(1.0 / len(x), len(x))
    h = np.repeat(softening, len(x))
    is_dup = np.zeros(len(x), bool)
    is_dup[:N_DUP] = True
    is_dup[N_BASE:] = True
    return x, m, h, is_dup


def test_build_survives_duplicates():
    """The build must terminate, keep every particle, and conserve mass."""
    x, m, h, _ = duplicated_particles(SOFT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ConstructTree(x, m, h)
    assert np.isclose(tree.Masses[tree.NumParticles], m.sum())
    assert np.array_equal(np.sort(tree.TreewalkIndices), np.arange(len(x)))


def test_softened_accel_matches_bruteforce():
    """Acceleration is unaffected by coincident sources -- they contribute a zero vector --
    so the tree must still agree with brute force to normal treecode accuracy."""
    x, m, h, _ = duplicated_particles(SOFT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a_tree = Accel(x, m, h, method="tree", parallel=True)
        a_bf = Accel(x, m, h, method="bruteforce", parallel=True)
    assert np.all(np.isfinite(a_tree))
    rel = np.sqrt(np.mean(np.sum((a_tree - a_bf) ** 2, axis=1))) / np.sqrt(np.mean(np.sum(a_bf**2, axis=1)))
    assert rel < 0.02


@pytest.mark.xfail(
    strict=True,
    reason="tree walks exclude self by position (r > 0), so a distinct particle coincident "
    "with the target is dropped; brute force keeps it via the r < h branch. The two differ "
    "by one softened self-pair, m*|PotentialKernel(0,h)|, on every duplicated particle.",
)
def test_softened_potential_matches_bruteforce():
    x, m, h, _ = duplicated_particles(SOFT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_tree = Potential(x, m, h, method="tree", parallel=True)
        p_bf = Potential(x, m, h, method="bruteforce", parallel=True)
    assert np.std(p_tree - p_bf) / np.std(p_bf) < 0.02


def test_softened_potential_discrepancy_is_one_self_pair():
    """Characterisation of the above: the disagreement is a clean offset on the duplicated
    particles of exactly one softened self-pair, not diffuse noise. If this ever changes,
    the xfail above needs revisiting."""
    from pytreegrav.kernel import PotentialKernel

    x, m, h, is_dup = duplicated_particles(SOFT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = Potential(x, m, h, method="tree", parallel=True) - Potential(x, m, h, method="bruteforce", parallel=True)
    expected = m[0] * abs(PotentialKernel(0.0, SOFT))
    assert np.mean(d[is_dup]) == pytest.approx(expected, rel=0.05)
    assert abs(np.mean(d[~is_dup])) < 0.02 * expected  # unique particles are unaffected


def test_unsoftened_duplicates_warn():
    """Without softening the answer is undefined for overlapping particles; the library's
    contract is that it warns, and the message must say so."""
    x, m, h, _ = duplicated_particles(0.0)
    with pytest.warns(UserWarning, match="non-unique") as rec:
        ConstructTree(x, m, h)
    assert any("singular" in str(w.message) or "garbage" in str(w.message) for w in rec)


def test_softened_duplicates_warn_differently():
    """With softening the warning should say softening determines the answer, not that the
    result is garbage."""
    x, m, h, _ = duplicated_particles(SOFT)
    with pytest.warns(UserWarning, match="non-unique") as rec:
        ConstructTree(x, m, h)
    assert any("Softening will" in str(w.message) for w in rec)


def test_unsoftened_bruteforce_stays_finite():
    """Brute force skips exactly-coincident pairs (r == 0) and so remains well behaved,
    unlike the tree -- worth pinning down, since it is the available fallback."""
    x, m, h, _ = duplicated_particles(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = Accel(x, m, h, method="bruteforce", parallel=True)
        p = Potential(x, m, h, method="bruteforce", parallel=True)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(p))
    assert np.abs(a).max() < 1e3  # sane magnitude for a unit-mass unit-cube system
