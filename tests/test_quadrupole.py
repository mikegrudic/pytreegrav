"""Node quadrupole moments, and the accuracy they buy.

Regression guard for a bug where ComputeMoments accumulated every child's quadrupole using
`mi` left over from the *previous* loop -- i.e. the last child's mass applied to all children.
The centres of mass stayed exact, so nothing crashed and no test failed; the only symptom was
that quadrupole=True was *less* accurate than the monopole walk for theta >= 0.5, which is
backwards for a higher-order correction.
"""

import numpy as np
import pytest

from pytreegrav import Accel, ConstructTree, Potential


def analytic_quadrupole(x, m, com):
    """Q_kl = sum_i m_i (3 r_k r_l - delta_kl r^2) about com -- the reduced (traceless)
    convention the tree's recursion is built on."""
    r = x - com
    Q = 3.0 * np.einsum("i,ik,il->kl", m, r, r)
    Q -= np.eye(3) * np.sum(m * np.sum(r**2, axis=1))
    return Q


def plummer(n, seed=42):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    r = np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))
    d = rng.normal(size=(n, 3))
    return np.ascontiguousarray((d.T * r / np.sum(d**2, axis=1) ** 0.5).T), np.repeat(1.0 / n, n)


@pytest.mark.parametrize("n", [8, 50, 500, 5000])
def test_root_quadrupole_matches_analytic(n):
    """The root node's quadrupole must equal a direct sum over all particles about its COM.
    This is the sharpest form of the test: it is exact up to accumulation roundoff, and the
    buggy version was wrong by 30-160%."""
    rng = np.random.default_rng(0)
    x, m = rng.random((n, 3)), rng.random(n)
    tree = ConstructTree(x, m, np.zeros(n), quadrupole=True)
    root = tree.NumParticles
    Q = np.array(tree.Quadrupoles[root])
    Q_ref = analytic_quadrupole(x, m, np.array(tree.Coordinates[root]))
    assert np.max(np.abs(Q - Q_ref)) / np.max(np.abs(Q_ref)) < 1e-10


def test_dynamic_tree_root_quadrupole_matches_analytic():
    """DynamicOctree carries its own copy of the moment recursion -- and the same bug."""
    n = 2000
    rng = np.random.default_rng(1)
    x, m, v = rng.random((n, 3)), rng.random(n), rng.normal(size=(n, 3))
    tree = ConstructTree(x, m, np.zeros(n), vel=v, quadrupole=True)
    root = tree.NumParticles
    Q = np.array(tree.Quadrupoles[root])
    Q_ref = analytic_quadrupole(x, m, np.array(tree.Coordinates[root]))
    assert np.max(np.abs(Q - Q_ref)) / np.max(np.abs(Q_ref)) < 1e-10


@pytest.mark.parametrize("theta", [0.4, 0.7])
def test_quadrupole_beats_monopole(theta):
    """A higher-order correction must improve accuracy at fixed opening angle.  With the bug
    this held only for theta < 0.5 and inverted above it."""
    n = 10000
    pos, m = plummer(n)
    h = np.zeros(n)
    a_exact = Accel(pos, m, h, method="bruteforce", parallel=True)
    phi_exact = Potential(pos, m, h, method="bruteforce", parallel=True)

    def rms(a, ref):
        return np.sqrt(np.mean(np.sum((a - ref) ** 2, axis=-1)))

    a_mono = Accel(pos, m, h, method="tree", parallel=True, theta=theta, quadrupole=False)
    a_quad = Accel(pos, m, h, method="tree", parallel=True, theta=theta, quadrupole=True)
    assert rms(a_quad, a_exact) < rms(a_mono, a_exact)

    p_mono = Potential(pos, m, h, method="tree", parallel=True, theta=theta, quadrupole=False)
    p_quad = Potential(pos, m, h, method="tree", parallel=True, theta=theta, quadrupole=True)
    assert np.std(p_quad - phi_exact) < np.std(p_mono - phi_exact)
