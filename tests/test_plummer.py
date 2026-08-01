"""Standard Plummer-sphere test: the tree solver (grouped walk, the default) must reproduce the
brute-force field to the expected Barnes-Hut accuracy, and the mean field must match the analytic
Plummer profile.  Uses the repo/JOSS convention: M=1, a=1, G=1.
"""

import numpy as np
from pytreegrav import Accel, Potential


def plummer(n, seed=42):
    """Sample n particles from a Plummer sphere (M=1, a=1), repo inverse-CDF convention."""
    np.random.seed(seed)
    u = np.random.rand(n)
    r = np.sqrt(u ** (2.0 / 3) * (1 + u ** (2.0 / 3) + u ** (4.0 / 3)) / (1 - u**2))
    d = np.random.normal(size=(n, 3))
    pos = (d.T * r / np.sum(d**2, axis=1) ** 0.5).T
    m = np.repeat(1.0 / n, n)
    return np.float64(pos), np.float64(m), np.float64(r)


def _rms_rel(a, ref):
    return np.sqrt(np.mean(np.sum((a - ref) ** 2, axis=-1 if a.ndim > 1 else 0))) / np.sqrt(
        np.mean(np.sum(ref**2, axis=-1 if ref.ndim > 1 else 0))
    )


def test_plummer_vs_bruteforce():
    """Tree accel & potential (grouped default) must match brute force to ~Barnes-Hut accuracy."""
    pos, m, r = plummer(30000)
    h = np.zeros_like(m)

    a_tree = Accel(pos, m, h, method="tree", parallel=True)
    a_bf = Accel(pos, m, h, method="bruteforce", parallel=True)
    phi_tree = Potential(pos, m, h, method="tree", parallel=True)
    phi_bf = Potential(pos, m, h, method="bruteforce", parallel=True)

    assert _rms_rel(a_tree, a_bf) < 0.02
    assert np.std(phi_tree - phi_bf) / np.std(phi_bf) < 0.02


def test_plummer_quadrupole_vs_bruteforce():
    """The quadrupole tree walk should be at least as accurate as the monopole one."""
    pos, m, r = plummer(30000)
    h = np.zeros_like(m)

    a_tree = Accel(pos, m, h, method="tree", parallel=True, quadrupole=True)
    a_bf = Accel(pos, m, h, method="bruteforce", parallel=True)
    phi_tree = Potential(pos, m, h, method="tree", parallel=True, quadrupole=True)
    phi_bf = Potential(pos, m, h, method="bruteforce", parallel=True)

    assert _rms_rel(a_tree, a_bf) < 0.02
    assert np.std(phi_tree - phi_bf) / np.std(phi_bf) < 0.02


def test_plummer_vs_analytic():
    """The tree field must match the analytic Plummer profile: phi(r) = -(1+r^2)^-1/2,
    g(r) = r/(1+r^2)^3/2, checked as a median ratio over a well-sampled radial band."""
    pos, m, r = plummer(30000)
    h = np.zeros_like(m)

    phi_tree = Potential(pos, m, h, method="tree", parallel=True)
    a_tree = Accel(pos, m, h, method="tree", parallel=True)

    band = (r > 0.2) & (r < 2.0)  # avoid center scatter and sparse outskirts
    phi_exact = -((1 + r**2) ** -0.5)
    assert abs(np.median(phi_tree[band] / phi_exact[band]) - 1) < 0.03

    g_radial = -np.sum(a_tree * pos, axis=1) / r  # inward radial acceleration
    g_exact = r / (1 + r**2) ** 1.5
    assert abs(np.median(g_radial[band] / g_exact[band]) - 1) < 0.05


def test_plummer_group_size_consistency():
    """group_size=1 (per-particle) and the default grouped walk must both be accurate."""
    pos, m, r = plummer(20000)
    h = np.zeros_like(m)
    a_bf = Accel(pos, m, h, method="bruteforce", parallel=True)

    for gs in (1, 8, 16):
        a_tree = Accel(pos, m, h, method="tree", parallel=True, group_size=gs)
        assert _rms_rel(a_tree, a_bf) < 0.02
