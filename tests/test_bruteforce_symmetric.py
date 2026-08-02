"""The symmetrized parallel brute-force kernels split the upper triangle across threads and
accumulate into per-thread buffers, so they must be checked against the serial
upper-triangular reference -- the tree-vs-bruteforce tests are far too loose to catch a
row-partitioning or reduction bug.
"""

import numpy as np
import pytest
from numba import set_num_threads, get_num_threads
from pytreegrav import Potential, Accel
from pytreegrav.bruteforce import Potential_bruteforce, Accel_bruteforce
from pytreegrav.bruteforce_symmetric import Potential_bruteforce_symmetric, Accel_bruteforce_symmetric, SYMMETRIC_NMIN

RTOL = 1e-12  # both sides do the same flops in a different order; only summation order differs


def random_particles(n, softened, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 3))
    m = rng.random(n) / n
    h = rng.random(n) * 0.05 if softened else np.zeros(n)
    return x, m, h


def rel_err(a, ref):
    scale = np.max(np.abs(ref))
    return np.max(np.abs(a - ref)) / scale if scale > 0 else np.max(np.abs(a - ref))


@pytest.mark.parametrize("n", [1, 2, 3, 17, 64, 65, 200, 1000])
@pytest.mark.parametrize("softened", [False, True])
def test_symmetric_matches_serial(n, softened):
    """Symmetrized result must reproduce the serial upper-triangular sum for any N."""
    x, m, h = random_particles(n, softened)
    assert rel_err(Potential_bruteforce_symmetric(x, m, h), Potential_bruteforce(x, m, h)) < RTOL
    assert rel_err(Accel_bruteforce_symmetric(x, m, h), Accel_bruteforce(x, m, h)) < RTOL


def test_thread_count_independence():
    """Each thread owns a private accumulator and an interleaved subset of the rows; changing
    the thread count changes both the work split and which rows land together, but must not
    change the result."""
    x, m, h = random_particles(500, softened=True)
    nthreads = get_num_threads()
    try:
        set_num_threads(1)
        phi1, a1 = Potential_bruteforce_symmetric(x, m, h), Accel_bruteforce_symmetric(x, m, h)
        set_num_threads(max(2, nthreads))
        phiN, aN = Potential_bruteforce_symmetric(x, m, h), Accel_bruteforce_symmetric(x, m, h)
    finally:
        set_num_threads(nthreads)
    assert rel_err(phiN, phi1) < RTOL
    assert rel_err(aN, a1) < RTOL


def test_momentum_conservation():
    """Symmetrized pairwise forces make sum(m*a) vanish to roundoff, not just to truncation."""
    x, m, h = random_particles(500, softened=True)
    a = Accel_bruteforce_symmetric(x, m, h)
    net = np.abs((m[:, None] * a).sum(axis=0)).max()
    assert net < 1e-14 * np.abs(m[:, None] * a).sum()


def test_coincident_particles():
    """Overlapping unsoftened particles contribute no self-force and must not produce NaN/inf."""
    x = np.zeros((4, 3))
    m = np.ones(4)
    h = np.zeros(4)
    phi = Potential_bruteforce_symmetric(x, m, h)
    a = Accel_bruteforce_symmetric(x, m, h)
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(a))
    assert rel_err(phi, Potential_bruteforce(x, m, h)) < RTOL
    assert rel_err(a, Accel_bruteforce(x, m, h)) < RTOL


@pytest.mark.parametrize("n", [SYMMETRIC_NMIN - 1, SYMMETRIC_NMIN])
def test_frontend_gate_agrees_on_both_sides(n):
    """The frontend dispatches to the simple kernels below SYMMETRIC_NMIN and the symmetrized ones at
    or above it; both branches must give the serial answer, so the threshold is purely a
    performance knob and can be retuned without changing results."""
    x, m, h = random_particles(n, softened=True)
    assert rel_err(Potential(x, m, h, method="bruteforce", parallel=True), Potential_bruteforce(x, m, h)) < RTOL
    assert rel_err(Accel(x, m, h, method="bruteforce", parallel=True), Accel_bruteforce(x, m, h)) < RTOL


@pytest.mark.parametrize("n,expect_sym", [(SYMMETRIC_NMIN - 1, False), (SYMMETRIC_NMIN, True)])
def test_frontend_gate_dispatches(n, expect_sym, monkeypatch):
    """...and the gate must actually route where we think it does, or the threshold is
    silently dead code."""
    import pytreegrav.frontend as fe

    called = []
    for name in (
        "Potential_bruteforce_symmetric",
        "Accel_bruteforce_symmetric",
        "Potential_bruteforce_parallel",
        "Accel_bruteforce_parallel",
    ):
        real = getattr(fe, name)
        monkeypatch.setattr(fe, name, (lambda nm, f: lambda *a, **k: (called.append(nm), f(*a, **k))[1])(name, real))

    x, m, h = random_particles(n, softened=True)
    Potential(x, m, h, method="bruteforce", parallel=True)
    Accel(x, m, h, method="bruteforce", parallel=True)
    assert called == (
        ["Potential_bruteforce_symmetric", "Accel_bruteforce_symmetric"]
        if expect_sym
        else ["Potential_bruteforce_parallel", "Accel_bruteforce_parallel"]
    )


def test_tree_does_not_leak_parallel_chunksize():
    """numba's parallel chunksize is PROCESS-GLOBAL.  The tree walks set it (64 in the grouped
    walk, 10000 elsewhere); if that isn't scoped it survives the call and silently degrades
    every later prange -- in this library and in user code.  Measured: one tree call made the
    brute-force kernels 2x slower (2.55 -> 5.12 ms at N=4000, 32 threads) until reset.
    """
    from numba import get_parallel_chunksize

    x, m, h = random_particles(2000, softened=False)
    before = get_parallel_chunksize()
    for kw in ({}, {"quadrupole": True}):
        Accel(x, m, h, method="tree", parallel=True, **kw)
        assert get_parallel_chunksize() == before, "Accel tree walk leaked the chunksize"
        Potential(x, m, h, method="tree", parallel=True, **kw)
        assert get_parallel_chunksize() == before, "Potential tree walk leaked the chunksize"


def test_target_parallel_variants_are_actually_parallel():
    """The *Target_bruteforce serial/parallel pairs are njit'd from the SAME Python function,
    so they share one on-disk cache entry (keyed by qualname+lineno).  Adding cache=True to
    them makes the parallel dispatcher silently load the serial code -- 8.4x -> 1.05x, no
    warning, correct answers.

    Which variant gets corrupted depends on which is *called* first, so exercise the serial
    one first (the realistic order) and assert both directions: serial must stay serial and
    parallel must stay parallel, measured as CPU time over wall time.
    """
    import time

    from pytreegrav.bruteforce import (
        AccelTarget_bruteforce,
        AccelTarget_bruteforce_parallel,
        PotentialTarget_bruteforce,
        PotentialTarget_bruteforce_parallel,
    )

    if get_num_threads() < 2:
        pytest.skip("needs >1 thread to distinguish parallel from serial")

    x, m, h = random_particles(3000, softened=False)

    def cpu_per_wall(f):
        f(x, h, x, m, h)  # compile or load from cache
        w0, c0 = time.perf_counter(), time.process_time()
        f(x, h, x, m, h)
        return (time.process_time() - c0) / (time.perf_counter() - w0)

    for serial, par in (
        (AccelTarget_bruteforce, AccelTarget_bruteforce_parallel),
        (PotentialTarget_bruteforce, PotentialTarget_bruteforce_parallel),
    ):
        r_ser = cpu_per_wall(serial)  # first, so a shared cache entry would be the serial one
        r_par = cpu_per_wall(par)
        assert r_ser < 1.5, f"{serial.__name__} ran in parallel (cpu/wall={r_ser:.2f}) -- cache collision?"
        assert r_par > 1.5, f"{par.__name__} ran serially (cpu/wall={r_par:.2f}) -- cache collision?"


def test_G_scaling():
    x, m, h = random_particles(200, softened=True)
    assert rel_err(Potential_bruteforce_symmetric(x, m, h, 3.0), 3.0 * Potential_bruteforce_symmetric(x, m, h)) < RTOL
    assert rel_err(Accel_bruteforce_symmetric(x, m, h, 3.0), 3.0 * Accel_bruteforce_symmetric(x, m, h)) < RTOL
