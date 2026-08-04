"""Field: tree built once, any subset of the fields from a single traversal.

Two things need pinning, and neither is about the physics -- that is covered by the per-quantity
suites already:

  * ``Field`` must reproduce the functional API *exactly*.  It is a caching wrapper, not a second
    implementation, so anything short of bit-identity means the permutation or the sorted arrays it
    holds have drifted from what the functions rebuild per call.
  * fusing must not change the answer.  One traversal serves every requested field, so a fused call
    and separate calls see the same accepted-node set and the same summation order -- this is what
    makes ``evaluate(potential=True, accel=True)`` a pure optimization.

    A *single*-field request is bit-identical, since it compiles to the same specialization.  A
    multi-field one is not, and that is expected rather than a defect: fusing hands LLVM a bigger
    basic block, so ``fastmath``'s FMA contraction fires differently -- ``K*dx`` becomes one
    fused multiply-add here and a separate multiply and add there.  Measured 0.6 ulp on the
    acceleration, 1.2 on the potential, 6.8 on the tidal tensor, all deterministic.  The fused
    value is not the worse of the two; an FMA rounds once where the split form rounds twice.

The single-field wrappers in grouped_treewalk are themselves thin calls into the fused core, so the
first property also guards the consolidation of the six hand-written kernels into one factory.
"""

import numpy as np
import pytest

from pytreegrav import Accel, Field, Potential, TidalTensor
from pytreegrav.grouped_treewalk import FieldsTarget_grouped


def cloud(n, seed=0, h=0.05):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.random(n) / n, np.full(n, h)


ALL_SUBSETS = [
    dict(potential=True),
    dict(accel=True),
    dict(tidal=True),
    dict(potential=True, accel=True),
    dict(potential=True, tidal=True),
    dict(accel=True, tidal=True),
    dict(potential=True, accel=True, tidal=True),
]

# Bound on fused-vs-separate disagreement, in ulps of the array's own scale.  Measured max 6.8
# (tidal, monopole); 64 leaves ~10x for a different compiler or FMA policy.  A real defect in the
# fused path moves this by orders of magnitude, not by a factor of ten -- see the module docstring.
FUSION_ULP = 64


def assert_agrees(got, ref, what):
    """Bit-identity where it must hold, an ulp bound where FMA contraction makes it not."""
    if np.array_equal(got, ref):
        return
    scale = np.abs(ref).max()
    ulps = np.abs(got - ref).max() / (scale * np.finfo(float).eps) if scale else 0.0
    assert ulps < FUSION_ULP, f"{what}: {ulps:.1f} ulp, over the {FUSION_ULP} ulp fusion bound"


@pytest.mark.parametrize("quadrupole", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_field_matches_functional_api(quadrupole, parallel):
    """Bit-identical to Potential/Accel/TidalTensor, which rebuild the tree and permutation."""
    x, m, h = cloud(3000, seed=1)
    kw = dict(method="tree", theta=0.5, quadrupole=quadrupole, parallel=parallel)
    f = Field(x, m, h, theta=0.5, quadrupole=quadrupole, parallel=parallel)
    assert np.array_equal(f.potential(), Potential(x, m, h, **kw))
    assert np.array_equal(f.accel(), Accel(x, m, h, **kw))
    assert np.array_equal(f.tidal(), TidalTensor(x, m, h, **kw))


@pytest.mark.parametrize("quadrupole", [False, True])
def test_fused_equals_separate(quadrupole):
    """Fusing is an optimization, not an approximation: every subset must reproduce the single-field
    answers, bit-identically when only one field is asked for and to within FUSION_ULP otherwise."""
    x, m, h = cloud(3000, seed=2)
    f = Field(x, m, h, theta=0.5, quadrupole=quadrupole, parallel=True)
    single = {"potential": f.potential(), "accel": f.accel(), "tidal": f.tidal()}
    for flags in ALL_SUBSETS:
        res = f.evaluate(**flags)
        assert set(res) == set(flags), f"{flags} returned {set(res)}"
        for k in flags:
            if len(flags) == 1:  # same specialization -> no excuse for any difference at all
                assert np.array_equal(res[k], single[k]), f"{k} alone differs from {k}()"
            else:
                assert_agrees(res[k], single[k], f"{k} fused as {sorted(flags)}")


@pytest.mark.parametrize("quadrupole", [False, True])
def test_fused_equals_separate_external_targets(quadrupole):
    """Same, at points that are not the sources -- exercises the Morton sort/unsort of targets."""
    x, m, h = cloud(2000, seed=3)
    tgt = np.random.default_rng(4).normal(size=(500, 3)) * 1.3
    ht = np.full(len(tgt), 0.02)
    f = Field(x, m, h, theta=0.5, quadrupole=quadrupole, parallel=True)
    single = {
        "potential": f.potential(tgt, softening_target=ht),
        "accel": f.accel(tgt, softening_target=ht),
        "tidal": f.tidal(tgt, softening_target=ht),
    }
    for flags in ALL_SUBSETS:
        res = f.evaluate(pos_target=tgt, softening_target=ht, **flags)
        for k in flags:
            if len(flags) == 1:
                assert np.array_equal(res[k], single[k]), f"{k} alone differs from {k}()"
            else:
                assert_agrees(res[k], single[k], f"{k} fused as {sorted(flags)}")


def test_external_targets_match_target_functions():
    """Field's external-target path must match the *Target functions, permutation included."""
    from pytreegrav import AccelTarget, PotentialTarget, TidalTensorTarget

    x, m, h = cloud(2000, seed=5)
    tgt = np.random.default_rng(6).normal(size=(400, 3)) * 1.3
    ht = np.full(len(tgt), 0.02)
    f = Field(x, m, h, theta=0.5, parallel=True)
    kw = dict(softening_target=ht, softening_source=h, method="tree", theta=0.5, parallel=True)
    assert np.array_equal(f.potential(tgt, softening_target=ht), PotentialTarget(tgt, x, m, **kw))
    assert np.array_equal(f.accel(tgt, softening_target=ht), AccelTarget(tgt, x, m, **kw))
    assert np.array_equal(f.tidal(tgt, softening_target=ht), TidalTensorTarget(tgt, x, m, **kw))


def test_repeated_evaluation_is_stable():
    """The cached permutation must not be consumed or mutated by a call.  Re-sorting already-sorted
    positions is the documented footgun in Accel (a non-involutive sigma giving X[sigma^2]); Field
    exists partly to make that unreachable, so evaluating twice must give the same answer."""
    x, m, h = cloud(2000, seed=7)
    f = Field(x, m, h, theta=0.5)
    first = f.evaluate(potential=True, accel=True, tidal=True)
    for _ in range(3):
        again = f.evaluate(potential=True, accel=True, tidal=True)
        for k in first:
            assert np.array_equal(first[k], again[k])


def test_per_call_overrides():
    """theta/parallel/group_size override the constructor defaults for that call only."""
    x, m, h = cloud(2000, seed=8)
    f = Field(x, m, h, theta=0.7, parallel=False, group_size=8)
    assert np.array_equal(f.potential(theta=0.4), Field(x, m, h, theta=0.4).potential())
    assert np.array_equal(f.accel(parallel=True), Field(x, m, h, theta=0.7, parallel=True).accel())
    assert np.array_equal(f.tidal(group_size=1), Field(x, m, h, theta=0.7, group_size=1).tidal())
    # and the defaults are untouched afterwards
    assert (f.theta, f.parallel, f.group_size) == (0.7, False, 8)


def test_requesting_nothing_raises():
    x, m, h = cloud(100, seed=9)
    with pytest.raises(ValueError):
        Field(x, m, h).evaluate()
    with pytest.raises(ValueError):
        FieldsTarget_grouped(x, h, Field(x, m, h).tree)


def test_quadrupole_field_needs_quadrupole_tree():
    """Field builds its own tree, so quadrupole=True must give it quadrupoles -- the mismatch is an
    unchecked out-of-bounds read in the walk, not an exception."""
    x, m, h = cloud(500, seed=10)
    assert Field(x, m, h, quadrupole=True).tree.HasQuads
    assert not Field(x, m, h, quadrupole=False).tree.HasQuads


def test_single_particle_and_repr():
    """N=1 has no self-interaction; also the github issue #31 scalar-collapse path."""
    f = Field([[0.0, 0.0, 0.0]], [1.0])
    res = f.evaluate(potential=True, accel=True, tidal=True)
    assert res["potential"].shape == (1,) and res["accel"].shape == (1, 3) and res["tidal"].shape == (1, 3, 3)
    assert not any(np.any(v) for v in res.values())
    assert "pytreegrav.Field" in repr(f) and "1 particles" in repr(f)


def test_G_scaling():
    x, m, h = cloud(500, seed=11)
    ref = Field(x, m, h).evaluate(potential=True, accel=True, tidal=True)
    got = Field(x, m, h, G=3.0).evaluate(potential=True, accel=True, tidal=True)
    for k in ref:
        assert np.allclose(got[k], 3.0 * ref[k], rtol=1e-13, atol=0)
