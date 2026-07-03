# tests/test_perturbation.py
import numpy as np
import pytest

from credit_assurance import perturbation as p

# Toy encoded space: col 0 = numeric "num"; cols 1-3 = a 3-level one-hot "cat".
FG = {"num": (0,), "cat": (1, 2, 3)}


def _donor_pool(n=60, seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)
    cat = np.zeros((n, 3))
    cat[np.arange(n), rng.integers(0, 3, size=n)] = 1.0
    return np.column_stack([num, cat])


X = _donor_pool()
X0 = np.array([2.0, 1.0, 0.0, 0.0])  # cat = category 0


def test_conditional_erases_group_atomically_and_keeps_valid_one_hot():
    perturb = p.make_perturber(X, FG, regime="conditional", seed=0, k_neighbors=10)
    pr = perturb(X0, ["cat"], m=8)
    assert pr.X.shape == (8, 4)
    assert np.allclose(pr.X[:, 0], 2.0)                 # untouched "num" fixed
    cat = pr.X[:, 1:4]
    assert np.all(np.isin(cat, [0.0, 1.0]))             # dummies stay 0/1
    assert np.all(cat.sum(axis=1) == 1)                 # exactly one hot per row -> no invalid pattern
    assert isinstance(pr.ood_distance, float) and pr.ood_distance >= 0.0


def test_conditional_multi_group_comes_from_one_donor():
    # joint-preservation: co-erased groups must come from the SAME donor row
    rng = np.random.default_rng(1)
    cat = np.zeros((40, 3))
    cat[np.arange(40), rng.integers(0, 3, 40)] = 1.0
    donor = np.column_stack([rng.normal(size=40), rng.normal(size=40), cat])   # cols: a,b,cat(2,3,4)
    fg = {"a": (0,), "b": (1,), "cat": (2, 3, 4)}
    perturb = p.make_perturber(donor, fg, regime="conditional", seed=0, k_neighbors=15)
    x = np.array([5.0, -5.0, 1.0, 0.0, 0.0])
    pr = perturb(x, ["a", "cat"], m=6)                  # erase a + cat; keep b
    erased = [0, 2, 3, 4]
    for row in pr.X:
        assert np.any(np.all(np.isclose(donor[:, erased], row[erased]), axis=1))  # matches one donor
    assert np.allclose(pr.X[:, 1], -5.0)                # kept "b" untouched


def test_baseline_is_deterministic_modal_category():
    perturb = p.make_perturber(X, FG, regime="baseline")
    pr = perturb(X0, ["cat"], m=3)
    assert np.all(pr.X == pr.X[0])                      # fixed baseline -> identical rows
    assert pr.X[0, 1:4].sum() == 1 and pr.X[0, 0] == 2.0


def test_marginal_draws_group_from_donor_pool():
    perturb = p.make_perturber(X, FG, regime="marginal", seed=1)
    pr = perturb(X0, ["num"], m=10)
    assert np.allclose(pr.X[:, 1:4], np.array([1.0, 0.0, 0.0]))   # "cat" untouched
    assert set(np.round(pr.X[:, 0], 6)).issubset(set(np.round(X[:, 0], 6)))


def test_perturb_is_reproducible_per_call_regardless_of_order():
    perturb = p.make_perturber(X, FG, regime="conditional", seed=7, k_neighbors=10)
    a = perturb(X0, ["cat"], m=5).X
    _ = perturb(np.array([0.5, 0.0, 1.0, 0.0]), ["num"], m=3)     # an interleaved, different call
    b = perturb(X0, ["cat"], m=5).X
    assert np.array_equal(a, b)                          # identical args -> identical draws


def test_empty_group_ids_returns_x_with_real_ood():
    perturb = p.make_perturber(X, FG, regime="conditional")
    pr = perturb(X0, [], m=5)
    assert pr.X.shape == (5, 4) and np.allclose(pr.X, X0)
    assert pr.ood_distance > 0.0                         # X0 is not a donor row -> strictly positive


def test_marginal_output_is_permutation_invariant():
    rng = np.random.default_rng(2)
    cat = np.zeros((30, 3))
    cat[np.arange(30), rng.integers(0, 3, 30)] = 1.0
    donor = np.column_stack([rng.normal(size=30), rng.normal(size=30), cat])
    fg = {"a": (0,), "b": (1,), "cat": (2, 3, 4)}
    pert = p.make_perturber(donor, fg, regime="marginal", seed=3)
    x = np.array([1.0, 2.0, 1.0, 0.0, 0.0])
    assert np.array_equal(pert(x, ["a", "cat"], m=6).X, pert(x, ["cat", "a"], m=6).X)


def test_zero_variance_column_is_handled():
    rng = np.random.default_rng(4)
    n = 30
    cat = np.zeros((n, 3))
    cat[np.arange(n), rng.integers(0, 3, n)] = 1.0
    donor = np.column_stack([rng.normal(size=n), np.ones(n), cat])   # col 1 is constant
    fg = {"num": (0,), "const": (1,), "cat": (2, 3, 4)}
    pert = p.make_perturber(donor, fg, regime="conditional", seed=0, k_neighbors=8)
    pr = pert(np.array([0.5, 1.0, 1.0, 0.0, 0.0]), ["num"], m=5)
    assert pr.X.shape == (5, 5) and np.isfinite(pr.ood_distance)     # no NaN from the constant column


def test_non_finite_x_rejected():
    pert = p.make_perturber(X, FG)
    with pytest.raises(ValueError):
        pert(np.array([np.nan, 1.0, 0.0, 0.0]), ["cat"], m=3)


def test_k_neighbors_larger_than_pool_is_clamped():
    perturb = p.make_perturber(X, FG, regime="conditional", k_neighbors=10_000)
    assert perturb(X0, ["cat"], m=4).X.shape == (4, 4)


def test_m_below_one_raises():
    perturb = p.make_perturber(X, FG)
    with pytest.raises(ValueError):
        perturb(X0, ["cat"], m=0)


def test_invalid_feature_groups_rejected():
    with pytest.raises(ValueError):
        p.make_perturber(X, {"num": (0,), "cat": (1, 2)})     # column 3 uncovered
    with pytest.raises(ValueError):
        p.make_perturber(X, {"a": (0, 1), "b": (1, 2, 3)})    # column 1 overlaps


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        p.make_perturber(X, FG, regime="bogus")
