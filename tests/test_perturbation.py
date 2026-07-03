# tests/test_perturbation.py
import numpy as np
import pytest

from credit_assurance import perturbation as p

# Toy encoded space: col 0 = numeric singleton "num"; cols 1-3 = a 3-level one-hot "cat".
FG = {"num": (0,), "cat": (1, 2, 3)}


def _donor_pool(n=60, seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)
    cat_idx = rng.integers(0, 3, size=n)
    cat = np.zeros((n, 3))
    cat[np.arange(n), cat_idx] = 1.0
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


def test_baseline_is_deterministic_modal_category():
    perturb = p.make_perturber(X, FG, regime="baseline")
    pr = perturb(X0, ["cat"], m=3)
    assert np.all(pr.X == pr.X[0])                      # fixed baseline -> identical rows
    assert pr.X[0, 1:4].sum() == 1                      # a real (valid) category
    assert pr.X[0, 0] == 2.0                            # "num" untouched


def test_marginal_draws_group_from_donor_pool():
    perturb = p.make_perturber(X, FG, regime="marginal", seed=1)
    pr = perturb(X0, ["num"], m=10)
    assert np.allclose(pr.X[:, 1:4], np.array([1.0, 0.0, 0.0]))   # "cat" untouched
    donor_nums = set(np.round(X[:, 0], 6))
    assert set(np.round(pr.X[:, 0], 6)).issubset(donor_nums)      # "num" from donor marginal


def test_empty_group_ids_is_a_noop():
    perturb = p.make_perturber(X, FG, regime="conditional")
    pr = perturb(X0, [], m=5)
    assert pr.X.shape == (5, 4) and np.allclose(pr.X, X0) and pr.ood_distance == 0.0


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        p.make_perturber(X, FG, regime="bogus")
