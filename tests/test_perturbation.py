# tests/test_perturbation.py
import numpy as np
from credit_assurance import perturbation as p

def test_baseline_replaces_with_fixed_values():
    X = np.array([[10.0, 0.0], [20.0, 1.0], [30.0, 1.0]])
    perturb = p.baseline_perturber(X)           # median per column
    x = np.array([99.0, 0.0])
    out = perturb(x, [0])
    assert out.shape == (1, 2)
    assert out[0, 0] == 20.0 and out[0, 1] == 0.0  # col0 -> median 20, col1 untouched

def test_marginal_draws_from_observed_values():
    X = np.array([[1.0], [2.0], [3.0]])
    perturb = p.marginal_perturber(X, m=5, seed=0)
    out = perturb(np.array([9.0]), [0])
    assert out.shape == (5, 1)
    assert set(np.unique(out[:, 0])).issubset({1.0, 2.0, 3.0})

def test_conditional_keeps_other_features_fixed():
    X = np.random.default_rng(0).normal(size=(200, 3))
    perturb = p.conditional_perturber(X, m=4, k_neighbors=10, seed=0)
    x = np.array([5.0, 0.0, -5.0])
    out = perturb(x, [0])
    assert out.shape == (4, 3)
    assert np.allclose(out[:, 1], 0.0) and np.allclose(out[:, 2], -5.0)  # untouched cols fixed
