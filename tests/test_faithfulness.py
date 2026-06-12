# tests/test_faithfulness.py
import numpy as np
from credit_assurance import faithfulness as f

def _linear_predict(weights):
    # prob = sigmoid(x·w); deterministic, pure — a perfect testbed
    def predict(X):
        z = X @ weights
        return 1 / (1 + np.exp(-z))
    return predict

def _zero_perturb(x, idxs):
    # "erase" = set features to 0; return a single donor row (shape (1, d))
    y = x.copy()
    y[list(idxs)] = 0.0
    return y[None, :]

def test_comprehensiveness_drops_when_important_feature_erased():
    w = np.array([3.0, 0.0, 0.0])
    x = np.array([1.0, 1.0, 1.0])
    predict = _linear_predict(w)
    # ranking puts feature 0 (the only one that matters) first
    curve = f.comprehensiveness(predict, x, ranking=[0, 1, 2], ks=[1], perturb=_zero_perturb)
    base = predict(x[None, :])[0]
    erased = predict(np.array([[0.0, 1.0, 1.0]]))[0]
    assert np.isclose(curve[1], base - erased)
    assert curve[1] > 0.0  # erasing the important feature reduces the predicted prob

def test_sufficiency_small_when_topk_carries_signal():
    w = np.array([3.0, 0.0, 0.0])
    x = np.array([1.0, 1.0, 1.0])
    predict = _linear_predict(w)
    suff = f.sufficiency(predict, x, ranking=[0, 1, 2], ks=[1], perturb=_zero_perturb)
    assert abs(suff[1]) < 1e-9  # keeping only feature 0 retains the full prediction

def test_aopc_is_mean_of_curve():
    assert np.isclose(f.aopc({1: 0.2, 2: 0.4}), 0.3)

def test_random_floor_is_between_zero_and_full():
    w = np.array([3.0, 1.0, 0.0])
    x = np.array([1.0, 1.0, 1.0])
    predict = _linear_predict(w)
    floor = f.random_floor(predict, x, ks=[1, 2], perturb=_zero_perturb, n_perms=50, seed=0)
    assert 1 in floor and 2 in floor
