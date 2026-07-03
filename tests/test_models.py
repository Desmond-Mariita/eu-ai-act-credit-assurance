# tests/test_models.py
import numpy as np

from credit_assurance import models as m


def test_decide_uses_cost_sensitive_threshold():
    probs = np.array([0.1, 0.2, 0.5])
    assert list(m.decide(probs, threshold=1 / 6)) == [0, 1, 1]   # > 1/6 -> 'bad'


def test_calibration_report_brier_and_bins():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    rep = m.calibration_report(y, p, n_bins=5)
    assert rep["brier"] == 0.0
    assert any(b["frac_pos"] == 1.0 for b in rep["reliability"])


def test_predict_bad_returns_probability_of_class_one():
    class Fake:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    p = m.predict_bad(Fake(), np.array([[0.2], [0.9]]))
    assert np.allclose(p, [0.2, 0.9])
