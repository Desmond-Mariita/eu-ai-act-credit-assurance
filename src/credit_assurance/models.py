# src/credit_assurance/models.py
"""The audited model (LightGBM) + an EBM glassbox challenger, plus pure decision/calibration helpers.

Positive class is fixed = `bad` == 1; `predict_bad` returns P(bad). Training is seeded and
deterministic (single-threaded LightGBM) so the saved model artifact hash is reproducible — the
audited model is frozen and recorded in governance/00-scenario.md §5.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(X, y, test_size: float = 0.3, seed: int = 0):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_lightgbm(X, y, seed: int = 0):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=20,
        deterministic=True, num_threads=1, random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    return model


def train_ebm(X, y, seed: int = 0):
    from interpret.glassbox import ExplainableBoostingClassifier
    model = ExplainableBoostingClassifier(random_state=seed)
    model.fit(X, y)
    return model


def predict_bad(model, X) -> np.ndarray:
    """P(bad) — probability of class 1 ('bad'). Assumes classes_ == [0, 1]."""
    return np.asarray(model.predict_proba(X))[:, 1]


def decide(probs, threshold: float) -> np.ndarray:
    """Predict 'bad' (1) when P(bad) > threshold (cost-sensitive)."""
    return (np.asarray(probs) > threshold).astype(int)


def calibration_report(y_true, probs, n_bins: int = 10) -> dict:
    """Brier score + a reliability table (mean predicted vs observed frequency per bin)."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    brier = float(np.mean((probs - y_true) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    reliability = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            reliability.append({
                "bin": b, "mean_pred": float(probs[m].mean()),
                "frac_pos": float(y_true[m].mean()), "n": int(m.sum()),
            })
    return {"brier": brier, "reliability": reliability}
