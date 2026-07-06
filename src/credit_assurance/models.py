# src/credit_assurance/models.py
"""The audited model (LightGBM) + an EBM glassbox challenger, plus pure decision/calibration helpers.

Positive class is fixed = `bad` == 1; `predict_bad` returns P(bad). Training is seeded and
deterministic (single-threaded LightGBM) so the saved model artifact hash is reproducible — the
audited model is frozen and recorded in governance/00-scenario.md §5. LightGBM and the EBM are the
models UNDER audit (intentionally external); the decision and calibration statistics below are
hand-implemented so they are auditable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 0) -> tuple:
    """Stratified train/test split (preserves the class balance in both folds).

    Args:
        X: Design matrix, shape (n, d).
        y: Binary labels, shape (n,).
        test_size: Test fraction.
        seed: RNG seed (fixed split for reproducibility).

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_lightgbm(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Any:
    """Train the audited LightGBM classifier (deterministic, single-threaded -> reproducible hash).

    Args:
        X: Training design matrix, shape (n, d).
        y: Binary training labels (1 = bad).
        seed: ``random_state`` for LightGBM.

    Returns:
        A fitted ``lightgbm.LGBMClassifier`` (the model under audit).
    """
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=20,
        deterministic=True, num_threads=1, random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    return model


def train_ebm(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Any:
    """Train the Explainable Boosting Machine glassbox challenger.

    Args:
        X: Training design matrix, shape (n, d).
        y: Binary training labels (1 = bad).
        seed: ``random_state`` for the EBM.

    Returns:
        A fitted ``ExplainableBoostingClassifier`` (challenger under audit).
    """
    from interpret.glassbox import ExplainableBoostingClassifier
    model = ExplainableBoostingClassifier(random_state=seed)
    model.fit(X, y)
    return model


def predict_bad(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(bad) = P(class 1) for each row.

    Assumes the model's ``classes_ == [0, 1]`` so column 1 is the positive ('bad') class.

    Args:
        model: A fitted classifier exposing ``predict_proba``.
        X: Design matrix, shape (n, d).

    Returns:
        P(bad) as a length-n array.
    """
    return np.asarray(model.predict_proba(X))[:, 1]


def decide(probs: np.ndarray, threshold: float) -> np.ndarray:
    r"""Cost-sensitive decision: predict 'bad' (1) when P(bad) exceeds the threshold.

    LaTeX: \hat y_i = \mathbb{1}\{ p_i > t \}.

    Args:
        probs: P(bad) scores, shape (n,).
        threshold: Decision threshold t (see ``data.cost_sensitive_threshold``).

    Returns:
        Integer 0/1 decisions, shape (n,).
    """
    return (np.asarray(probs) > threshold).astype(int)


def calibration_report(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    r"""Brier score + a reliability table (mean predicted vs observed frequency per bin).

    LaTeX (Brier): \text{BS} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2. The reliability table bins
    scores into ``n_bins`` equal-width bins on [0, 1] and reports (mean predicted, observed frequency,
    count) per non-empty bin — a well-calibrated model has mean predicted ≈ observed per bin.

    Args:
        y_true: Binary outcomes, shape (n,).
        probs: P(bad) scores, shape (n,).
        n_bins: Number of equal-width reliability bins.

    Returns:
        ``{"brier": float, "reliability": [{"bin", "mean_pred", "frac_pos", "n"}, ...]}``.
    """
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
