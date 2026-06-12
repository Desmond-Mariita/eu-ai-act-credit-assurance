# src/credit_assurance/faithfulness.py
"""Pure perturbation-faithfulness metrics (model- and perturbation-agnostic).

comprehensiveness(k): base prob - prob after ERASING the top-k ranked features.
sufficiency(k):       base prob - prob after KEEPING ONLY the top-k (erase the rest).
aopc:                 mean of a {k: value} curve.
random_floor:         expected comprehensiveness curve under random feature rankings.

`predict(X)->probs` and `perturb(x, idxs)->(m,d) donor draws` are injected callables.
"""
from __future__ import annotations
from typing import Callable, Sequence
import numpy as np

Predict = Callable[[np.ndarray], np.ndarray]
Perturb = Callable[[np.ndarray, Sequence[int]], np.ndarray]

def _erased_prob(predict: Predict, x: np.ndarray, idxs: Sequence[int], perturb: Perturb) -> float:
    if len(idxs) == 0:
        return float(predict(x[None, :])[0])
    donors = perturb(x, idxs)                    # (m, d)
    return float(np.mean(predict(donors)))       # expectation over donor draws

def comprehensiveness(predict, x, ranking, ks, perturb) -> dict:
    base = float(predict(x[None, :])[0])
    return {k: base - _erased_prob(predict, x, list(ranking[:k]), perturb) for k in ks}

def sufficiency(predict, x, ranking, ks, perturb) -> dict:
    base = float(predict(x[None, :])[0])
    d = x.shape[0]
    out = {}
    for k in ks:
        keep = set(int(i) for i in ranking[:k])
        rest = [i for i in range(d) if i not in keep]
        out[k] = base - _erased_prob(predict, x, rest, perturb)
    return out

def aopc(curve: dict) -> float:
    return float(np.mean(list(curve.values()))) if curve else 0.0

def random_floor(predict, x, ks, perturb, n_perms=100, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    d = x.shape[0]
    acc = {k: 0.0 for k in ks}
    for _ in range(n_perms):
        perm = rng.permutation(d)
        c = comprehensiveness(predict, x, perm, ks, perturb)
        for k in ks:
            acc[k] += c[k]
    return {k: acc[k] / n_perms for k in ks}

def bootstrap_ci(per_instance: np.ndarray, n_boot=2000, seed=0, alpha=0.05) -> tuple[float, float, float]:
    """Instance-level bootstrap CI for the mean of a per-instance metric array."""
    rng = np.random.default_rng(seed)
    n = len(per_instance)
    means = np.array([per_instance[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(per_instance.mean()), float(lo), float(hi)
