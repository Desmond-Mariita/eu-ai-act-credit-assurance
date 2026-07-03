# src/credit_assurance/faithfulness.py
"""Pure perturbation-faithfulness metrics over LOGICAL feature-groups (model- & explainer-agnostic).

Rankings are expressed as ordered lists of logical group names (from an explainer's Attribution);
erasure is delegated to a group-aware `perturber(x, group_ids, m) -> PerturbResult` (see
`perturbation.make_perturber`). This keeps the metric core decoupled from both the model and the
encoding: it never touches raw column indices, so one-hot groups stay intact.

comprehensiveness(k): base P(bad) - P(bad) after ERASING the top-k ranked groups.
sufficiency(k):       base P(bad) - P(bad) after KEEPING ONLY the top-k groups (erase the rest).
aopc:                 mean of a {k: value} curve.
random_floor:         expected comprehensiveness under RANDOM group orderings (the non-informative
                      baseline that any faithful explainer must beat; a random / label-shuffled
                      attribution should land here).

`predict(X)->P(bad)` takes (n, d) and returns (n,); positive class fixed = bad == 1.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

Predict = Callable[[np.ndarray], np.ndarray]


def _erased_mean(predict: Predict, x: np.ndarray, groups: Sequence[str], perturber, m: int) -> float:
    if len(groups) == 0:
        return float(predict(x[None, :])[0])
    pr = perturber(x, list(groups), m)
    return float(np.mean(predict(pr.X)))


def comprehensiveness(predict, x, ordered_groups, ks, perturber, m: int = 20) -> dict:
    base = float(predict(x[None, :])[0])
    return {k: base - _erased_mean(predict, x, ordered_groups[:k], perturber, m) for k in ks}


def sufficiency(predict, x, ordered_groups, all_groups, ks, perturber, m: int = 20) -> dict:
    base = float(predict(x[None, :])[0])
    out = {}
    for k in ks:
        keep = set(ordered_groups[:k])
        erase = [g for g in all_groups if g not in keep]
        out[k] = base - _erased_mean(predict, x, erase, perturber, m)
    return out


def aopc(curve: dict) -> float:
    return float(np.mean(list(curve.values()))) if curve else 0.0


def random_floor(predict, x, all_groups, ks, perturber, m: int = 20, n_perms: int = 100, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = list(all_groups)
    acc = {k: 0.0 for k in ks}
    for _ in range(n_perms):
        rng.shuffle(groups)
        c = comprehensiveness(predict, x, groups, ks, perturber, m)
        for k in ks:
            acc[k] += c[k]
    return {k: acc[k] / n_perms for k in ks}


def bootstrap_ci(per_instance: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Instance-level bootstrap CI for the mean of a per-instance metric array."""
    rng = np.random.default_rng(seed)
    n = len(per_instance)
    means = np.array([per_instance[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(per_instance.mean()), float(lo), float(hi)
