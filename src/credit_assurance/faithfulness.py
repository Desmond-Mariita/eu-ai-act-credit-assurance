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
                      baseline any faithful explainer must beat; a random / label-shuffled
                      attribution should land here).

All statistics are hand-implemented (no black-box faithfulness library) so every formula is auditable.
`predict(X)->P(bad)` takes (n, d) and returns (n,); positive class fixed = bad == 1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

if TYPE_CHECKING:  # type-only; keeps this core independently liftable (no runtime coupling)
    from credit_assurance.perturbation import Perturb

# A model scorer: maps a design matrix (n, d) to P(bad)=P(class 1) as a length-n vector.
Predict = Callable[[np.ndarray], np.ndarray]


def _erased_mean(predict: Predict, x: np.ndarray, groups: Sequence[str],
                 perturber: "Perturb", m: int) -> float:
    r"""Mean model score after erasing ``groups`` from instance ``x`` over ``m`` donor draws.

    LaTeX: \bar f_{\text{erase}(G)}(x) = \frac{1}{m} \sum_{j=1}^{m} f\!\big(\tilde x^{(j)}\big),
    where each \tilde x^{(j)} replaces the columns of groups G via the perturber; the empty-set
    erasure returns f(x) itself (no averaging).

    Args:
        predict: Scorer mapping (n, d) -> P(bad) length n.
        x: Single instance, shape (d,).
        groups: Logical group names to erase (may be empty).
        perturber: Group-aware erasure operator (see ``perturbation.make_perturber``).
        m: Number of donor draws to average over.

    Returns:
        The scalar mean P(bad) over the m perturbed rows (or f(x) if ``groups`` is empty).
    """
    if len(groups) == 0:
        return float(predict(x[None, :])[0])
    pr = perturber(x, list(groups), m)
    return float(np.mean(predict(pr.X)))


def comprehensiveness(predict: Predict, x: np.ndarray, ordered_groups: Sequence[str],
                      ks: Sequence[int], perturber: "Perturb", m: int = 20,
                      absolute: bool = False) -> dict[int, float]:
    r"""Comprehensiveness curve: the drop in P(bad) from erasing the top-k ranked groups.

    LaTeX (signed):   \text{comp}_k(x) = f(x) - \bar f_{\text{erase}(g_{1:k})}(x)
    LaTeX (absolute): \text{comp}_k(x) = \big| f(x) - \bar f_{\text{erase}(g_{1:k})}(x) \big|

    The absolute variant is direction-agnostic: it avoids sign cancellation when the top-k mix
    risk-increasing (positive) and protective (negative) features whose movements partly cancel.

    Args:
        predict: Scorer mapping (n, d) -> P(bad) length n.
        x: Single instance, shape (d,).
        ordered_groups: Logical groups ranked most- to least-important by the explainer.
        ks: The prefix sizes k at which to evaluate the curve.
        perturber: Group-aware erasure operator.
        m: Donor draws per erasure.
        absolute: If True, report |movement| instead of the signed drop.

    Returns:
        Mapping ``{k: comprehensiveness_at_k}``.
    """
    base = float(predict(x[None, :])[0])
    out: dict[int, float] = {}
    for k in ks:
        diff = base - _erased_mean(predict, x, ordered_groups[:k], perturber, m)
        out[k] = abs(diff) if absolute else diff
    return out


def sufficiency(predict: Predict, x: np.ndarray, ordered_groups: Sequence[str],
                all_groups: Sequence[str], ks: Sequence[int], perturber: "Perturb",
                m: int = 20) -> dict[int, float]:
    r"""Sufficiency curve: the drop in P(bad) from KEEPING ONLY the top-k groups (erasing the rest).

    LaTeX: \text{suff}_k(x) = f(x) - \bar f_{\text{erase}(\,\text{all}\setminus g_{1:k})}(x).
    A faithful, sufficient top-k leaves the score close to f(x) (small drop).

    Args:
        predict: Scorer mapping (n, d) -> P(bad) length n.
        x: Single instance, shape (d,).
        ordered_groups: Logical groups ranked most- to least-important.
        all_groups: The full set of logical groups.
        ks: Prefix sizes k to evaluate.
        perturber: Group-aware erasure operator.
        m: Donor draws per erasure.

    Returns:
        Mapping ``{k: sufficiency_at_k}``.
    """
    base = float(predict(x[None, :])[0])
    out: dict[int, float] = {}
    for k in ks:
        keep = set(ordered_groups[:k])
        erase = [g for g in all_groups if g not in keep]
        out[k] = base - _erased_mean(predict, x, erase, perturber, m)
    return out


def aopc(curve: dict[int, float]) -> float:
    r"""Area Over the Perturbation Curve — the mean of a ``{k: value}`` curve.

    LaTeX: \text{AOPC} = \frac{1}{|K|} \sum_{k \in K} \text{value}_k.

    Args:
        curve: Mapping ``{k: value}`` (e.g. a comprehensiveness or sufficiency curve).

    Returns:
        The mean of the curve values, or 0.0 for an empty curve.
    """
    return float(np.mean(list(curve.values()))) if curve else 0.0


def random_floor(predict: Predict, x: np.ndarray, all_groups: Sequence[str], ks: Sequence[int],
                 perturber: "Perturb", m: int = 20, n_perms: int = 100, seed: int = 0,
                 absolute: bool = False) -> dict[int, float]:
    r"""Non-informative floor: expected comprehensiveness under RANDOM group orderings.

    LaTeX: \text{floor}_k(x) = \mathbb{E}_{\pi \sim \text{Unif}(S_G)}\big[\text{comp}_k(x; \pi)\big]
    \approx \frac{1}{P} \sum_{p=1}^{P} \text{comp}_k(x; \pi_p),
    estimated by P=`n_perms` uniform random permutations \pi_p of the logical groups. Any faithful
    explainer must beat this floor; a random or label-shuffled attribution should land on it.

    Args:
        predict: Scorer mapping (n, d) -> P(bad) length n.
        x: Single instance, shape (d,).
        all_groups: The full set of logical groups to permute.
        ks: Prefix sizes k to evaluate.
        perturber: Group-aware erasure operator.
        m: Donor draws per erasure.
        n_perms: Number of random permutations averaged (Monte-Carlo estimate of the expectation).
        seed: RNG seed (reproducible floor).
        absolute: If True, use the absolute (movement) comprehensiveness.

    Returns:
        Mapping ``{k: floor_at_k}``.
    """
    rng = np.random.default_rng(seed)
    groups = list(all_groups)
    acc = {k: 0.0 for k in ks}
    for _ in range(n_perms):
        rng.shuffle(groups)
        c = comprehensiveness(predict, x, groups, ks, perturber, m, absolute=absolute)
        for k in ks:
            acc[k] += c[k]
    return {k: acc[k] / n_perms for k in ks}


def bootstrap_ci(per_instance: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    r"""Percentile bootstrap CI for the mean of a per-instance metric array.

    LaTeX: draw B bootstrap resamples \{i_1,\dots,i_n\} \sim \text{Unif}(\{1..n\}) with replacement,
    form \bar\theta^{*b} = \frac1n \sum_j \theta_{i_j}; report the point mean and the
    [\alpha/2, 1-\alpha/2] empirical quantiles of \{\bar\theta^{*b}\}_{b=1}^{B}.

    Args:
        per_instance: Per-instance metric values, shape (n,).
        n_boot: Number of bootstrap resamples B.
        seed: RNG seed.
        alpha: Two-sided miscoverage (0.05 -> 95% CI).

    Returns:
        Tuple ``(point_mean, ci_low, ci_high)``.
    """
    rng = np.random.default_rng(seed)
    n = len(per_instance)
    means = np.array([per_instance[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(per_instance.mean()), float(lo), float(hi)
