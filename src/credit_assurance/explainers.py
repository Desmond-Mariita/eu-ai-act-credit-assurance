# src/credit_assurance/explainers.py
"""Explainers under audit -> LOGICAL-feature rankings + values.

Per-encoded-column attributions are aggregated to logical feature-groups (sum of member-column
contributions), so faithfulness is measured at the group level and one-hot groups stay intact.
`shap` / `lime` are imported lazily (the Phase-2 `[models]` extra) so importing this module — and the
pure helpers below — needs neither.
"""
from __future__ import annotations

import itertools

import numpy as np

from .perturbation import FeatureGroups


def aggregate_to_logical(col_values, feature_groups: FeatureGroups, logical_names) -> dict:
    """Sum member-column attributions into a per-logical-feature signed value."""
    col_values = np.asarray(col_values, dtype=float)
    return {name: float(sum(col_values[c] for c in feature_groups[name])) for name in logical_names}


def rank_groups(values: dict) -> list:
    """Logical group names ordered by |value| descending."""
    return sorted(values, key=lambda n: -abs(values[n]))


def treeshap_ranking(shap_explainer, x, feature_groups, logical_names):
    raw = shap_explainer.shap_values(np.asarray(x, dtype=float)[None, :])
    if isinstance(raw, list):      # some shap versions return [class0, class1]
        raw = raw[1]
    sv = np.asarray(raw)
    if sv.ndim == 3:               # (n, d, n_class) -> class 1
        col = sv[0, :, 1]
    else:                          # (n, d) already the positive-class contribution
        col = sv[0]
    vals = aggregate_to_logical(col, feature_groups, logical_names)
    return rank_groups(vals), vals


def lime_ranking(lime_explainer, predict_proba, x, feature_groups, logical_names, num_features):
    exp = lime_explainer.explain_instance(
        np.asarray(x, dtype=float), predict_proba, num_features=num_features, labels=(1,))
    col = np.zeros(num_features)
    for j, w in exp.as_map()[1]:
        col[int(j)] = w
    vals = aggregate_to_logical(col, feature_groups, logical_names)
    return rank_groups(vals), vals


def random_ranking(logical_names, seed=0):
    rng = np.random.default_rng(seed)
    vals = {n: float(rng.normal()) for n in logical_names}
    return rank_groups(vals), vals


def shuffled_shap_ranking(shap_explainer, x, feature_groups, logical_names, seed=0):
    """CLEAN negative control: the real model's SHAP magnitudes randomly reassigned across logical
    features — a genuinely random ranking that carries no importance information, so it should land
    at the random floor. (Contrast the label-shuffled-MODEL control, which is confounded because a
    tree fit on permuted labels still ranks high-information features.)"""
    _, vals = treeshap_ranking(shap_explainer, x, feature_groups, logical_names)
    rng = np.random.default_rng(seed)
    shuffled = dict(zip(list(vals.keys()), rng.permutation(list(vals.values()))))
    return rank_groups(shuffled), shuffled


def make_shap_explainer(model):
    import shap
    return shap.TreeExplainer(model)


def make_lime_explainer(X_train, seed=0):
    from lime.lime_tabular import LimeTabularExplainer
    # discretize_continuous=True (LIME's default): weights act as instance-specific CONTRIBUTIONS
    # (active bin == 1), not raw scaled-feature sensitivities — the correct attribution to rank.
    # KNOWN LIMITATION (documented in finding 10 §Deviations): `categorical_features` is NOT passed, so
    # LIME treats the one-hot dummies as continuous in its neighbourhood sampling (it evaluates the model
    # on off-manifold fractional-dummy states). This is the *naive off-the-shelf LIME baseline* as
    # practitioners commonly use it — part of what the audit surfaces; our faithfulness EVALUATION still
    # aggregates LIME's ranking to logical groups via a one-hot-respecting perturber. Passing
    # `categorical_features` (from feature_groups) is the correct-usage alternative (future work).
    return LimeTabularExplainer(np.asarray(X_train, dtype=float), mode="classification",
                                discretize_continuous=True, random_state=seed)


def lime_topk_stability(X_train, predict_proba, x, feature_groups, logical_names,
                        num_features, seeds, topk=5):
    """Mean pairwise top-k Jaccard of LIME rankings across seeds (1.0 = perfectly stable)."""
    tops = []
    for s in seeds:
        le = make_lime_explainer(X_train, seed=s)
        r, _ = lime_ranking(le, predict_proba, x, feature_groups, logical_names, num_features)
        tops.append(set(r[:topk]))
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    return float(np.mean(js)) if js else 1.0
