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


# --- Group-valid LIME -------------------------------------------------------------------------
# Operate in the ORIGINAL (label-encoded) feature space so one-hot exclusivity is preserved. Passing
# the one-hot dummies to LIME as independent categorical columns lets it sample invalid multi-hot /
# zero-hot states (empirically ~99.9% of neighbours off-manifold). Instead we label-encode each logical
# categorical to a SINGLE column, let LIME sample one valid category, and re-expand to a valid one-hot
# inside predict_fn. LIME then attributes directly at the logical-feature level (no re-aggregation).

def to_logical_space(X_encoded, cols, feature_groups, logical_names):
    """One-hot encoded matrix -> label-encoded logical matrix (one column per logical feature).
    Returns (X_log, cat_positions, cat_names, cat_groups, num_groups)."""
    X_encoded = np.asarray(X_encoded, dtype=float)
    X_log = np.zeros((X_encoded.shape[0], len(logical_names)))
    cat_positions, cat_names, cat_groups, num_groups = [], {}, [], []
    for j, name in enumerate(logical_names):
        idxs = list(feature_groups[name])
        if len(idxs) > 1 or "_" in cols[idxs[0]]:            # one-hot categorical logical feature
            X_log[:, j] = np.argmax(X_encoded[:, idxs], axis=1)
            cat_positions.append(j)
            cat_names[j] = [cols[c].split("_", 1)[1] if "_" in cols[c] else cols[c] for c in idxs]
            cat_groups.append((j, idxs))
        else:                                                 # numeric singleton
            X_log[:, j] = X_encoded[:, idxs[0]]
            num_groups.append((j, idxs[0]))
    return X_log, cat_positions, cat_names, cat_groups, num_groups


def logical_predict_fn(predict_proba, n_encoded, cat_groups, num_groups):
    """Wrap predict_proba to accept label-encoded logical rows, re-expanding each logical categorical
    to a VALID one-hot (exactly one active dummy) before scoring — so LIME only ever sees on-manifold
    rows."""
    def f(X_log):
        X_log = np.asarray(X_log, dtype=float)
        n = X_log.shape[0]
        Xe = np.zeros((n, n_encoded))
        rows = np.arange(n)
        for j, idxs in cat_groups:
            pos = np.clip(np.rint(X_log[:, j]).astype(int), 0, len(idxs) - 1)
            Xe[rows, np.asarray(idxs)[pos]] = 1.0
        for j, ei in num_groups:
            Xe[:, ei] = X_log[:, j]
        return predict_proba(Xe)
    return f


def make_lime_explainer(X_log_train, cat_positions, cat_names, seed=0):
    from lime.lime_tabular import LimeTabularExplainer
    # categorical_features + categorical_names => LIME samples one valid category per logical group;
    # discretize_continuous=True => numeric weights are instance contributions (active bin == 1).
    return LimeTabularExplainer(np.asarray(X_log_train, dtype=float), mode="classification",
                                categorical_features=cat_positions, categorical_names=cat_names,
                                discretize_continuous=True, random_state=seed)


def lime_ranking(lime_explainer, predict_fn, x_log, logical_names, num_features):
    """LIME ranking in logical space: LIME columns map 1:1 to logical features (no re-aggregation)."""
    exp = lime_explainer.explain_instance(
        np.asarray(x_log, dtype=float), predict_fn, num_features=num_features, labels=(1,))
    vals = {name: 0.0 for name in logical_names}
    for j, w in exp.as_map()[1]:
        vals[logical_names[int(j)]] = float(w)
    return rank_groups(vals), vals


def lime_topk_stability(explainer_factory, predict_fn, x_log, logical_names, num_features, seeds,
                        topk=5):
    """Mean pairwise top-k Jaccard of LIME rankings across seeds (1.0 = perfectly stable).
    `explainer_factory(seed)` returns a fresh group-valid LIME explainer."""
    tops = []
    for s in seeds:
        r, _ = lime_ranking(explainer_factory(s), predict_fn, x_log, logical_names, num_features)
        tops.append(set(r[:topk]))
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    return float(np.mean(js)) if js else 1.0
