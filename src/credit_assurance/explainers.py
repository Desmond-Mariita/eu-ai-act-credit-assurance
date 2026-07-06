# src/credit_assurance/explainers.py
"""Explainers under audit -> LOGICAL-feature rankings + values.

Per-encoded-column attributions are aggregated to logical feature-groups (sum of member-column
contributions), so faithfulness is measured at the group level and one-hot groups stay intact.
`shap` / `lime` are imported lazily (the Phase-2 `[models]` extra) so importing this module — and the
pure helpers below — needs neither. TreeSHAP and LIME are the third-party explainers UNDER audit; they
are intentionally external (we measure them, we do not reimplement them). The aggregation, ranking, and
stability statistics around them are hand-implemented here.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Sequence

import numpy as np

from .perturbation import FeatureGroups

# A ranking result: (groups ordered by |value| desc, {group: signed value}).
Ranking = tuple[list[str], dict[str, float]]
# predict_proba: (n, d) -> (n, n_classes) class-probability matrix.
PredictProba = Callable[[np.ndarray], np.ndarray]


def aggregate_to_logical(col_values: np.ndarray, feature_groups: FeatureGroups,
                         logical_names: Sequence[str]) -> dict[str, float]:
    r"""Sum member-column attributions into a per-logical-feature signed value.

    LaTeX: \phi_g = \sum_{c \in \text{cols}(g)} \phi_c — the group attribution is the sum of its
    encoded columns' attributions (so a one-hot categorical's dummies contribute as one feature).

    Args:
        col_values: Per-encoded-column attributions, shape (d,).
        feature_groups: Logical name -> column-index tuple.
        logical_names: Logical feature names to emit.

    Returns:
        Mapping ``{logical_name: signed_group_attribution}``.
    """
    col_values = np.asarray(col_values, dtype=float)
    return {name: float(sum(col_values[c] for c in feature_groups[name])) for name in logical_names}


def rank_groups(values: dict[str, float]) -> list[str]:
    r"""Order logical group names by attribution magnitude, descending.

    LaTeX: ranking \pi such that |\phi_{\pi(1)}| \ge |\phi_{\pi(2)}| \ge \dots

    Args:
        values: Mapping ``{group: signed value}``.

    Returns:
        Group names sorted by ``|value|`` descending.
    """
    return sorted(values, key=lambda n: -abs(values[n]))


def treeshap_ranking(shap_explainer: Any, x: np.ndarray, feature_groups: FeatureGroups,
                     logical_names: Sequence[str]) -> Ranking:
    """Group-level TreeSHAP ranking for the positive class at instance ``x``.

    Handles the several SHAP return conventions (list-of-classes, (n, d, n_class), or (n, d)) and
    extracts the class-1 contribution before aggregating to logical groups.

    Args:
        shap_explainer: A fitted ``shap.TreeExplainer`` (external, under audit).
        x: Instance, shape (d,).
        feature_groups: Logical name -> column-index tuple.
        logical_names: Logical feature names.

    Returns:
        ``(ranked_groups, {group: value})``.
    """
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


def random_ranking(logical_names: Sequence[str], seed: int = 0) -> Ranking:
    """Draw a genuinely random logical ranking (i.i.d. standard-normal pseudo-attributions).

    Args:
        logical_names: Logical feature names.
        seed: RNG seed.

    Returns:
        ``(ranked_groups, {group: value})`` carrying no importance information.
    """
    rng = np.random.default_rng(seed)
    vals = {n: float(rng.normal()) for n in logical_names}
    return rank_groups(vals), vals


def shuffled_shap_ranking(shap_explainer: Any, x: np.ndarray, feature_groups: FeatureGroups,
                          logical_names: Sequence[str], seed: int = 0) -> Ranking:
    """CLEAN negative control: the real model's SHAP magnitudes randomly reassigned across features.

    A genuinely random ranking that carries no importance information, so it should land at the random
    floor. (Contrast the label-shuffled-MODEL control, which is confounded because a tree fit on
    permuted labels still ranks high-information features.)

    Args:
        shap_explainer: Fitted ``shap.TreeExplainer`` for the real model.
        x: Instance, shape (d,).
        feature_groups: Logical name -> column-index tuple.
        logical_names: Logical feature names.
        seed: Permutation seed.

    Returns:
        ``(ranked_groups, {group: value})`` with the real SHAP values permuted across groups.
    """
    _, vals = treeshap_ranking(shap_explainer, x, feature_groups, logical_names)
    rng = np.random.default_rng(seed)
    shuffled = dict(zip(list(vals.keys()), rng.permutation(list(vals.values()))))
    return rank_groups(shuffled), shuffled


def make_shap_explainer(model: Any) -> Any:
    """Build a ``shap.TreeExplainer`` (external explainer under audit; imported lazily).

    Args:
        model: A fitted tree model (e.g. LightGBM).

    Returns:
        A ``shap.TreeExplainer`` bound to ``model``.
    """
    import shap
    return shap.TreeExplainer(model)


# --- Group-valid LIME -------------------------------------------------------------------------
# Operate in the ORIGINAL (label-encoded) feature space so one-hot exclusivity is preserved. Passing
# the one-hot dummies to LIME as independent categorical columns lets it sample invalid multi-hot /
# zero-hot states (empirically ~99.9% of neighbours off-manifold). Instead we label-encode each logical
# categorical to a SINGLE column, let LIME sample one valid category, and re-expand to a valid one-hot
# inside predict_fn. LIME then attributes directly at the logical-feature level (no re-aggregation).

def to_logical_space(
    X_encoded: np.ndarray, cols: Sequence[str], feature_groups: FeatureGroups,
    logical_names: Sequence[str],
) -> tuple[np.ndarray, list[int], dict[int, list[str]], list[tuple[int, list[int]]],
           list[tuple[int, int]]]:
    r"""Map a one-hot encoded matrix to a label-encoded logical matrix (one column per logical feature).

    LaTeX: for a one-hot group with columns ``idxs``, the logical value is
    \arg\max_{c \in idxs} X_{:,c} (the active dummy's position); numeric singletons pass through.

    Args:
        X_encoded: One-hot encoded design matrix, shape (n, d).
        cols: Encoded column names (``<logical>_<value>`` for one-hots).
        feature_groups: Logical name -> column-index tuple.
        logical_names: Logical feature names in output-column order.

    Returns:
        Tuple ``(X_log, cat_positions, cat_names, cat_groups, num_groups)`` where ``X_log`` is the
        (n, n_logical) label-encoded matrix, ``cat_positions`` the logical indices that are categorical,
        ``cat_names`` maps each categorical logical index to its category labels, ``cat_groups`` pairs
        each categorical logical index with its encoded columns, and ``num_groups`` pairs each numeric
        logical index with its single encoded column.
    """
    X_encoded = np.asarray(X_encoded, dtype=float)
    X_log = np.zeros((X_encoded.shape[0], len(logical_names)))
    cat_positions: list[int] = []
    cat_names: dict[int, list[str]] = {}
    cat_groups: list[tuple[int, list[int]]] = []
    num_groups: list[tuple[int, int]] = []
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


def logical_predict_fn(predict_proba: PredictProba, n_encoded: int,
                       cat_groups: Sequence[tuple[int, list[int]]],
                       num_groups: Sequence[tuple[int, int]]) -> PredictProba:
    """Wrap ``predict_proba`` to accept label-encoded logical rows, re-expanding to valid one-hots.

    Each logical categorical is re-expanded to exactly one active dummy before scoring, so LIME only
    ever sees on-manifold (valid one-hot) rows. The rounded category index is clipped into range as a
    safety invariant (LIME samples integer category labels, but clipping guards any out-of-range draw).

    Args:
        predict_proba: The model's ``(n, d) -> (n, n_classes)`` scorer.
        n_encoded: Number of encoded columns d (width of the re-expanded matrix).
        cat_groups: ``(logical_index, encoded_columns)`` pairs for categoricals.
        num_groups: ``(logical_index, encoded_column)`` pairs for numerics.

    Returns:
        A function ``f(X_log) -> (n, n_classes)`` scoring label-encoded logical rows.
    """
    def f(X_log: np.ndarray) -> np.ndarray:
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


def make_lime_explainer(X_log_train: np.ndarray, cat_positions: Sequence[int],
                        cat_names: dict[int, list[str]], seed: int = 0) -> Any:
    """Build a group-valid ``LimeTabularExplainer`` in the label-encoded logical space (lazy import).

    Args:
        X_log_train: Training data in logical space, shape (n, n_logical).
        cat_positions: Logical indices that are categorical.
        cat_names: Category labels per categorical logical index.
        seed: LIME ``random_state``.

    Returns:
        A configured ``LimeTabularExplainer`` (external explainer under audit).
    """
    from lime.lime_tabular import LimeTabularExplainer
    # categorical_features + categorical_names => LIME samples one valid category per logical group;
    # discretize_continuous=True => numeric weights are instance contributions (active bin == 1).
    return LimeTabularExplainer(np.asarray(X_log_train, dtype=float), mode="classification",
                                categorical_features=cat_positions, categorical_names=cat_names,
                                discretize_continuous=True, random_state=seed)


def lime_ranking(lime_explainer: Any, predict_fn: PredictProba, x_log: np.ndarray,
                 logical_names: Sequence[str], num_features: int) -> Ranking:
    """LIME ranking in logical space: LIME columns map 1:1 to logical features (no re-aggregation).

    Args:
        lime_explainer: A group-valid ``LimeTabularExplainer``.
        predict_fn: The re-expanding logical scorer (see ``logical_predict_fn``).
        x_log: Instance in logical space, shape (n_logical,).
        logical_names: Logical feature names (LIME column j -> ``logical_names[j]``).
        num_features: Number of features LIME should return weights for.

    Returns:
        ``(ranked_groups, {group: weight})`` for the positive class.
    """
    exp = lime_explainer.explain_instance(
        np.asarray(x_log, dtype=float), predict_fn, num_features=num_features, labels=(1,))
    vals = {name: 0.0 for name in logical_names}
    for j, w in exp.as_map()[1]:
        vals[logical_names[int(j)]] = float(w)
    return rank_groups(vals), vals


def lime_topk_stability(explainer_factory: Callable[[int], Any], predict_fn: PredictProba,
                        x_log: np.ndarray, logical_names: Sequence[str], num_features: int,
                        seeds: Sequence[int], topk: int = 5) -> float:
    r"""Mean pairwise top-k Jaccard of LIME rankings across seeds (1.0 = perfectly stable).

    LaTeX: \text{stability} = \operatorname{mean}_{s<t} J(T_s, T_t),\;
    J(A,B) = \frac{|A \cap B|}{|A \cup B|}, where T_s is the top-k group set from seed s.

    Args:
        explainer_factory: ``seed -> fresh group-valid LIME explainer``.
        predict_fn: The re-expanding logical scorer.
        x_log: Instance in logical space, shape (n_logical,).
        logical_names: Logical feature names.
        num_features: Number of LIME features per explanation.
        seeds: Seeds to compare (each yields one top-k set).
        topk: Prefix size k for the Jaccard overlap.

    Returns:
        Mean pairwise top-k Jaccard, or 1.0 if fewer than two seeds are given.
    """
    tops = []
    for s in seeds:
        r, _ = lime_ranking(explainer_factory(s), predict_fn, x_log, logical_names, num_features)
        tops.append(set(r[:topk]))
    js = [len(a & b) / len(a | b) for a, b in itertools.combinations(tops, 2)]
    return float(np.mean(js)) if js else 1.0
