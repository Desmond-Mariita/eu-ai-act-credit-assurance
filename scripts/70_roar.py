"""scripts/70_roar.py — Phase 2 supplement: ROAR (RemOve And Retrain) faithfulness anchor.

ROAR (Hooker et al. 2019) is the retrain-based ground-truth-ish check for global feature importance:
rank features by an explainer, remove the top-k LOGICAL features, RETRAIN a fresh model on the reduced
feature set, and measure held-out AUROC. A FAITHFUL ranking degrades accuracy FASTER than a random
ranking (retraining removes the off-manifold artefact that plagues perturbation-only tests). Compares
TreeSHAP-global, LIME-global, the model's own gain, and a random baseline. Writes
metrics/roar_german_credit.json. Deterministic; model verified == the SHA-pinned artefact. Neutral.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_assurance import data as D
from credit_assurance import explainers as E
from credit_assurance import models as M

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
SEED = 0
N_LIME = 100          # LIME global is per-instance; sample for tractability
N_RANDOM = 5          # random-ranking curves to average


def _auroc_after_removing(X, y, fg, drop_groups, tr, te):
    drop = sorted({c for g in drop_groups for c in fg[g]})
    keep = [j for j in range(X.shape[1]) if j not in drop]
    if not keep:
        return 0.5
    m = M.train_lightgbm(X[np.ix_(tr, keep)], y[tr], SEED)
    return float(roc_auc_score(y[te], M.predict_bad(m, X[np.ix_(te, keep)])))


def _curve(X, y, fg, ranking, ks, tr, te):
    return [round(_auroc_after_removing(X, y, fg, ranking[:k], tr, te), 4) for k in ks]


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    cols = list(df.drop(columns=["y"]).columns)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    fg = D.feature_groups_from_columns(cols)
    logical = list(fg.keys())
    K = len(logical) // 2
    ks = list(range(1, K + 1))

    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    full = M.train_lightgbm(X[tr], y[tr], SEED)
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lightgbm.txt"))
    if not np.allclose(M.predict_bad(full, X[te]), booster.predict(X[te]), atol=1e-9):
        raise SystemExit("ROAR model != pinned audited model")
    base_auroc = round(float(roc_auc_score(y[te], M.predict_bad(full, X[te]))), 4)

    shap_ex = E.make_shap_explainer(full)
    lime_ex = E.make_lime_explainer(X[tr], seed=SEED)
    d = X.shape[1]

    # global rankings (mean |importance| over train instances, aggregated to logical groups)
    ts_imp = {g: 0.0 for g in logical}
    for i in tr:
        _, vals = E.treeshap_ranking(shap_ex, X[i], fg, logical)
        for g in logical:
            ts_imp[g] += abs(vals[g])
    treeshap_rank = sorted(logical, key=lambda g: -ts_imp[g])

    lm_imp = {g: 0.0 for g in logical}
    for i in tr[:N_LIME]:
        _, vals = E.lime_ranking(lime_ex, full.predict_proba, X[i], fg, logical, d)
        for g in logical:
            lm_imp[g] += abs(vals[g])
    lime_rank = sorted(logical, key=lambda g: -lm_imp[g])

    gain = full.booster_.feature_importance(importance_type="gain")
    gain_imp = {g: float(sum(gain[c] for c in fg[g])) for g in logical}
    gain_rank = sorted(logical, key=lambda g: -gain_imp[g])

    rng = np.random.default_rng(SEED)
    rand_curves = []
    for _ in range(N_RANDOM):
        r = list(logical)
        rng.shuffle(r)
        rand_curves.append(_curve(X, y, fg, r, ks, tr, te))
    random_curve = [round(float(np.mean(c)), 4) for c in zip(*rand_curves)]

    curves = {"treeshap": _curve(X, y, fg, treeshap_rank, ks, tr, te),
              "lime": _curve(X, y, fg, lime_rank, ks, tr, te),
              "model_gain": _curve(X, y, fg, gain_rank, ks, tr, te),
              "random": random_curve}

    # faithfulness = AUROC drops FASTER than random -> larger mean (base - auroc_k) across k
    aoc = {name: round(float(np.mean([base_auroc - a for a in c])), 4) for name, c in curves.items()}

    out = {
        "dataset": "german_credit", "model": "models/lightgbm.txt (SHA verified; retrained per removal)",
        "method": "ROAR: remove top-k logical features by each ranking, RETRAIN, measure test AUROC",
        "base_auroc": base_auroc, "top_k": K, "ks": ks,
        "auroc_vs_k": curves,
        "mean_auroc_drop_vs_k": aoc,
        "ranking_top5": {"treeshap": treeshap_rank[:5], "lime": lime_rank[:5], "gain": gain_rank[:5]},
        "note": "Larger mean_auroc_drop = more faithful (removing genuinely important features hurts "
                "more). A faithful explainer should exceed 'random'. Retraining avoids the off-manifold "
                "artefact of perturbation-only tests. Global importances = summed |attribution| (ranking-equivalent to the mean) over "
                f"train (LIME sampled n={N_LIME}); random averaged over {N_RANDOM} orderings.",
    }
    (METRICS / "roar_german_credit.json").write_text(json.dumps(out, indent=2))
    print(f"base AUROC {base_auroc} | mean AUROC drop: " +
          " | ".join(f"{k} {v}" for k, v in aoc.items()))
    print(f"TreeSHAP top5 {treeshap_rank[:5]}")


if __name__ == "__main__":
    main()
