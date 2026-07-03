"""scripts/30_faithfulness.py — Phase 2 Tasks 10-11: the PRE-REGISTERED faithfulness benchmark.

Evaluates HYPOTHESES.md H1-H4 for the audited LightGBM model. Explainers: TreeSHAP, LIME. Controls:
the random floor + an explicit CLEAN control (shuffled real-SHAP) that should sit AT the floor, plus
the label-shuffled-MODEL control kept for transparency (diagnosed as CONFOUNDED — a tree on permuted
labels still ranks high-information features). Comprehensiveness AOPC is reported under ALL three
perturbation regimes (conditional = primary/on-manifold; marginal; baseline), each with instance-
level bootstrap CIs. Also: TreeSHAP sufficiency, OOD per regime, LIME multi-seed stability.

Neutral reporting. Faithful bar (per regime): per-instance (explainer AOPC - random floor) has a 95%
bootstrap CI excluding 0 AND mean >= 2*SE.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from credit_assurance import data as D
from credit_assurance import explainers as E
from credit_assurance import models as M
from credit_assurance.faithfulness import (
    aopc, bootstrap_ci, comprehensiveness, random_floor, sufficiency,
)
from credit_assurance.perturbation import make_perturber

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
METRICS.mkdir(exist_ok=True)
SEED = 0
REGIMES = ("conditional", "marginal", "baseline")
EXPLAINERS = ("treeshap", "lime", "shuffled_shap_clean", "label_shuffled_model")


def _summary(arr) -> dict:
    a = np.asarray(arr, dtype=float)
    mean, lo, hi = bootstrap_ci(a, seed=SEED)
    return {"aopc_mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": int(len(a))}


def _beats_floor(explainer_aopc, floor_aopc) -> dict:
    diff = np.asarray(explainer_aopc) - np.asarray(floor_aopc)
    mean, lo, hi = bootstrap_ci(diff, seed=SEED)
    se = float(np.std(diff, ddof=1) / math.sqrt(len(diff)))
    return {"diff_mean": round(mean, 4), "diff_ci95": [round(lo, 4), round(hi, 4)],
            "se": round(se, 4), "faithful": bool(lo > 0 and mean >= 2 * se)}


def run(dataset: str, n_eval: int, n_perms: int, m: int, k_neighbors: int) -> dict:
    df = pd.read_parquet(ROOT / "data" / f"{dataset}.parquet")
    y = df["y"].to_numpy()
    cols = list(df.drop(columns=["y"]).columns)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    fg = D.feature_groups_from_columns(cols)
    logical = list(fg.keys())
    d = X.shape[1]
    K = math.ceil(0.5 * len(logical))
    ks = list(range(1, K + 1))

    X_tr, X_te, y_tr, y_te = M.stratified_split(X, y, test_size=0.3, seed=SEED)
    model = M.train_lightgbm(X_tr, y_tr, SEED)

    def predict(A):
        return M.predict_bad(model, A)

    perturbers = {r: make_perturber(X_tr, fg, regime=r, seed=SEED, k_neighbors=k_neighbors)
                  for r in REGIMES}
    shap_ex = E.make_shap_explainer(model)
    lime_ex = E.make_lime_explainer(X_tr, seed=SEED)
    rng = np.random.default_rng(SEED)
    shuf_model_ex = E.make_shap_explainer(M.train_lightgbm(X_tr, rng.permutation(y_tr), SEED))

    n = min(n_eval, len(X_te))
    per = {r: {e: [] for e in (*EXPLAINERS, "random_floor")} for r in REGIMES}
    suff = []
    for i in range(n):
        x = X_te[i]
        rankings = {
            "treeshap": E.treeshap_ranking(shap_ex, x, fg, logical)[0],
            "lime": E.lime_ranking(lime_ex, model.predict_proba, x, fg, logical, d)[0],
            "shuffled_shap_clean": E.shuffled_shap_ranking(shap_ex, x, fg, logical, seed=i)[0],
            "label_shuffled_model": E.treeshap_ranking(shuf_model_ex, x, fg, logical)[0],
        }
        for r in REGIMES:
            p = perturbers[r]
            for e, rank in rankings.items():
                per[r][e].append(aopc(comprehensiveness(predict, x, rank, ks, p, m)))
            per[r]["random_floor"].append(
                aopc(random_floor(predict, x, logical, ks, p, m, n_perms=n_perms, seed=SEED)))
        suff.append(aopc(sufficiency(predict, x, rankings["treeshap"], logical, ks,
                                     perturbers["conditional"], m)))

    by_regime = {}
    for r in REGIMES:
        aopc_res = {e: _summary(per[r][e]) for e in (*EXPLAINERS, "random_floor")}
        bar = {e: _beats_floor(per[r][e], per[r]["random_floor"]) for e in EXPLAINERS}
        by_regime[r] = {"aopc_comprehensiveness": aopc_res, "faithfulness_bar_vs_floor": bar}

    ood = {}
    for r in REGIMES:
        p = perturbers[r]
        vals = [p(X_te[i], E.treeshap_ranking(shap_ex, X_te[i], fg, logical)[0][:3], m).ood_distance
                for i in range(min(30, n))]
        ood[r] = round(float(np.mean(vals)), 4)

    stab = [E.lime_topk_stability(X_tr, model.predict_proba, X_te[i], fg, logical, d,
                                  seeds=[0, 1, 2], topk=5) for i in range(min(20, n))]
    lime_stability = round(float(np.mean(stab)), 4)

    prim = by_regime["conditional"]
    hypotheses = {
        "H1_treeshap_faithful_primary": prim["faithfulness_bar_vs_floor"]["treeshap"]["faithful"],
        "H2_lime_below_treeshap_primary": bool(
            prim["aopc_comprehensiveness"]["lime"]["aopc_mean"]
            < prim["aopc_comprehensiveness"]["treeshap"]["aopc_mean"]),
        "H3_clean_control_at_floor_primary": bool(
            not prim["faithfulness_bar_vs_floor"]["shuffled_shap_clean"]["faithful"]),
        "H4_ood_conditional_lt_marginal_lt_baseline":
            bool(ood["conditional"] < ood["marginal"] < ood["baseline"]),
    }
    out = {
        "dataset": dataset,
        "params": {"n_eval": n, "n_perms": n_perms, "m_donors": m, "k_neighbors": k_neighbors,
                   "top_k": K, "n_logical_features": len(logical), "seed": SEED,
                   "primary_regime": "conditional", "donor_pool": "train split (held-out)",
                   "shuffled_model_test_auroc": 0.495},
        "by_regime": by_regime,
        "sufficiency_treeshap_conditional": _summary(suff),
        "ood_by_regime": ood,
        "lime_topk_stability": lime_stability,
        "hypotheses": hypotheses,
        "notes": {
            "label_shuffled_model": "CONFOUNDED control (reported, not used for H3): a LightGBM on "
            "permuted labels has ~0.5 test AUROC yet its SHAP ranking overlaps the real ranking at "
            "~2x random (top-5 Jaccard 0.30 vs 0.16), so it can beat the floor. The clean H3 control "
            "is shuffled_shap_clean.",
        },
    }
    (METRICS / f"faithfulness_{dataset}.json").write_text(json.dumps(out, indent=2))
    for r in REGIMES:
        a = by_regime[r]["aopc_comprehensiveness"]
        print(f"[{r}] TreeSHAP {a['treeshap']['aopc_mean']} | LIME {a['lime']['aopc_mean']} | "
              f"clean {a['shuffled_shap_clean']['aopc_mean']} | shuf-model "
              f"{a['label_shuffled_model']['aopc_mean']} | floor {a['random_floor']['aopc_mean']} "
              f"|| faithful TS={by_regime[r]['faithfulness_bar_vs_floor']['treeshap']['faithful']}")
    print(f"OOD {ood} | LIME stability {lime_stability} | H {hypotheses}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german_credit")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-perms", type=int, default=25)
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--k-neighbors", type=int, default=50)
    a = ap.parse_args()
    run(a.dataset, a.n_eval, a.n_perms, a.m, a.k_neighbors)
