"""scripts/30_faithfulness.py — Phase 2 Tasks 10-11: run the PRE-REGISTERED faithfulness benchmark.

Evaluates the pre-registered hypotheses (HYPOTHESES.md H1-H4) for the audited LightGBM model:
TreeSHAP vs LIME vs a label-shuffled control vs the random floor, under the primary on-manifold
(conditional) perturbation regime, with instance-level bootstrap CIs, LIME multi-seed stability, and
OOD reporting per regime. Neutral reporting — whatever the data show is written to
metrics/faithfulness_<dataset>.json.

Faithful (pre-registered bar): per-instance (explainer AOPC - random floor) has a 95% bootstrap CI
excluding 0 AND mean >= 2 * SE.
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

    def predict(A):                       # P(bad)
        return M.predict_bad(model, A)

    perturber = make_perturber(X_tr, fg, regime="conditional", seed=SEED, k_neighbors=k_neighbors)
    shap_ex = E.make_shap_explainer(model)
    lime_ex = E.make_lime_explainer(X_tr, seed=SEED)
    rng = np.random.default_rng(SEED)
    shuf_ex = E.make_shap_explainer(M.train_lightgbm(X_tr, rng.permutation(y_tr), SEED))

    n = min(n_eval, len(X_te))
    rows = {"treeshap": [], "lime": [], "label_shuffled": [], "random_floor": []}
    suff = []
    for i in range(n):
        x = X_te[i]
        ts, _ = E.treeshap_ranking(shap_ex, x, fg, logical)
        lm, _ = E.lime_ranking(lime_ex, model.predict_proba, x, fg, logical, d)
        sf, _ = E.treeshap_ranking(shuf_ex, x, fg, logical)
        rows["treeshap"].append(aopc(comprehensiveness(predict, x, ts, ks, perturber, m)))
        rows["lime"].append(aopc(comprehensiveness(predict, x, lm, ks, perturber, m)))
        rows["label_shuffled"].append(aopc(comprehensiveness(predict, x, sf, ks, perturber, m)))
        rows["random_floor"].append(
            aopc(random_floor(predict, x, logical, ks, perturber, m, n_perms=n_perms, seed=SEED)))
        suff.append(aopc(sufficiency(predict, x, ts, logical, ks, perturber, m)))

    aopc_res = {k: _summary(v) for k, v in rows.items()}
    verdict = {name: _beats_floor(rows[name], rows["random_floor"])
               for name in ("treeshap", "lime", "label_shuffled")}

    ood = {}
    for regime in ("conditional", "marginal", "baseline"):
        pert = make_perturber(X_tr, fg, regime=regime, seed=SEED, k_neighbors=k_neighbors)
        vals = [pert(X_te[i], E.treeshap_ranking(shap_ex, X_te[i], fg, logical)[0][:3], m).ood_distance
                for i in range(min(30, n))]
        ood[regime] = round(float(np.mean(vals)), 4)

    stab = [E.lime_topk_stability(X_tr, model.predict_proba, X_te[i], fg, logical, d,
                                  seeds=[0, 1, 2], topk=5) for i in range(min(20, n))]
    lime_stability = round(float(np.mean(stab)), 4)

    hypotheses = {
        "H1_treeshap_faithful": verdict["treeshap"]["faithful"],
        "H2_lime_below_treeshap": bool(aopc_res["lime"]["aopc_mean"] < aopc_res["treeshap"]["aopc_mean"]),
        "H3_controls_not_faithful": bool(not verdict["label_shuffled"]["faithful"]),
        "H4_ood_conditional_lt_marginal_lt_baseline":
            bool(ood["conditional"] < ood["marginal"] < ood["baseline"]),
    }
    out = {
        "dataset": dataset,
        "params": {"n_eval": n, "n_perms": n_perms, "m_donors": m, "k_neighbors": k_neighbors,
                   "top_k": K, "n_logical_features": len(logical), "seed": SEED,
                   "primary_regime": "conditional", "donor_pool": "train split (held-out)"},
        "aopc_comprehensiveness": aopc_res,
        "sufficiency_treeshap": _summary(suff),
        "faithfulness_bar_vs_floor": verdict,
        "ood_by_regime": ood,
        "lime_topk_stability": lime_stability,
        "hypotheses": hypotheses,
    }
    (METRICS / f"faithfulness_{dataset}.json").write_text(json.dumps(out, indent=2))
    print(f"[{dataset}] n={n}  TreeSHAP AOPC {aopc_res['treeshap']['aopc_mean']} | "
          f"LIME {aopc_res['lime']['aopc_mean']} | shuffled {aopc_res['label_shuffled']['aopc_mean']} | "
          f"floor {aopc_res['random_floor']['aopc_mean']}")
    print(f"  faithful: TreeSHAP={verdict['treeshap']['faithful']} LIME={verdict['lime']['faithful']} "
          f"shuffled={verdict['label_shuffled']['faithful']} | LIME stability {lime_stability}")
    print(f"  OOD {ood} | hypotheses {hypotheses}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="german_credit")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-perms", type=int, default=20)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--k-neighbors", type=int, default=50)
    a = ap.parse_args()
    run(a.dataset, a.n_eval, a.n_perms, a.m, a.k_neighbors)
