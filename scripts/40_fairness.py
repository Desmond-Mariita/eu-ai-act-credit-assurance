"""scripts/40_fairness.py — Phase 2 Task 12: fairness assessment of the AUDITED (hash-pinned) model.

EU AI Act Art. 10(2)(f-g) bias examination + Art. 15. Group metrics at the cost-sensitive 5:1
threshold across German Credit's non-discrimination-protected attributes (sex, foreign_worker,
age_band), recovered from the encoded parquet and aligned to the model's stratified split. Reports
disparities WITH bootstrap CIs (single split -> uncertainty matters), signed pairwise diffs for the
binary attributes, plus per-group base rates and AUROC (threshold-independent). Loads the pinned
model artifact (SHA verified) rather than retraining. Neutral report.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, count, demographic_parity_difference, equalized_odds_difference,
    false_positive_rate, selection_rate, true_positive_rate,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_assurance import data as D

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
SEED = 0
B_BOOT = 2000
MIN_N = 30                       # groups smaller than this -> disparities not inferable
METRIC_FNS = {"count": count, "selection_rate_declined": selection_rate,
              "tpr_recall_bad": true_positive_rate, "fpr": false_positive_rate,
              "accuracy": accuracy_score}


def _clean(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4)


def _boot_ci(stat, n, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B_BOOT):
        v = stat(rng.integers(0, n, n))
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return None
    return [round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4)]


def _fpr(y_true, y_pred):
    neg = y_true == 0
    return float(y_pred[neg].mean()) if neg.any() else np.nan


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    prot = D.protected_attributes_from_encoded(df.drop(columns=["y"]))

    # audit the PINNED artifact (verify SHA), not a retrain
    model_path = ROOT / "models" / "lightgbm.txt"
    pinned = json.loads((METRICS / "models.json").read_text())["audited_model"]["sha256"]
    got = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if got != pinned:
        raise SystemExit(f"audited model hash mismatch: {got} != {pinned} (re-run scripts/10_train.py)")
    booster = lgb.Booster(model_file=str(model_path))

    _, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    thr = D.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    y_te = y[idx_te]
    p_te = booster.predict(X[idx_te])          # P(bad)
    y_hat = (p_te > thr).astype(int)
    n = len(y_te)

    out = {
        "dataset": "german_credit", "model": "models/lightgbm.txt (SHA verified)",
        "threshold": round(thr, 4), "n_test": int(n), "n_bootstrap": B_BOOT, "min_group_n": MIN_N,
        "note": "decisions at the cost-sensitive 5:1 threshold; 'bad'==1==declined; selection_rate = "
                "fraction DECLINED. DP/EO are non-negative (folded) max-differences -> bootstrap CIs "
                "are biased away from 0; for the binary attributes the SIGNED pairwise diff (with a CI "
                "that can include 0) is the valid significance test. The audited model TRAINS ON "
                "personal-status (sex proxy) + age directly. n<30 groups -> disparities not inferable.",
        "by_attribute": {},
    }
    for attr, g_all in prot.items():
        g = g_all[idx_te]
        mf = MetricFrame(metrics=METRIC_FNS, y_true=y_te, y_pred=y_hat, sensitive_features=g)
        by_group = {}
        for grp in mf.by_group.index:
            row = mf.by_group.loc[grp]
            cnt = int(row["count"])
            gm = (g == grp)
            auc = (roc_auc_score(y_te[gm], p_te[gm])
                   if len(np.unique(y_te[gm])) == 2 else np.nan)
            by_group[str(grp)] = {
                "count": cnt, "base_rate_bad": _clean(float(y_te[gm].mean())),
                "selection_rate_declined": _clean(row["selection_rate_declined"]),
                "tpr_recall_bad": _clean(row["tpr_recall_bad"]), "fpr": _clean(row["fpr"]),
                "accuracy": _clean(row["accuracy"]), "auroc": _clean(auc),
                "inferable": bool(cnt >= MIN_N),
            }
        dp = float(demographic_parity_difference(y_te, y_hat, sensitive_features=g))
        eo = float(equalized_odds_difference(y_te, y_hat, sensitive_features=g))
        entry = {
            "by_group": by_group,
            "demographic_parity_difference": round(dp, 4),
            "dp_ci95_folded": _boot_ci(
                lambda idx, g=g: demographic_parity_difference(
                    y_te[idx], y_hat[idx], sensitive_features=g[idx]), n),
            "equalized_odds_difference": round(eo, 4),
            "eo_ci95_folded": _boot_ci(
                lambda idx, g=g: equalized_odds_difference(
                    y_te[idx], y_hat[idx], sensitive_features=g[idx]), n),
        }
        groups = sorted(set(g))
        if len(groups) == 2:      # signed diff (b - a): the valid significance test
            a, b = groups
            for name, fn in (("decline_rate", lambda yt, yp: float(yp.mean())), ("fpr", _fpr)):
                point = fn(y_te[g == b], y_hat[g == b]) - fn(y_te[g == a], y_hat[g == a])

                def stat(idx, fn=fn):
                    gg, yt, yp = g[idx], y_te[idx], y_hat[idx]
                    return fn(yt[gg == b], yp[gg == b]) - fn(yt[gg == a], yp[gg == a])
                entry[f"signed_{name}_diff_{b}_minus_{a}"] = _clean(point)
                entry[f"signed_{name}_diff_ci95"] = _boot_ci(stat, n)
        out["by_attribute"][attr] = entry
        print(f"[{attr}] DP {dp:.3f} ci{entry['dp_ci95_folded']} | EO {eo:.3f} ci{entry['eo_ci95_folded']}")

    (METRICS / "fairness_german_credit.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
