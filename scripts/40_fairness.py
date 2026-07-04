"""scripts/40_fairness.py — Phase 2: fairness assessment of the audited model (EU AI Act Art. 10(2)(f-g)
bias examination; Art. 15; GDPR). Group metrics at the cost-sensitive 5:1 threshold across the proxy /
special-category-adjacent attributes German Credit ships (sex, foreign_worker, age_band), aligned to
the SAME stratified split the model uses. Writes metrics/fairness_german_credit.json. Neutral report.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, count, demographic_parity_difference, equalized_odds_difference,
    false_positive_rate, selection_rate, true_positive_rate,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from credit_assurance import data as D
from credit_assurance import models as M

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
METRICS.mkdir(exist_ok=True)
SEED = 0
METRIC_FNS = {"count": count, "selection_rate_declined": selection_rate,
              "tpr_recall_bad": true_positive_rate, "fpr": false_positive_rate,
              "accuracy": accuracy_score}


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    prot = D.protected_attributes_from_encoded(df.drop(columns=["y"]))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    model = M.train_lightgbm(X[idx_tr], y[idx_tr], SEED)
    thr = D.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    y_te = y[idx_te]
    y_hat = M.decide(M.predict_bad(model, X[idx_te]), thr)

    out = {
        "dataset": "german_credit", "threshold": round(thr, 4), "n_test": int(len(y_te)),
        "note": "decisions at the cost-sensitive 5:1 threshold; 'bad'==1==declined. selection_rate = "
                "fraction DECLINED. Attributes are proxies / special-category-adjacent; small subgroups "
                "(esp. foreign_worker='no', age '<25'/'60+') -> low power; disparities are indicative.",
        "by_attribute": {},
    }
    for attr, groups_all in prot.items():
        g = groups_all[idx_te]
        mf = MetricFrame(metrics=METRIC_FNS, y_true=y_te, y_pred=y_hat, sensitive_features=g)
        by_group = {}
        for grp in mf.by_group.index:
            row = mf.by_group.loc[grp]
            by_group[str(grp)] = {k: (int(row[k]) if k == "count" else round(float(row[k]), 4))
                                  for k in METRIC_FNS}
        dp = float(demographic_parity_difference(y_te, y_hat, sensitive_features=g))
        eo = float(equalized_odds_difference(y_te, y_hat, sensitive_features=g))
        out["by_attribute"][attr] = {"by_group": by_group,
                                     "demographic_parity_difference": round(dp, 4),
                                     "equalized_odds_difference": round(eo, 4)}
        rates = {k: v["selection_rate_declined"] for k, v in by_group.items()}
        print(f"[{attr}] DP diff {dp:.3f} | EO diff {eo:.3f} | decline rate by group {rates}")

    (METRICS / "fairness_german_credit.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
