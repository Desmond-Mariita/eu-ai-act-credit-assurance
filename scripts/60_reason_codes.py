"""scripts/60_reason_codes.py — Phase 2: counterfactual RECOURSE / reason codes (GDPR Arts. 13-15).

DiCE counterfactuals for DECLINED applicants, varying ONLY genuinely actionable loan parameters
(duration, credit amount, installment rate) — NOT age/residence/dependents (immutable/protected) and
NOT the one-hot categoricals (frozen -> valid). Recourse is checked against the DEPLOYED 1/6 policy
(P(bad) <= 1/6 = accepted), not DiCE's 0.5 argmax. Reports recourse availability + the reason-code
features. Uses the deterministic model verified identical to the SHA-pinned artifact. Neutral report.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import dice_ml
import numpy as np
import pandas as pd
from dice_ml import Dice
from sklearn.model_selection import train_test_split

from credit_assurance import data as D
from credit_assurance import models as M

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
SEED = 0
N_SAMPLE = 80
TOTAL_CFS = 10
ACTIONABLE = {"Attribute2": "loan_duration_months", "Attribute5": "credit_amount",
              "Attribute8": "installment_rate_pct"}


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    Xdf = df.drop(columns=["y"]).astype(float)          # cast bool one-hots -> float for DiCE
    cols = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)

    lgbm = M.train_lightgbm(X[idx_tr], y[idx_tr], SEED)
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / "models" / "lightgbm.txt"))
    if not np.allclose(M.predict_bad(lgbm, X[idx_te]), booster.predict(X[idx_te]), atol=1e-9):
        raise SystemExit("recourse model != pinned audited model")

    thr = D.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    train_df = Xdf.iloc[idx_tr].copy()
    train_df["y"] = y[idx_tr]
    dice_data = dice_ml.Data(dataframe=train_df, continuous_features=cols, outcome_name="y")
    exp = Dice(dice_data, dice_ml.Model(model=lgbm, backend="sklearn"), method="random")

    p_all = M.predict_bad(lgbm, X)
    declined = [i for i in idx_te if p_all[i] > thr][:N_SAMPLE]

    n_model, n_policy, no_cf = 0, 0, 0
    sparsity, feat_freq = [], Counter()
    examples = []
    for i in declined:
        qi = Xdf.iloc[[i]]
        try:
            cf = exp.generate_counterfactuals(qi, total_CFs=TOTAL_CFS, desired_class=0,
                                              features_to_vary=list(ACTIONABLE), random_seed=SEED)
            cfdf = cf.cf_examples_list[0].final_cfs_df
        except Exception:
            cfdf = None
        if cfdf is None or not len(cfdf):
            no_cf += 1
            continue
        pcf = M.predict_bad(lgbm, cfdf[cols].to_numpy(dtype=float))
        n_model += int((pcf < 0.5).any())
        policy = cfdf[pcf <= thr]
        if not len(policy):
            continue
        n_policy += 1
        # sparsest policy-valid recourse
        changed = [[a for a in ACTIONABLE if not np.isclose(policy.iloc[k][a], qi.iloc[0][a])]
                   for k in range(len(policy))]
        best = int(np.argmin([len(c) for c in changed]))
        sparsity.append(len(changed[best]))
        feat_freq.update(ACTIONABLE[a] for a in changed[best])
        if len(examples) < 5:
            examples.append({"orig_P_bad": round(float(p_all[i]), 3),
                             "changes": {ACTIONABLE[a]: [round(float(qi.iloc[0][a]), 1),
                                                         round(float(policy.iloc[best][a]), 1)]
                                         for a in changed[best]}})

    n = len(declined)
    out = {
        "dataset": "german_credit", "threshold": round(thr, 4),
        "n_declined_evaluated": n, "total_cfs_per_instance": TOTAL_CFS,
        "actionable_features": ACTIONABLE,
        "recourse_model_class_rate": round(n_model / n, 4),
        "recourse_policy_1_6_rate": round(n_policy / n, 4),
        "no_counterfactual_found_rate": round(no_cf / n, 4),
        "mean_features_changed_policy_recourse": round(float(np.mean(sparsity)), 3) if sparsity else None,
        "reason_code_feature_frequency": dict(feat_freq),
        "example_recourse": examples,
        "note": "recourse varies ONLY actionable loan parameters (age/residence/dependents excluded as "
                "immutable/protected; one-hots frozen). policy recourse = a CF reaching P(bad)<=1/6 "
                "(deployed accept threshold), stricter than DiCE's 0.5 argmax.",
    }
    (METRICS / "reason_codes_german_credit.json").write_text(json.dumps(out, indent=2))
    print(f"declined evaluated {n} | model-class recourse {out['recourse_model_class_rate']} | "
          f"policy(1/6) recourse {out['recourse_policy_1_6_rate']} | no-CF {out['no_counterfactual_found_rate']}")
    print(f"mean features changed {out['mean_features_changed_policy_recourse']} | freq {dict(feat_freq)}")


if __name__ == "__main__":
    main()
