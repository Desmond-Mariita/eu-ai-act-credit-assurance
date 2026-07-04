"""scripts/50_robustness.py — Phase 2 Task 13: robustness of the AUDITED (SHA-pinned) model (Art. 15).

Decision-flip rate + prediction shift under small Gaussian perturbations of the CONTINUOUS features
(loan duration, credit amount, age) — small ordinals and one-hot categoricals left intact — at several
noise levels eps (fraction of each feature's train std), clipped to the train range and rounded to
valid integers. Reports per-eps flip rate with an instance bootstrap CI, plus the near-threshold share.
Uses the deployed 1/6 decision. Writes metrics/robustness_german_credit.json. Neutral report.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_assurance import data as D

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
SEED = 0
EPS = [0.05, 0.1, 0.2, 0.5]
N_DRAWS = 10
CONTINUOUS = ["Attribute2", "Attribute5", "Attribute13"]   # duration, credit amount, age


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    cols = list(df.drop(columns=["y"]).columns)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    cont = [cols.index(c) for c in CONTINUOUS]

    model_path = ROOT / "models" / "lightgbm.txt"
    pinned = json.loads((METRICS / "models.json").read_text())["audited_model"]["sha256"]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != pinned:
        raise SystemExit("audited model hash mismatch (re-run scripts/10_train.py)")
    booster = lgb.Booster(model_file=str(model_path))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    thr = D.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    X_te = X[idx_te]
    n = len(X_te)
    std, lo, hi = X[idx_tr].std(axis=0), X[idx_tr].min(axis=0), X[idx_tr].max(axis=0)
    p0 = booster.predict(X_te)
    dec0 = p0 > thr
    rng = np.random.default_rng(SEED)

    near = float(np.mean(np.abs(p0 - thr) < 0.05))
    by_eps = {}
    for eps in EPS:
        flip_i, dp_i = np.zeros(n), np.zeros(n)          # per-instance means over draws
        for _ in range(N_DRAWS):
            Xp = X_te.copy()
            for j in cont:
                Xp[:, j] = np.clip(np.round(Xp[:, j] + rng.normal(0, eps * std[j], n)), lo[j], hi[j])
            pp = booster.predict(Xp)
            flip_i += (pp > thr) != dec0
            dp_i += np.abs(pp - p0)
        flip_i /= N_DRAWS
        boot = np.array([flip_i[rng.integers(0, n, n)].mean() for _ in range(2000)])   # instance bootstrap
        by_eps[str(eps)] = {
            "decision_flip_rate": round(float(flip_i.mean()), 4),
            "flip_ci95": [round(float(np.quantile(boot, 0.025)), 4),
                          round(float(np.quantile(boot, 0.975)), 4)],
            "mean_abs_delta_p": round(float((dp_i / N_DRAWS).mean()), 4),
        }

    out = {
        "dataset": "german_credit", "model": "models/lightgbm.txt (SHA verified)",
        "threshold": round(thr, 4), "n_test": int(n), "n_draws": N_DRAWS,
        "perturbed": "continuous features (duration, credit amount, age); Gaussian eps*train-std, "
                     "clipped to train range, rounded to int; ordinals + one-hots left intact",
        "share_near_threshold_|p-thr|<0.05": round(near, 4),
        "by_eps": by_eps,
        "note": "eps = noise sd as a fraction of each feature's train std (a synthetic probe, not "
                "calibrated to real error logs). flip = fraction of decisions changed vs unperturbed; "
                "flip_ci95 = 95% instance-bootstrap CI. Age is perturbed as generic input noise.",
    }
    (METRICS / "robustness_german_credit.json").write_text(json.dumps(out, indent=2))
    for eps, v in by_eps.items():
        print(f"eps={eps}: flip {v['decision_flip_rate']} {v['flip_ci95']} | mean|dP| {v['mean_abs_delta_p']}")
    print(f"near-threshold share {near:.3f}")


if __name__ == "__main__":
    main()
