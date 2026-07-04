"""scripts/50_robustness.py — Phase 2: robustness of the AUDITED (SHA-pinned) model (EU AI Act Art. 15).

Decision-flip rate + prediction shift under small Gaussian perturbations of the NUMERIC features
(one-hot categoricals left intact), at several noise levels eps (fraction of each feature's std).
Also reports the share of instances near the decision boundary (the fragile ones). Uses the deployed
1/6 decision. Writes metrics/robustness_german_credit.json. Neutral report.
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


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    cols = list(df.drop(columns=["y"]).columns)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    numeric = [i for i, c in enumerate(cols) if "_" not in c]

    model_path = ROOT / "models" / "lightgbm.txt"
    pinned = json.loads((METRICS / "models.json").read_text())["audited_model"]["sha256"]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != pinned:
        raise SystemExit("audited model hash mismatch (re-run scripts/10_train.py)")
    booster = lgb.Booster(model_file=str(model_path))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    thr = D.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    X_te = X[idx_te]
    std = X[idx_tr].std(axis=0)            # per-feature scale from train
    lo = X[idx_tr].min(axis=0)
    hi = X[idx_tr].max(axis=0)
    p0 = booster.predict(X_te)
    dec0 = p0 > thr
    rng = np.random.default_rng(SEED)

    near = float(np.mean(np.abs(p0 - thr) < 0.05))     # fragile band around the threshold
    by_eps = {}
    for eps in EPS:
        flips, dP, tot = 0, [], 0
        for _ in range(N_DRAWS):
            Xp = X_te.copy()
            for j in numeric:
                Xp[:, j] = np.clip(np.round(Xp[:, j] + rng.normal(0, eps * std[j], len(Xp))),
                                   lo[j], hi[j])
            pp = booster.predict(Xp)
            flips += int(np.sum((pp > thr) != dec0))
            dP.extend(np.abs(pp - p0))
            tot += len(Xp)
        by_eps[str(eps)] = {"decision_flip_rate": round(flips / tot, 4),
                            "mean_abs_delta_p": round(float(np.mean(dP)), 4)}

    out = {
        "dataset": "german_credit", "model": "models/lightgbm.txt (SHA verified)",
        "threshold": round(thr, 4), "n_test": int(len(X_te)), "n_draws": N_DRAWS,
        "perturbed": "numeric features only, Gaussian eps*std, clipped to train range, rounded to int",
        "share_near_threshold_|p-thr|<0.05": round(near, 4),
        "by_eps": by_eps,
        "note": "eps is the noise sd as a fraction of each numeric feature's train std. decision_flip "
                "= fraction of (instance,draw) pairs whose 1/6 accept/decline flips vs unperturbed.",
    }
    (METRICS / "robustness_german_credit.json").write_text(json.dumps(out, indent=2))
    for eps, v in by_eps.items():
        print(f"eps={eps}: flip_rate {v['decision_flip_rate']} | mean|dP| {v['mean_abs_delta_p']}")
    print(f"near-threshold share {near:.3f}")


if __name__ == "__main__":
    main()
