"""scripts/60_reason_codes.py — Phase 2 Task 13: counterfactual RECOURSE / reason codes (GDPR 13-15).

Recourse feasibility is computed by an EXHAUSTIVE integer grid/line search over the genuinely
actionable, independently-negotiable loan terms — **loan duration** and **credit amount** (every valid
integer in the train range) — scored by the SHA-pinned model, against the DEPLOYED 1/6 accept
threshold. (Installment rate is dropped: coupled to amount/duration/income, not independently
actionable. Age/residence/dependents are immutable/protected; one-hots are never touched.) This
replaces an initial DiCE random search that conflated search-failure with infeasibility. Reason-code
attribution uses a **range-normalised** minimal change (so months and DM are comparable). Reports
recourse availability + sparsity + reason-code features, with Wilson CIs. Neutral report.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
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
COL = {"Attribute2": "loan_duration_months", "Attribute5": "credit_amount"}   # actionable, negotiable


def _wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    cols = list(df.drop(columns=["y"]).columns)
    X = df.drop(columns=["y"]).to_numpy(dtype=float)
    ci = {c: cols.index(c) for c in COL}

    model_path = ROOT / "models" / "lightgbm.txt"
    pinned = json.loads((METRICS / "models.json").read_text())["audited_model"]["sha256"]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != pinned:
        raise SystemExit("audited model hash mismatch (re-run scripts/10_train.py)")
    booster = lgb.Booster(model_file=str(model_path))

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    thr = D.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    lo, hi = X[idx_tr].min(axis=0), X[idx_tr].max(axis=0)
    dur_grid = np.arange(int(lo[ci["Attribute2"]]), int(hi[ci["Attribute2"]]) + 1)
    amt_grid = np.arange(int(lo[ci["Attribute5"]]), int(hi[ci["Attribute5"]]) + 1)   # exact integers
    grids = {"Attribute2": dur_grid, "Attribute5": amt_grid}
    rng_span = {c: float(hi[ci[c]] - lo[ci[c]]) for c in COL}                          # for normalising

    p_all = booster.predict(X)
    declined = [i for i in idx_te if p_all[i] > thr]

    def line(x, col, grid):
        """min-|Δ| single-feature value on `grid` reaching P(bad)<=thr, else None."""
        rows = np.tile(x, (len(grid), 1))
        rows[:, ci[col]] = grid
        ok = grid[booster.predict(rows) <= thr]
        return float(ok[np.argmin(np.abs(ok - x[ci[col]]))]) if len(ok) else None

    def two_feature(x):
        """any (duration, amount) integer pair reaching acceptance? (low-memory duration loop)."""
        for d in dur_grid:
            xd = x.copy()
            xd[ci["Attribute2"]] = d
            rows = np.tile(xd, (len(amt_grid), 1))
            rows[:, ci["Attribute5"]] = amt_grid
            if (booster.predict(rows) <= thr).any():
                return True
        return False

    feasible, sparsity, feat_freq, examples = 0, [], Counter(), []
    for i in declined:
        x = X[i]
        found = {c: line(x, c, grids[c]) for c in COL}
        found = {c: v for c, v in found.items() if v is not None}
        if found:                                   # 1-feature recourse (range-normalised choice)
            feasible += 1
            sparsity.append(1)
            best = min(found, key=lambda c: abs(found[c] - x[ci[c]]) / rng_span[c])
            feat_freq[COL[best]] += 1
            if len(examples) < 6:
                examples.append({"orig_P_bad": round(float(p_all[i]), 3),
                                 "change": {COL[best]: [round(float(x[ci[best]]), 1),
                                                        round(float(found[best]), 1)]}})
        elif two_feature(x):                        # needs both
            feasible += 1
            sparsity.append(2)
            feat_freq[COL["Attribute2"]] += 1
            feat_freq[COL["Attribute5"]] += 1

    n = len(declined)
    out = {
        "dataset": "german_credit", "model": "models/lightgbm.txt (SHA verified)",
        "threshold": round(thr, 4), "n_declined": int(n),
        "method": "exhaustive integer grid/line search over actionable loan terms, pinned model",
        "actionable_features": COL,
        "policy_recourse_rate": round(feasible / n, 4),
        "policy_recourse_wilson95": _wilson(feasible, n),
        "infeasible_rate": round((n - feasible) / n, 4),
        "infeasible_wilson95": _wilson(n - feasible, n),
        "mean_features_changed": round(float(np.mean(sparsity)), 3) if sparsity else None,
        "single_feature_recourse_rate": round(sum(s == 1 for s in sparsity) / n, 4),
        "reason_code_feature_frequency_range_normalised": dict(feat_freq),
        "example_recourse": examples,
        "note": "Recourse = a valid integer change to loan duration and/or credit amount reaching "
                "P(bad)<=1/6 (deployed accept). Grid is EVERY integer in the train range, so "
                "'infeasible' means no loan-term change works (decline rests on non-actionable "
                "factors). Reason-code attribution uses a range-normalised minimal change (months vs "
                "DM comparable). NB: an initial DiCE random search found only 72.5% recourse (a "
                "superseded run, not recomputed here) — off-the-shelf CF tools can conflate search-"
                "failure with infeasibility. Real-world feasibility (can an applicant borrow less?) "
                "is not assessed.",
    }
    (METRICS / "reason_codes_german_credit.json").write_text(json.dumps(out, indent=2))
    print(f"declined {n} | recourse {out['policy_recourse_rate']} {out['policy_recourse_wilson95']} | "
          f"infeasible {out['infeasible_rate']} {out['infeasible_wilson95']} | "
          f"1-feat {out['single_feature_recourse_rate']} | freq(norm) {dict(feat_freq)}")


if __name__ == "__main__":
    main()
