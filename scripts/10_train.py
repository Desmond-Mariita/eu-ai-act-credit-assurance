"""scripts/10_train.py — Phase 2 Task 8: train the audited LightGBM + EBM challenger.

Reads the frozen German-Credit snapshot, does a stratified holdout with a leakage check, applies the
cost-sensitive (5:1) threshold, reports calibration, and writes metrics/models.json. Saves and
SHA256-hashes the audited model (LightGBM text format = deterministic) so the ToE model version can
be pinned in governance/00-scenario.md.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from credit_assurance import models as M
from credit_assurance.data import cost_sensitive_threshold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
MODELS = ROOT / "models"
METRICS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def main() -> None:
    """Train + freeze the audited LightGBM (and EBM challenger) and write ``metrics/models.json``.

    Runs a univariate leakage check, a stratified holdout, the cost-sensitive decision + calibration
    report, then serialises the LightGBM to deterministic text and records its SHA256 as the pinned ToE
    model version. The narrow (univariate, linear) leakage check is intentional and disclosed — it does
    not test nonlinear or multivariate leakage.
    """
    df = pd.read_parquet(ROOT / "data" / "german_credit.parquet")
    y = df["y"].to_numpy()
    X = df.drop(columns=["y"]).to_numpy(dtype=float)

    # Leakage check: no single feature (near-)perfectly predicts the target.
    # LaTeX: r_j = |corr(X_{:,j}, y)| = |cov(X_j, y)| / (sigma_{X_j} sigma_y); pass iff max_j r_j < 0.95.
    corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if np.std(X[:, j]) > 0 else 0.0
                     for j in range(X.shape[1])])
    leakage = {"max_abs_feature_target_corr": round(float(np.nanmax(corr)), 4),
               "passed": bool(np.nanmax(corr) < 0.95)}

    X_tr, X_te, y_tr, y_te = M.stratified_split(X, y, test_size=0.3, seed=0)

    lgbm = M.train_lightgbm(X_tr, y_tr, seed=0)
    p_te = M.predict_bad(lgbm, X_te)
    thr = cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    cm = confusion_matrix(y_te, M.decide(p_te, thr)).tolist()
    cal = M.calibration_report(y_te, p_te)

    ebm = M.train_ebm(X_tr, y_tr, seed=0)
    p_ebm = M.predict_bad(ebm, X_te)

    model_path = MODELS / "lightgbm.txt"          # deterministic text serialisation
    lgbm.booster_.save_model(str(model_path))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metrics = {
        "dataset": "german_credit",
        "split": {"test_size": 0.3, "seed": 0, "protocol": "stratified holdout", "leakage_check": leakage},
        "audited_model": {
            "type": "LightGBM", "artifact": "models/lightgbm.txt", "sha256": model_sha,
            "auroc": round(float(roc_auc_score(y_te, p_te)), 4),
            "auprc": round(float(average_precision_score(y_te, p_te)), 4),
            "brier": round(cal["brier"], 4),
            "cost_sensitive_threshold": round(thr, 4),
            "confusion_at_threshold": cm,
        },
        "challenger_model": {
            "type": "EBM (glassbox, validation comparator only — not in the faithfulness leaderboard)",
            "auroc": round(float(roc_auc_score(y_te, p_ebm)), 4),
            "auprc": round(float(average_precision_score(y_te, p_ebm)), 4),
        },
        "calibration": cal,
    }
    (METRICS / "models.json").write_text(json.dumps(metrics, indent=2))
    print(f"LightGBM AUROC {metrics['audited_model']['auroc']} | "
          f"EBM AUROC {metrics['challenger_model']['auroc']} | "
          f"Brier {metrics['audited_model']['brier']}")
    print(f"model sha256 {model_sha[:16]}… | leakage passed: {leakage['passed']} "
          f"(max |corr| {leakage['max_abs_feature_target_corr']})")


if __name__ == "__main__":
    main()
