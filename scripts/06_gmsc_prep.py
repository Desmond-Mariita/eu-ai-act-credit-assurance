"""scripts/06_gmsc_prep.py — prepare a benchmark-ready Give Me Some Credit snapshot.

Cleans the documented data errors (past-due sentinels 96/98 -> NaN; age < 18 -> NaN) and
stratified-subsamples to keep the conditional-kNN donor pool tractable (the frozen perturber refits
NearestNeighbors per call). NaN is PRESERVED — median imputation is fit on the train split only, in
30_faithfulness (no leakage). Writes data/gmsc.parquet (X + y). This is the canonical GMSC prep.
All-numeric -> each column is its own logical feature group. Generalization check only (no protected
attributes shipped -> not part of the governance dossier).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
N_SUBSAMPLE = 6000
SEED = 0
PASTDUE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
           "NumberOfTime60-89DaysPastDueNotWorse"]


def main() -> None:
    """Clean documented GMSC data errors, stratified-subsample, and write ``data/gmsc.parquet``.

    Maps past-due sentinels 96/98 and age<18 to NaN (NaN preserved for train-split imputation
    downstream — no leakage), then takes a class-stratified subsample to keep the conditional-kNN donor
    pool tractable. Prints the parquet's shape, default rate, and SHA256.
    """
    df = pd.read_csv(DATA / "cs-training.csv", index_col=0)
    y = df.pop("SeriousDlqin2yrs").astype(int)

    # documented data errors -> NaN, then median impute (see governance/data-dictionary.md)
    for c in PASTDUE:
        df.loc[df[c].isin([96, 98]), c] = np.nan
    df.loc[df["age"] < 18, "age"] = np.nan
    n_sentinels = int(sum((pd.read_csv(DATA / "cs-training.csv", index_col=0)[c].isin([96, 98])).sum()
                          for c in PASTDUE))
    X = df   # keep NaN — median imputation is fit on the TRAIN split only (in 30_faithfulness), no leak

    # stratified subsample (preserve the ~6.7% default rate) for a tractable donor pool
    rng = np.random.default_rng(SEED)
    idx = []
    for cls in (0, 1):
        pool = np.where(y.to_numpy() == cls)[0]
        take = round(N_SUBSAMPLE * len(pool) / len(y))
        idx.extend(rng.choice(pool, size=take, replace=False))
    idx = np.sort(np.array(idx))
    Xs, ys = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    out = Xs.copy()
    out["y"] = ys.to_numpy()
    path = DATA / "gmsc.parquet"
    out.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"gmsc.parquet: {out.shape} | default rate {ys.mean():.4f} | "
          f"past-due sentinels cleaned {n_sentinels} | features {list(Xs.columns)}")
    print(f"sha256 {sha}")


if __name__ == "__main__":
    main()
