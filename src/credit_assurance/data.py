# src/credit_assurance/data.py
"""Load + clean the two credit datasets; derive/handle protected attributes; splits; costs.

German Credit (Statlog, UCI id 144) is the mainline. Give Me Some Credit (GMSC) is the
larger generalization check. Raw data lands under data/ and is git-ignored.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

_PS = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}
def sex_from_personal_status(s: pd.Series) -> pd.Series:
    return s.map(_PS)

def cost_sensitive_threshold(cost_fn: float, cost_fp: float) -> float:
    # threshold on P(bad): predict 'bad' when P(bad) > cost_fp/(cost_fn+cost_fp)
    return cost_fp / (cost_fn + cost_fp)

def load_german_credit() -> dict:
    """Returns {X (numeric, encoded), y (1='bad'), protected (sex/age_band/foreign_worker),
    feature_names, cost_fn, cost_fp, name}. Statlog attribute codes per the UCI data dictionary.

    Column names confirmed from ucimlrepo fetch_ucirepo(id=144):
      Attribute9  = personal status and sex (Statlog codes A91-A95)
      Attribute13 = age in years (integer)
      Attribute20 = foreign worker (A201=yes, A202=no)
    """
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=144)
    raw = ds.data.features.copy()
    y = (ds.data.targets.iloc[:, 0] == 2).astype(int)        # UCI target: 1=good, 2=bad -> 1=bad
    sex = sex_from_personal_status(raw["Attribute9"])         # personal_status/sex code
    age = pd.to_numeric(raw["Attribute13"])                   # age in years (already int64, confirmed)
    foreign = (raw["Attribute20"] == "A201").map({True: "yes", False: "no"})  # A201=yes, A202=no
    protected = pd.DataFrame({"sex": sex.values,
                              "age_band": pd.cut(age, [0, 25, 40, 60, 200],
                                                 labels=["<=25", "26-40", "41-60", "60+"]),
                              "foreign_worker": foreign.values})
    X = pd.get_dummies(raw, drop_first=False)                 # one-hot categoricals; numeric kept
    return {"X": X, "y": y, "protected": protected, "feature_names": list(X.columns),
            "cost_fn": 5.0, "cost_fp": 1.0, "name": "german_credit"}

def load_gmsc() -> dict:
    """Give Me Some Credit. Download cs-training.csv into data/ (Kaggle public mirror) if absent;
    target = SeriousDlqin2yrs (1=default). No protected attributes shipped, so it is the
    generalization check for the faithfulness method only, not the governance dossier."""
    path = DATA / "cs-training.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Place the Give Me Some Credit cs-training.csv at {path} "
            "(download from the public Kaggle dataset 'GiveMeSomeCredit').")
    df = pd.read_csv(path, index_col=0)
    y = df.pop("SeriousDlqin2yrs").astype(int)
    X = df.fillna(df.median(numeric_only=True))
    return {"X": X, "y": y, "feature_names": list(X.columns), "name": "gmsc"}
