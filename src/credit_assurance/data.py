# src/credit_assurance/data.py
"""Load + clean the two credit datasets; derive/handle protected attributes; splits; costs.

German Credit (Statlog, UCI id 144) is the mainline. Give Me Some Credit (GMSC) is the
larger generalization check. Raw data lands under data/ and is git-ignored.

`sex`, `age`, `foreign_worker` are treated as PROTECTED / PROXY attributes (non-discrimination +
AI Act Art. 10(2)(f)) — NOT GDPR Art. 9 special-category (which is invoked only where special-
category data is actually processed or proxy-inferred; see governance/06-gdpr-dpia.md).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

_PS = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}


def sex_from_personal_status(s: pd.Series) -> pd.Series:
    return s.map(_PS)


# Shared age banding — used by both the raw loader and the encoded-parquet recovery so a given age
# always maps to the same band. right=False -> [0,25),[25,40),[40,60),[60,200).
_AGE_BINS = [0, 25, 40, 60, 200]
_AGE_LABELS = ["<25", "25-39", "40-59", "60+"]


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=_AGE_BINS, labels=_AGE_LABELS, right=False)


def protected_attributes_from_encoded(df: pd.DataFrame) -> dict:
    """Recover German-Credit protected/proxy attributes from the ENCODED parquet columns, aligned to
    its row order: sex (Attribute9 personal-status one-hot), foreign_worker (Attribute20), age_band
    (Attribute13). Protected under non-discrimination law (NOT GDPR Art. 9). Fairness analysis only.
    NB: sex is derived from personal-status, which conflates marital status (A92 = female
    divorced/separated/**married**); and the audited model trains on Attribute9 + Attribute13
    directly, so these attributes are model inputs, not merely external labels."""
    a9 = [c for c in df.columns if c.startswith("Attribute9_")]
    onehot = df[a9].astype(int)
    if not (onehot.sum(axis=1) == 1).all():
        raise ValueError("Attribute9 one-hot rows must have exactly one active dummy")
    ps = onehot.idxmax(axis=1).str.replace("Attribute9_", "", regex=False)
    return {
        "sex": ps.map(_PS).to_numpy(),
        "foreign_worker": df["Attribute20_A201"].astype(int).map({1: "yes", 0: "no"}).to_numpy(),
        "age_band": _age_band(df["Attribute13"]).astype(str).to_numpy(),
    }


def cost_sensitive_threshold(cost_fn: float, cost_fp: float) -> float:
    # threshold on P(bad): predict 'bad' when P(bad) > cost_fp/(cost_fn+cost_fp)
    return cost_fp / (cost_fn + cost_fp)


def feature_groups_from_columns(columns: Iterable[str]) -> dict[str, tuple[int, ...]]:
    """Logical feature -> encoded column indices. One-hot columns named `<logical>_<value>`
    (e.g. `Attribute1_A11`) collapse to `<logical>`; columns without `_` are numeric singletons.
    Feeds the group-aware perturber (perturbation.make_perturber)."""
    groups: dict[str, list[int]] = {}
    for idx, col in enumerate(columns):
        logical = col.split("_", 1)[0] if "_" in col else col
        groups.setdefault(logical, []).append(idx)
    return {k: tuple(v) for k, v in groups.items()}


def load_german_credit() -> dict:
    """Returns {X (encoded), y (1='bad'), protected (sex/age_band/foreign_worker), feature_names,
    feature_groups, cost_fn, cost_fp, name}. Statlog attribute codes per the UCI data dictionary.

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
                              "age_band": _age_band(age),
                              "foreign_worker": foreign.values})
    X = pd.get_dummies(raw, drop_first=False)                 # one-hot categoricals; numeric kept
    return {"X": X, "y": y, "protected": protected, "feature_names": list(X.columns),
            "feature_groups": feature_groups_from_columns(X.columns),
            "cost_fn": 5.0, "cost_fp": 1.0, "name": "german_credit"}


def load_gmsc() -> dict:
    """Give Me Some Credit. Download cs-training.csv into data/ (Kaggle) if absent; target =
    SeriousDlqin2yrs (1=default). All-numeric, so each column is its own feature group. No protected
    attributes shipped -> the faithfulness generalization check only, not the governance dossier."""
    path = DATA / "cs-training.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Place the Give Me Some Credit cs-training.csv at {path} "
            "(download from the public Kaggle dataset 'GiveMeSomeCredit').")
    df = pd.read_csv(path, index_col=0)
    y = df.pop("SeriousDlqin2yrs").astype(int)
    # keep NaN — median imputation is fit on the TRAIN split only (30_faithfulness), never full-data.
    # NB: the audited GMSC benchmark parquet is produced by scripts/06_gmsc_prep.py (sentinel cleaning
    # + stratified subsample), not by this convenience loader.
    return {"X": df, "y": y, "feature_names": list(df.columns),
            "feature_groups": feature_groups_from_columns(df.columns), "name": "gmsc"}
