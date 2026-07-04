import numpy as np
import pandas as pd

from credit_assurance import data as d


def test_sex_derived_from_personal_status():
    s = pd.Series(["A91", "A92", "A93", "A94", "A95"])  # Statlog codes
    sex = d.sex_from_personal_status(s)
    assert list(sex) == ["male", "female", "male", "male", "female"]


def test_cost_sensitive_threshold_uses_5to1_matrix():
    # Statlog: predicting good when actually bad costs 5; bad-when-good costs 1.
    # optimal threshold on P(bad) = cost_fp/(cost_fp+cost_fn) framing
    thr = d.cost_sensitive_threshold(cost_fn=5.0, cost_fp=1.0)
    assert np.isclose(thr, 1.0 / 6.0)


def test_protected_attributes_from_encoded():
    df = pd.DataFrame({
        "Attribute9_A91": [1, 0, 0], "Attribute9_A92": [0, 1, 0],
        "Attribute9_A93": [0, 0, 1], "Attribute9_A94": [0, 0, 0],
        "Attribute20_A201": [1, 0, 1], "Attribute20_A202": [0, 1, 0],
        "Attribute13": [22, 45, 67],
    })
    p = d.protected_attributes_from_encoded(df)
    assert list(p["sex"]) == ["male", "female", "male"]           # A91,A92,A93
    assert list(p["foreign_worker"]) == ["yes", "no", "yes"]
    assert list(p["age_band"]) == ["<25", "40-59", "60+"]


def test_feature_groups_from_columns_groups_one_hot_atomically():
    cols = ["Attribute2", "Attribute1_A11", "Attribute1_A12", "Attribute1_A13", "Attribute3_A30"]
    g = d.feature_groups_from_columns(cols)
    assert g["Attribute2"] == (0,)              # numeric singleton
    assert g["Attribute1"] == (1, 2, 3)         # one-hot dummies grouped
    assert g["Attribute3"] == (4,)
    # full coverage, no overlap
    covered = sorted(i for idxs in g.values() for i in idxs)
    assert covered == list(range(len(cols)))
