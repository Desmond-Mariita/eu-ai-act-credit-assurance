import numpy as np, pandas as pd
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
