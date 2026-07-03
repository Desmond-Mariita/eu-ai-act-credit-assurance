# tests/test_faithfulness.py
import numpy as np

from credit_assurance import faithfulness as f
from credit_assurance import perturbation as p

# Same toy space as test_perturbation: "num" (col 0) is the ONLY predictive group; "cat" is inert.
FG = {"num": (0,), "cat": (1, 2, 3)}
ALL = ["num", "cat"]


def _donor_pool(n=80, seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)
    cat = np.zeros((n, 3))
    cat[np.arange(n), rng.integers(0, 3, size=n)] = 1.0
    return np.column_stack([num, cat])


X = _donor_pool()
X0 = np.array([2.0, 1.0, 0.0, 0.0])
W = np.array([3.0, 0.0, 0.0, 0.0])   # prediction depends only on "num"


def _predict(A):
    return 1.0 / (1.0 + np.exp(-(A @ W)))


def _perturber(seed=0):
    return p.make_perturber(X, FG, regime="conditional", seed=seed, k_neighbors=10)


def test_comprehensiveness_higher_for_informative_ranking():
    pert = _perturber()
    true = f.comprehensiveness(_predict, X0, ["num", "cat"], ks=[1], perturber=pert, m=30)
    bad = f.comprehensiveness(_predict, X0, ["cat", "num"], ks=[1], perturber=pert, m=30)
    assert true[1] > 0.1                 # erasing the important group drops P(bad)
    assert abs(bad[1]) < 0.05            # erasing the inert group barely moves P
    assert true[1] > bad[1]              # a faithful ranking beats a non-informative one


def test_negative_control_ranking_lands_near_floor():
    pert = _perturber()
    bad = f.comprehensiveness(_predict, X0, ["cat", "num"], ks=[1], perturber=pert, m=30)
    floor = f.random_floor(_predict, X0, ALL, ks=[1], perturber=pert, m=30, n_perms=25, seed=0)
    # a non-informative (here: inert-first) attribution scores no better than the random floor
    assert bad[1] <= floor[1] + 0.05


def test_sufficiency_small_when_top_group_carries_signal():
    pert = _perturber()
    suff = f.sufficiency(_predict, X0, ["num", "cat"], ALL, ks=[1], perturber=pert, m=30)
    assert abs(suff[1]) < 0.1            # keeping "num" retains the prediction


def test_aopc_is_mean_of_curve():
    assert np.isclose(f.aopc({1: 0.2, 2: 0.4}), 0.3)


def test_bootstrap_ci_brackets_the_mean():
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mean, lo, hi = f.bootstrap_ci(arr, n_boot=500, seed=0)
    assert lo <= mean <= hi
