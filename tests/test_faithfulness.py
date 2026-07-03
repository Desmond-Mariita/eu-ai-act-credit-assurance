# tests/test_faithfulness.py
import numpy as np

from credit_assurance import faithfulness as f
from credit_assurance import perturbation as p

# Toy space: "num" (col 0) is the ONLY predictive group; "cat" (one-hot cols 1-3) is inert.
FG = {"num": (0,), "cat": (1, 2, 3)}
ALL = ["num", "cat"]


def _donor_pool(n=120, seed=0):
    rng = np.random.default_rng(seed)
    cat = np.zeros((n, 3))
    cat[np.arange(n), rng.integers(0, 3, size=n)] = 1.0
    return np.column_stack([rng.normal(size=n), cat])


X = _donor_pool()
X0 = np.array([2.0, 1.0, 0.0, 0.0])
W = np.array([3.0, 0.0, 0.0, 0.0])   # prediction depends only on "num"


def _predict(A):
    return 1.0 / (1.0 + np.exp(-(A @ W)))


def _perturber(seed=0):
    return p.make_perturber(X, FG, regime="conditional", seed=seed, k_neighbors=15)


def test_informative_ranking_beats_random_floor():
    pert = _perturber()
    true = f.comprehensiveness(_predict, X0, ["num", "cat"], ks=[1], perturber=pert, m=40)
    floor = f.random_floor(_predict, X0, ALL, ks=[1], perturber=pert, m=40, n_perms=100, seed=0)
    assert true[1] > 0.1                 # erasing the important group drops P(bad)
    assert true[1] > floor[1] + 0.05     # a faithful ranking beats the non-informative baseline


def test_non_informative_ranking_no_better_than_floor():
    # negative control: an inert-first ranking scores no higher than the random floor
    pert = _perturber()
    bad = f.comprehensiveness(_predict, X0, ["cat", "num"], ks=[1], perturber=pert, m=40)
    floor = f.random_floor(_predict, X0, ALL, ks=[1], perturber=pert, m=40, n_perms=100, seed=0)
    assert abs(bad[1]) < 0.05            # erasing the inert group barely moves P
    assert bad[1] <= floor[1] + 1e-6     # not above the floor


def test_sufficiency_small_when_top_group_carries_signal():
    pert = _perturber()
    suff = f.sufficiency(_predict, X0, ["num", "cat"], ALL, ks=[1], perturber=pert, m=40)
    assert abs(suff[1]) < 0.1            # keeping "num" retains the prediction


def test_aopc_is_mean_of_curve():
    assert np.isclose(f.aopc({1: 0.2, 2: 0.4}), 0.3)


def test_absolute_comprehensiveness_is_magnitude_of_signed():
    pert = _perturber()
    signed = f.comprehensiveness(_predict, X0, ["num", "cat"], ks=[1, 2], perturber=pert, m=20)
    absol = f.comprehensiveness(_predict, X0, ["num", "cat"], ks=[1, 2], perturber=pert, m=20,
                                absolute=True)
    for k in (1, 2):
        assert absol[k] >= 0 and np.isclose(absol[k], abs(signed[k]))   # deterministic per-call RNG


def test_bootstrap_ci_brackets_the_mean():
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mean, lo, hi = f.bootstrap_ci(arr, n_boot=500, seed=0)
    assert lo <= mean <= hi
