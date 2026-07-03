# tests/test_explainers.py
import numpy as np

from credit_assurance import explainers as E

FG = {"a": (0,), "b": (1, 2), "c": (3,)}
NAMES = ["a", "b", "c"]


def test_aggregate_sums_member_columns():
    col = np.array([0.5, 0.2, 0.3, -0.4])
    assert E.aggregate_to_logical(col, FG, NAMES) == {"a": 0.5, "b": 0.5, "c": -0.4}


def test_rank_by_absolute_value():
    assert E.rank_groups({"a": 0.1, "b": -0.9, "c": 0.5}) == ["b", "c", "a"]


def test_random_ranking_is_permutation_and_seed_stable():
    r1, _ = E.random_ranking(NAMES, seed=3)
    r2, _ = E.random_ranking(NAMES, seed=3)
    assert sorted(r1) == sorted(NAMES) and r1 == r2


def test_shuffled_shap_control_is_permutation_and_seed_stable():
    class FakeShap:
        def shap_values(self, Xarr):
            return np.array([[0.9, 0.1, 0.0, -0.5]])   # (1, d)

    r1, _ = E.shuffled_shap_ranking(FakeShap(), np.zeros(4), FG, NAMES, seed=2)
    r2, _ = E.shuffled_shap_ranking(FakeShap(), np.zeros(4), FG, NAMES, seed=2)
    assert sorted(r1) == sorted(NAMES) and r1 == r2
