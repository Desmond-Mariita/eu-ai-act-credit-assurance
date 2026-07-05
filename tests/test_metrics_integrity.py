# tests/test_metrics_integrity.py
"""Guards against committing a SMOKE-TEST metrics artifact instead of a full run.

Motivated by audit finding A1: a `--n-eval 2` smoke run once overwrote the flagship
`faithfulness_german_credit.json` and was committed, contradicting the headline. These asserts run in
CI (the metrics JSONs are committed) and fail loudly if any published artifact is a debug-scale run.
"""
import hashlib
import json
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "metrics"


def _load(name):
    return json.loads((METRICS / name).read_text())


def _sha8(name):
    return hashlib.sha256((METRICS / name).read_bytes()).hexdigest()[:8]


def test_metrics_are_full_runs_not_smoke():
    assert _load("faithfulness_german_credit.json")["params"]["n_eval"] >= 100
    assert _load("faithfulness_gmsc.json")["params"]["n_eval"] >= 100
    assert _load("roar_german_credit.json")["n_splits"] >= 5
    assert _load("reason_codes_german_credit.json")["n_declined"] >= 100
    assert _load("robustness_german_credit.json")["n_test"] >= 100
    assert _load("fairness_german_credit.json")["n_test"] >= 100


def test_audited_model_metrics_present():
    m = _load("models.json")["audited_model"]
    assert m["sha256"] == "1456b07fb365dcbc91d6b139c51e1269976509249bdb8d7d4b4cd8e48ab5eab5"
    assert m["auroc"] >= 0.7


def test_metrics_hashes_match_evidence_index():
    # sha256 prefixes recorded in governance/09-evidence-index.md (EV-008/009/011/012/013/015):
    # any content drift fails here, forcing the EV index + docs to be updated in lockstep.
    expected = {
        "models.json": "8ef65b5d",
        "faithfulness_german_credit.json": "ce746733",
        "faithfulness_gmsc.json": "2097fe27",
        "fairness_german_credit.json": "954d5bb6",
        "robustness_german_credit.json": "2941addc",
        "reason_codes_german_credit.json": "a0b17b9d",
        "roar_german_credit.json": "ea08b093",
    }
    for name, want in expected.items():
        assert _sha8(name) == want, f"{name} sha {_sha8(name)} != EV {want}; update EV index + this guard"
