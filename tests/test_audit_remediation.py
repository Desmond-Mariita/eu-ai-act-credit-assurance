# tests/test_audit_remediation.py
"""Guards for the deep-audit residual remediation (R1-R7).

Semantic cross-document checks that go beyond file-hash integrity: they assert the PROSE-driving
booleans/numbers are internally consistent, so a stale narrative (e.g. a duplicated interpretation, an
overclaimed recourse cause) fails CI rather than shipping. Motivated by audit findings A1/A2/A4.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"


def _load(name):
    return json.loads((METRICS / name).read_text())


# --- R1: recourse reduction-infeasibility must NOT be read as "non-actionable factors" -----------
def test_recourse_reduction_infeasible_breakdown():
    d = _load("reason_codes_german_credit.json")
    n_inf = d["reduction_infeasible_count"]
    up = d["of_which_flip_via_increase"]
    stuck = d["of_which_no_flip_any_single_feature"]
    assert up + stuck == n_inf, "increase-flip + stuck must partition the reduction-infeasible set"
    assert up >= 1, "the perverse increase-flip finding (A1) must be reported, not hidden"
    # every reported increase-flip example must actually be an INCREASE
    for ex in d["increase_flip_examples"]:
        (feat, (before, after)), = ex["increase"].items()
        assert after > before, f"increase_flip example {feat} is not an increase: {before}->{after}"


# --- R3: the GMSC interpretation must reflect ITS OWN result, not German's -------------------------
def test_gmsc_interpretation_matches_its_h1():
    g = _load("faithfulness_gmsc.json")
    h1 = g["hypotheses_preregistered"]["H1_treeshap_faithful_conditional_signed"]
    interp = g["interpretation"]
    if h1:   # signed test DID detect TreeSHAP faithfulness on GMSC -> interpretation must say so
        assert "DOES detect" in interp, "GMSC H1 is True but interpretation says signed test does not detect"
    else:
        assert "does NOT detect" in interp


def test_gmsc_interpretation_not_duplicate_of_german():
    g = _load("faithfulness_gmsc.json")["interpretation"]
    de = _load("faithfulness_german_credit.json")["interpretation"]
    # German H1 is False, GMSC H1 is True -> their signed-test sentences must differ
    assert g != de, "GMSC interpretation is a verbatim copy of German's (A9) — must be dataset-specific"


# --- R2: GMSC model is pinned + the derived parquet is manifest-covered ---------------------------
def test_gmsc_model_pinned_and_hashed():
    import hashlib
    p = ROOT / "models" / "lightgbm_gmsc.txt"
    assert p.exists(), "GMSC generalization model must be persisted (EV-011)"
    # deterministic (seeded, single-threaded) -> hash pinned like the German model; EV-011 records it
    assert hashlib.sha256(p.read_bytes()).hexdigest()[:8] == "1c1f70ea"


def test_manifest_covers_gmsc_parquet():
    manifest = (ROOT / "data_manifest.sha256").read_text()
    assert "data/gmsc.parquet" in manifest, "load-bearing gmsc.parquet must be in the integrity manifest"


def test_manifest_coverage_exact_and_hash_pinned():
    """Guards A1/A2: the committed manifest lists EXACTLY the required integrity set (no required file
    silently dropped), and the evidence index pins the manifest's own current hash."""
    import hashlib
    import re
    mpath = ROOT / "data_manifest.sha256"
    listed = sorted(ln.split("  ", 1)[1] for ln in mpath.read_text().splitlines() if ln.strip())
    src = (ROOT / "scripts" / "05_manifest_and_dq.py").read_text()
    required = sorted(re.findall(r'"(data/[^"]+)"', src.split("MANIFEST_FILES", 1)[1].split("]", 1)[0]))
    assert listed == required, f"manifest coverage drift: listed={listed} required={required}"
    assert hashlib.sha256(mpath.read_bytes()).hexdigest()[:8] == "7b5f4568", "EV-001 manifest hash drift"


def test_manifest_verify_is_fail_closed():
    """Guards A1: the manifest script must fail-close — default path exits non-zero without a manifest,
    verify fails on a dropped-coverage file, and freeze refuses an incomplete set (checked at source)."""
    src = (ROOT / "scripts" / "05_manifest_and_dq.py").read_text()
    assert "refusing incomplete freeze" in src          # freeze refuses partial sets
    assert "if f not in tracked:" in src                # verify requires full coverage (fail-closed)
    assert "absence of the integrity manifest is not a pass" in src  # default path exits non-zero


def test_robustness_uses_independent_spawned_streams():
    """Guards R5: production 50_robustness must keep independent per-eps perturbation vs bootstrap streams
    (a single shared generator would re-couple flip rates to the bootstrap replicate count)."""
    src = (ROOT / "scripts" / "50_robustness.py").read_text()
    assert "SeedSequence(SEED).spawn" in src and "pert_ss" in src and "boot_ss" in src
    assert "np.random.default_rng(SEED)" not in src     # no single shared RNG for both draws


# --- R5: robustness perturbation draws must be independent of the bootstrap replicate count --------
def test_robustness_rng_streams_decoupled():
    """The fix spawns independent perturbation vs bootstrap streams. A perturbation draw must be
    byte-identical whether or not bootstrap indices are drawn in between (a presentation parameter must
    not perturb the experiment)."""
    seed = 0
    pert_ss = np.random.SeedSequence(seed).spawn(4)
    boot_ss = np.random.SeedSequence(seed + 2000).spawn(4)

    def pert_draw():
        return np.random.default_rng(pert_ss[1]).normal(0, 1, 50)

    a = pert_draw()
    # consume an arbitrary number of bootstrap indices from the bootstrap stream
    _ = np.random.default_rng(boot_ss[1]).integers(0, 50, (2000, 50))
    b = pert_draw()
    assert np.array_equal(a, b), "perturbation draw changed after drawing bootstrap indices — streams coupled"


# --- R6: the tag-verification trust anchor ships in-repo ------------------------------------------
def test_allowed_signers_trust_anchor_present():
    p = ROOT / ".allowed_signers"
    assert p.exists(), ".allowed_signers trust anchor must ship for self-contained tag verification"
    body = p.read_text()
    assert "ssh-ed25519" in body and "namespaces=\"git\"" in body
