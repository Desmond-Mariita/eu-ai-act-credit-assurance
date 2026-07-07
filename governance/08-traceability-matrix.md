# 08 — Traceability Matrix (claim → evidence → reproduction)

**Purpose:** every load-bearing claim in the audit resolves to a content-addressed artifact **and** a
one-command reproduction, so an independent reviewer can re-perform it. Complements `09-evidence-index`
(artifact hashes) and the finding docs. Environment: `uv` + Python 3.12, all seeds = 0, audited model
SHA pinned in `00-scenario.md §5`. **Self-assessment** (see `13 §E`).

| ID | Claim (as stated in the finding) | Source | Evidence | Reproduce | Obligation |
|---|---|---|---|---|---|
| **MODEL-1** | Audited LightGBM: test **AUROC 0.769**, Brier 0.178, cost-threshold 1/6, **deterministic** (SHA `1456b07f…`); EBM challenger 0.794 | `00 §5` | EV-007/008, `models.json` | `python scripts/10_train.py` | Art. 10/11/15 |
| **DATA-1** | Data manifest + DQ profile (GMSC sentinels 96/98, age 0; foreign-worker skew 963/37) | `data-dictionary` | EV-001/002/003 | `python scripts/05_manifest_and_dq.py` | Art. 10 |
| **PREREG-1** | Method + hypotheses fixed **before results**; signed tag `prereg-v1` + OpenTimestamps | `HYPOTHESES.md`, `09` proof | tag `prereg-v1`, `HYPOTHESES.md.ots` | `git verify-tag prereg-v1` | integrity |
| **FAITH-1** | Movement metric (exploratory; clean control at floor = calibration): **TreeSHAP +0.106 [.094,.118]**, **LIME +0.090 [.080,.100]** both beat floor; **TreeSHAP more consistent** (signed: TS inconclusive, LIME below floor; baseline: TreeSHAP only) | `10 §Results` | EV-009 | `python scripts/30_faithfulness.py` | Art. 86 |
| **FAITH-2** | Signed on-manifold test (German): **TreeSHAP inconclusive** (CI spans 0) while **LIME separates *below* floor** (CI excludes 0) — **not a uniform null**; sign cancellation + low n | `10` | EV-009 | ↑ | Art. 86 |
| **FAITH-3** | Core movement result **replicates on GMSC**; signed null is dataset-specific | `10 §Generalization` | EV-011 | `python scripts/06_gmsc_prep.py && python scripts/30_faithfulness.py --dataset gmsc` | — |
| **FAITH-4** | **ROAR** (8 splits, *global* ranking utility): TreeSHAP 0.108±0.017 ≈ LIME 0.109±0.021 (indistinguishable on a 2·SE heuristic), both ≫ random 0.036 | `10 §ROAR` | EV-015 | `python scripts/70_roar.py` | Art. 86 |
| **FAIR-1** | **No disparity significant at n=300** (sex FPR within-negatives permutation **p=0.091**, Fisher 0.096; age omnibus p=0.39); folded DP/EO CIs are not a significance test | `11` | EV-012 | `python scripts/40_fairness.py` | Art. 10 |
| **FAIR-2** | Model **trains on personal-status (sex proxy), age, and foreign-worker status directly** | `11` | EV-012 | ↑ | Art. 10 |
| **ROB-1** | Decision-flip **3.8–16.7%** (eps 0.05–0.5, bootstrap CIs; independent per-eps RNG streams); **13.3% near-threshold** | `12 §Robustness` | EV-013 | `python scripts/50_robustness.py` | Art. 15 |
| **REC-1** | **87.4% [81.2, 91.8]** actionable loan-term **reduction** recourse; **12.6% (19/151) reduction-infeasible** (of which **4 flip via *increase*** — perverse non-monotonicity, NOT "non-actionable factors"); reason codes duration-leaning | `12 §Recourse` | EV-013 | `python scripts/60_reason_codes.py` | Art. 86 (recourse = good practice, not a GDPR 13-15 test) |

## Notes
- **Tag verification (self-contained):** the repo ships the signer trust anchor at
  [`.allowed_signers`](../.allowed_signers) (author's Ed25519 public key + `SHA256:lr+8ntekiE/WkJo4/SUOEmoMdaBJ9XMWHyIEOLGLuUo`
  fingerprint). Verify both signed tags in one line:
  `git -c gpg.ssh.allowedSignersFile=.allowed_signers verify-tag prereg-v1 v1.1.1` (all of prereg-v1 /
  v1.0 / v1.1 / v1.1.1 verify; also run by CI). The
  key is published in-repo *and* in the commit history; an independent reviewer should confirm the
  fingerprint through a second channel before trusting the identity. The OTS proof is **Bitcoin-confirmed
  (block 956533)** (`09`).
- **Hashing:** data + metrics artifacts carry SHA256 prefixes in `09`; living governance docs are
  content-addressed by git (blob SHA at tag `v1.1.1`); the pre-registration is OpenTimestamps
  **Bitcoin-confirmed (block 956533)**.
- **Determinism:** every script above re-runs to byte-identical output under the fixed seed; the
  recourse/robustness/ROAR/fairness scripts load and **SHA-verify** the pinned model
  (`models/lightgbm.txt`) or verify predictions identical to it.
- **Not covered here (see `13 §E`):** obligations with no produced artifact (QMS, logging,
  cybersecurity, post-market monitoring, CE/registration) have no traceability row because no claim is
  made about them. (A *template* DPIA exists — EV-016 — but a real operational DPIA is a Gap.)
