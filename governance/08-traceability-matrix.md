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
| **FAITH-1** | Movement metric **validated** (clean control at floor); **TreeSHAP +0.106 [.094,.118]**, **LIME +0.076 [.066,.086]** both faithful; TreeSHAP > LIME *under this metric* (ROAR: indistinguishable) | `10 §Results` | EV-009 | `python scripts/30_faithfulness.py` | Art. 86 / GDPR 13-15 |
| **FAITH-2** | Signed on-manifold test = **underpowered null** (H1 refuted; sign cancellation) | `10` | EV-009 | ↑ | Art. 86 |
| **FAITH-3** | Core movement result **replicates on GMSC**; signed null is dataset-specific | `10 §Generalization` | EV-011 | `python scripts/06_gmsc_prep.py && python scripts/30_faithfulness.py --dataset gmsc` | — |
| **FAITH-4** | **ROAR** (8 splits): TreeSHAP 0.108±0.017 ≈ LIME 0.111±0.016 (**indistinguishable**), both ≫ random 0.036 | `10 §ROAR` | EV-015 | `python scripts/70_roar.py` | Art. 86 |
| **FAIR-1** | **No disparity significant at n=300** (sex FPR-diff CI includes 0, permutation **p=0.086**); folded DP/EO CIs are not a significance test | `11` | EV-012 | `python scripts/40_fairness.py` | Art. 10 |
| **FAIR-2** | Model **trains on personal-status (sex proxy) + age directly** | `11` | EV-012 | ↑ | Art. 10 |
| **ROB-1** | Decision-flip **3.6–16.3%** (eps 0.05–0.5, bootstrap CIs); **13.3% near-threshold** | `12 §Robustness` | EV-013 | `python scripts/50_robustness.py` | Art. 15 |
| **REC-1** | **94.0% [89.1, 96.8]** loan-term recourse; **~6% infeasible** core; reason codes ≈ even (duration/amount) | `12 §Recourse` | EV-013 | `python scripts/60_reason_codes.py` | GDPR 13-15 / Art. 86 |

## Notes
- **Tag verification:** `git verify-tag prereg-v1` requires the verifier to configure an SSH
  `gpg.ssh.allowedSignersFile` with the author's public key; the OTS proof is **Bitcoin-confirmed (block 956533)**
  (`09`).
- **Hashing:** data + metrics artifacts carry SHA256 prefixes in `09`; living governance docs are
  content-addressed at the v1 release freeze; the pre-registration is OpenTimestamps **Bitcoin-confirmed
  (block 956533)**.
- **Determinism:** every script above re-runs to byte-identical output under the fixed seed; the
  recourse/robustness/ROAR/fairness scripts load and **SHA-verify** the pinned model
  (`models/lightgbm.txt`) or verify predictions identical to it.
- **Not covered here (see `13 §E`):** obligations with no produced artifact (DPIA, QMS, logging,
  cybersecurity, post-market monitoring, CE/registration) have no traceability row because no claim is
  made about them.
