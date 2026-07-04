# 13 — Conformity-Readiness Dossier (Annex IV technical documentation)

**Status:** synthesis of the audit (consolidates `00`, `02`, `09`, `data-dictionary`, findings `10`–
`12`, `metrics/`). **Nature:** a **conformity-READINESS demonstration and self-assessment — NOT an EU
declaration of conformity** and **not** a legal conformity assessment. No opinion is expressed that
the system is or is not compliant; this maps the evidence produced to the obligations and **states the
gaps honestly**. Criteria edition-pinned in `00-scenario.md §4`. Target of Evaluation: the audited,
SHA-pinned LightGBM credit-scoring model (`00-scenario.md §5`).

## A. Annex IV (Art. 11) technical-documentation elements
| # | Annex IV element | Where / evidence | Status |
|---|---|---|---|
| 1 | General description (intended purpose, provider, versions, use) | `00-scenario.md §1–3` | Ready |
| 2 | Development elements & process (data, architecture, training, validation, metrics, oversight assessment) | `data-dictionary.md`, `metrics/models.json`, `scripts/10_train.py`, `00-scenario.md §5` | **Partial** — cybersecurity + full datasheet not produced |
| 3 | Monitoring, functioning & control (capabilities/limitations, accuracy, foreseeable unintended outcomes, human oversight, input specs) | `metrics/models.json`, findings `10`–`12`, `00-scenario.md §2` | **Partial** — oversight effectiveness not tested |
| 4 | Appropriateness of performance metrics | `metrics/models.json` (AUROC/Brier/cost-threshold), `10`–`12` | Ready |
| 5 | Risk management system (Art. 9) | `02-risk-management-file.md` | **Partial** — living register, not full lifecycle |
| 6 | Lifecycle changes | git history; pre-registration `prereg-v1` | Ready (static study) |
| 7 | Harmonised standards applied | `00-scenario.md §4` (ISO/IEC 42001, NIST AI RMF, BCBS 239) | Ready (referenced) |
| 8 | EU declaration of conformity | — | **Not issued** (readiness only, by design) |
| 9 | Post-market monitoring system (Art. 72) | — | **Gap** (not produced) |

## B. Article-by-article conformity-readiness matrix
Status legend: **Ready** = evidence produced & reviewed · **Partial** = partially evidenced /
design-level · **Gap** = not produced · **Scope** = deliberately out of scope (`00-scenario.md §6`).

| Article | Obligation (short) | Evidence | Status |
|---|---|---|---|
| 6(2)/6(3) | High-risk (Annex III §5(b)) + exemption analysed & rejected | `00-scenario.md §4` | Ready |
| 9 | Risk management system | `02-risk-management-file.md` | Partial |
| 10 | Data governance; examination for bias | `data-dictionary.md`, `11-fairness-findings.md` | Partial |
| 11 / Annex IV | Technical documentation | **this dossier** | Partial |
| 12 | Logging / automatic record-keeping | `00-scenario.md §2` (design) | **Gap** (no logging impl/test) |
| 13 | Transparency & instructions for use | `12` (reason codes), `metrics/models.json` (card) | Partial (no IFU document) |
| 14 | Human oversight | `00-scenario.md §2` (loan-officer review) | Partial (design; effectiveness untested) |
| 15 | Accuracy, robustness, cybersecurity | `metrics/models.json`, `12-robustness-and-recourse.md` | Partial (robustness synthetic; **no security**) |
| 27 | Fundamental-rights impact assessment (FRIA) | `11-fairness-findings.md` + this | Partial (full FRIA not produced) |
| 26 | Deployer obligations | `00-scenario.md §3` (roles) | Scope (deployer) |
| 72 / 73 | Post-market monitoring / incident reporting | — | **Gap** |
| GDPR 13–15 | Meaningful information / recourse | `12` (recourse), `10` (faithfulness) | Partial |
| GDPR 22(3) | Automated-decision safeguards | `00-scenario.md §2` (human review) | Partial (Art. 22 trigger analysis deferred) |
| GDPR 35 | DPIA | — | **Gap** (`06-gdpr-dpia.md` not written) |

## C. Consolidated findings (the four analyses — reviewed)
1. **Faithfulness (`10`, EV-009/011):** explanation faithfulness is **metric-dependent**; under a
   *validated* direction-agnostic movement metric both **TreeSHAP and LIME are faithful** (TreeSHAP >
   LIME); the pre-registered signed on-manifold test is an **underpowered null** (sign cancellation),
   not evidence of unfaithfulness; **replicated on GMSC**. *Governance: faithfulness claims must name
   the regime + sign convention (Art. 86 / GDPR 13–15).*
2. **Fairness (`11`, EV-012):** disparities are **directionally consistent** (women/young/foreign
   declined more) **but none is statistically significant at n=300** (signed CIs include 0; sex FPR
   permutation p=0.086); the model **trains on personal-status (sex proxy) + age directly**. *Monitoring
   flags, not established gaps (Art. 10).*
3. **Robustness (`12`, EV-013):** moderately stable — **3.6–16.3%** decision flips as noise grows;
   **13.3% near-threshold** fragile band. *Art. 15 concern to monitor.*
4. **Recourse (`12`, EV-013):** **94.0% [89.1, 96.8]** of declined applicants have loan-term recourse
   (mostly a shorter term or lower amount, ~even); a **~6% infeasible core** rests on non-actionable
   factors and needs a genuine explanation. *GDPR 13–15.*

## D. Traceability (claim → evidence)
Every quantitative claim above resolves to an `EV-###` row in `09-evidence-index.md` with a content
hash and the producing script; the pre-registration (`HYPOTHESES.md`, tag `prereg-v1`, OpenTimestamps)
fixes the faithfulness method before results. A standalone `08-traceability-matrix.md` is **not yet
produced**; this section + the evidence index serve as the interim spine.

## E. Residual gaps & open risks (honest register)
- **Not produced:** GDPR DPIA (`06`), standalone traceability matrix (`08`), logging/record-keeping
  (Art. 12), cybersecurity assessment (Art. 15), post-market monitoring (Art. 72), instructions-for-use
  (Art. 13), a full FRIA (Art. 27), human-oversight *effectiveness* testing (Art. 14).
- **Method limits:** single stratified split, small n (≈300 test; subgroup claims low-powered);
  robustness noise is synthetic (not calibrated to real error logs); recourse is model-side
  (real-world feasibility unassessed); one model, mainline dataset + one generalization check.
- **Independence:** **self-assessment** — the author built the audited model and performs the
  assessment; the "externally reviewed" claim is only made once ≥1 external peer review signs off
  (plan Task 24, pending). Method + code hardened through internal + external model-reviewer gauntlets,
  which is **not** a substitute for human independent review.

## F. Overall readiness verdict
Against the pinned criteria: **Ready** on the classification, metrics, and documentation-structure
elements; **Partial** on risk management, data governance, transparency, human oversight, accuracy/
robustness, and FRIA; **Gap** on DPIA, logging, cybersecurity, and post-market monitoring. This is a
**credible conformity-readiness demonstration on the modelling/assurance dimensions**, explicitly
**not** a conformity declaration and **not** independent. The signed audit opinion (`14`, next) draws
its scope, findings, and limitations from this dossier.
