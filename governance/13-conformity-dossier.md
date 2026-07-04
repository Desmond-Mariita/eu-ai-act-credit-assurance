# 13 — Conformity-Readiness Dossier (Annex IV technical documentation)

**Status:** synthesis of the audit (consolidates `00`, `02`, `09`, `data-dictionary`, findings `10`–
`12`, `metrics/`). **Nature:** a **conformity-READINESS demonstration and self-assessment — NOT an EU
declaration of conformity** and **not** a legal conformity assessment. No opinion is expressed that
the system is or is not compliant; this maps the evidence produced to the obligations and **states the
gaps honestly**. Criteria edition-pinned in `00-scenario.md §4`. Target of Evaluation: the audited,
SHA-pinned LightGBM credit-scoring model (`00-scenario.md §5`).

**Applicable conformity-assessment route:** as an Annex III §5(b) high-risk system, the route is
**internal control (Annex VI)** per **Art. 43(2)** — the provider self-assesses; **no notified body**
is involved. This dossier is readiness material for that route, not its completion.

## A. Annex IV (Art. 11) technical-documentation elements
| # | Annex IV element | Where / evidence | Status |
|---|---|---|---|
| 1 | General description (intended purpose, provider, versions, use, **instructions for use, UI**) | `00-scenario.md §1–3` | **Partial** — no instructions-for-use / UI |
| 2 | Development elements & process (data, architecture, training, validation, metrics, oversight, **cybersecurity**) | `data-dictionary.md`, `metrics/models.json`, `scripts/10_train.py`, `00-scenario.md §5` | **Partial** — cybersecurity + full datasheet not produced |
| 3 | Monitoring, functioning & control (capabilities/limitations, accuracy, unintended outcomes, oversight, input specs) | `metrics/models.json`, findings `10`–`12`, `00-scenario.md §2` | **Partial** — oversight effectiveness not tested |
| 4 | Appropriateness of performance metrics | `metrics/models.json`, `10`–`12` | **Partial** — metrics reported; selection rationale thin |
| 5 | Risk management system (Art. 9) | `02-risk-management-file.md` | **Partial** — living register, not full lifecycle |
| 6 | Lifecycle changes | git history; pre-registration `prereg-v1` | Ready (static study) |
| 7 | **Standards/frameworks referenced** | `00-scenario.md §4` (ISO/IEC 42001:2023, NIST AI RMF 1.0, BCBS 239) | **Partial** — **none is an EU *harmonised* standard** (Art. 40 / Reg. 1025/2012); no harmonised AI standards adopted yet → **no presumption of conformity** |
| 8 | EU declaration of conformity | — | **Not issued** (readiness only, by design) |
| 9 | Post-market monitoring system (Art. 72) | — | **Gap** (not produced) |

## B. Article-by-article readiness matrix
Status legend: **Ready** = evidence produced & reviewed · **Partial** = partially evidenced /
design-level · **Gap** = not produced · **Scope** = deliberately out of scope (deployer duty or
`00-scenario.md §6`).

| Article | Obligation (short) | Evidence | Status |
|---|---|---|---|
| 6(2) | High-risk classification (Annex III §5(b)) | `00-scenario.md §4` | Ready |
| 6(3) | No exemption — credit scoring **profiles natural persons**, so the Art. 6(3) derogation is categorically unavailable (final subparagraph) | `00-scenario.md §4` | Ready |
| 9 | Risk management system | `02-risk-management-file.md` | Partial |
| 10 | Data governance; examination for bias | `data-dictionary.md`, `11-fairness-findings.md` | Partial |
| 11 / Annex IV | Technical documentation | **this dossier** | Partial |
| 12 | Logging / automatic record-keeping | `00-scenario.md §2` (design) | **Gap** (no logging impl/test) |
| 13 | Transparency & instructions for use | `12` (reason codes), model card | Partial (no IFU document) |
| 14 | Human oversight | `00-scenario.md §2` (loan-officer review) | Partial (design; effectiveness untested) |
| 15 | Accuracy, robustness, **cybersecurity** | `metrics/models.json`, `12-robustness-and-recourse.md` | Partial (robustness synthetic; **no security assessment**) |
| 16 | Provider obligations (umbrella; incl. Art. 18 doc-retention, 20 corrective action, 21 cooperation — not itemised) | via `9`–`15`, `17` | Partial |
| 17 | Quality management system | — | **Gap** |
| 43(2) / Annex VI | Conformity-assessment route = internal control (no notified body) | this dossier (readiness) | Partial |
| 48 | CE marking | — | **Gap** (post-conformity step) |
| 49 | Registration in the EU database | — | **Gap** |
| 26 | Deployer obligations | `00-scenario.md §3` (roles) | Scope (deployer) |
| 27 | Fundamental-rights impact assessment (FRIA) — **a deployer duty** | `11-fairness-findings.md` informs it | Scope (deployer; not produced) |
| 72 / 73 | Post-market monitoring / serious-incident reporting | — | **Gap** |
| 86 | Right to explanation of individual decision-making (**deployer-facing duty**; the *capability* is provider-side) | `12` (recourse), `10` (faithfulness) | Partial |
| GDPR 5 | Data-processing principles | `00-scenario.md §4` | Partial (referenced) |
| GDPR 6/9 | Lawful basis / special-category data | `data-dictionary.md`, `11` | Scope/Partial — public research data; **sex/age are NOT GDPR Art. 9 special categories**, and no Art. 9 data is processed (sex is inferred as a proxy from personal-status for the fairness test only) |
| GDPR 13–15 | Meaningful information / recourse | `12`, `10` | Partial |
| GDPR 22(3) | Automated-decision safeguards | `00-scenario.md §2` (human review) | Partial (Art. 22 trigger analysis deferred) |
| GDPR 35 | DPIA | — | **Gap** (`06-gdpr-dpia.md` not written) |

## C. Consolidated findings (the four analyses — reviewed)
1. **Faithfulness (`10`, EV-009/011):** explanation faithfulness is **metric-dependent**; under a
   *validated* direction-agnostic movement metric both **TreeSHAP and LIME are faithful** (TreeSHAP >
   LIME); the pre-registered signed on-manifold test is an **underpowered null** (sign cancellation),
   not evidence of unfaithfulness. **The core movement-metric result replicated on GMSC** (the signed
   null did *not* — it is dataset-specific). *Governance: faithfulness claims must name the regime +
   sign convention (Art. 86 / GDPR 13–15).*
2. **Fairness (`11`, EV-012):** disparities are **directionally consistent** (women/young/foreign
   declined more) **but none is statistically significant at n=300** (signed CIs include 0; sex FPR
   permutation p=0.086); the model **trains on personal-status (sex proxy) + age directly**. *Monitoring
   flags, not established gaps (Art. 10).*
3. **Robustness (`12`, EV-013):** moderately stable — **3.6–16.3%** decision flips as noise grows;
   **13.3% near-threshold** fragile band. *Art. 15 concern to monitor.*
4. **Recourse (`12`, EV-013):** **94.0% [89.1, 96.8]** of declined applicants have loan-term recourse
   (roughly evenly a shorter term or lower amount); a **~6% infeasible core** rests on non-actionable
   factors and needs a genuine explanation. *GDPR 13–15.*

## D. Traceability (claim → evidence)
Each quantitative claim above maps to an `EV-###` row in `09-evidence-index.md` and the producing
script. **Provenance maturity is partial:** data artifacts carry SHA256 prefixes; living governance
docs are content-addressed **at the v1 release freeze** (not yet hashed); the pre-registration
(`HYPOTHESES.md`, tag `prereg-v1`) is committed and **OpenTimestamps-stamped, with Bitcoin
verification pending** (`09` note; Task 26). A standalone `08-traceability-matrix.md` is **not yet
produced**; this section + the evidence index are the interim spine.

## E. Residual gaps & open risks (honest register)
- **Not produced (obligations):** QMS (Art. 17), logging/record-keeping (Art. 12), cybersecurity
  assessment (Art. 15), post-market monitoring (Art. 72) & incident reporting (Art. 73), instructions-
  for-use (Art. 13), CE marking (Art. 48), EU-database registration (Art. 49), human-oversight
  *effectiveness* testing (Art. 14), GDPR DPIA (Art. 35 / `06`), FRIA (Art. 27 — deployer), standalone
  traceability matrix (`08`).
- **Method limits:** single stratified split, small n (≈300 test; subgroup claims low-powered);
  robustness noise is synthetic (not calibrated to real error logs); recourse is model-side; one model,
  mainline dataset + one generalization check.
- **Independence:** **self-assessment** — the author built the audited model and performs the
  assessment. Method + code were hardened through internal + **external model-reviewer** gauntlets;
  this is a quality control, **not** human independent review and confers **no** independence. The
  "externally reviewed / independently audited" claim is made only once ≥1 human external peer review
  signs off (plan Task 24, pending).

## F. Overall readiness verdict
Against the pinned criteria: **Ready** only on the classification (Art. 6) and lifecycle elements;
**Partial** on risk management, data governance, technical documentation, transparency, human
oversight, accuracy/robustness, provider obligations, the Annex VI route, and Art. 86; **Gap** on QMS,
logging, cybersecurity, post-market monitoring, DPIA, CE marking, and registration; **Scope** on the
deployer duties (Arts. 26, 27). This is a **credible readiness demonstration on the modelling and
assurance dimensions**, explicitly **not** a conformity declaration, **not** independent, and **not**
complete against the full high-risk regime. The signed audit opinion (`14`, next) draws its scope,
findings, and limitations from this dossier.
