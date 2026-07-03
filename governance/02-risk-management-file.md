# 02 — Risk Management File (EU AI Act Art. 9) + Assumptions Log

**Status:** Phase 0.5 — the **Assumptions log (§2) is initiated now, before any results**; the risk
register (§1) is seeded here and finalized in plan Task 16.

## 1. Risk register (Art. 9) — seed
Likelihood/Impact on a defined scale (see `10-severity-rubric.md`, plan Task 19). Owner = the role
accountable in the scenario.

| ID | Hazard | Likelihood | Impact | Mitigation | Residual | Owner |
|----|--------|-----------|--------|-----------|----------|-------|
| R1 | Proxy discrimination via `personal_status`/`foreign_worker`/`age` | TBD | High | Fairness audit + one mitigation pass; protected/proxy policy | TBD | Deployer |
| R2 | Unfaithful explanations shown to officers/applicants | TBD | High | Perturbation-faithfulness benchmark + negative controls | TBD | Provider |
| R3 | Covariate drift degrades performance post-deployment | TBD | Med | Drift mini-audit; monitoring plan (simulated) | TBD | Deployer |
| R4 | Adversarial gaming of visible feature weights | TBD | Med | Gaming probe; feature-exposure review | TBD | Provider |
| R5 | Data-quality defects (completeness/errors) | TBD | Med | `data_quality.json` profile (Art. 10(3)) | TBD | Provider |
| R6 | Over-reliance on automation (human oversight fails) | TBD | High | Human-in-the-loop design; Art. 22 trigger analysis | TBD | Deployer |

*(Likelihood/Impact/Residual populated in Task 16 once evidence exists.)*

## 2. Assumptions log (initiated Phase 0.5 — before results; Art. 9 / Annex IV)
Design, method, and data assumptions fixed **before** running, appended as the build proceeds.
Recording them pre-results guards against post-hoc rationalisation.

| # | Assumption | Type | Rationale / caveat |
|---|-----------|------|--------------------|
| A1 | Positive class = `bad` == 1; models output `P(bad)` | Method | Fixes sign conventions across cores (see plan Conventions). |
| A2 | German Credit is representative *enough* for a methodology demonstration | Data | It is small/old/saturated — the **vehicle, not the headline**; GMSC is the generalization check. |
| A3 | The 5:1 cost matrix is taken as given | Design | Dataset-provided; not validated for any real lender (see ToE §6). |
| A4 | TreeSHAP gives (near-)exact Shapley values for the tree model | Method | Basis for the directional hypothesis that TreeSHAP is more faithful than LIME. |
| A5 | The conditional kNN donor approximates on-manifold perturbation | Method | Approximate; residual OOD risk is **measured and reported**, not eliminated. |
| A6 | Label-shuffled + random-attribution runs are known-bad negative controls | Method | The instrument must flag them as unfaithful (pre-registered). |
| A7 | The data-quality profile does **not** inform the faithfulness hypotheses | Method | Hypotheses are method-derived and pre-registered before training (anti-leakage). |

*(Append further assumptions with their date/phase as they arise.)*
