# 02 — Risk Management File (EU AI Act Art. 9) + Assumptions Log

**Status:** v1.0 — **risk register (§1) finalised from the completed findings** (`10`–`12`, `06`); the
Assumptions log (§2) was initiated Phase 0.5, before results.

## 1. Risk register (Art. 9) — finalised
Scale: Likelihood / Residual ∈ {Low, Med, High}; **assessed 2026-07-05 from findings `10`–`12` + DPIA
`06`**. Owner = the accountable role. The DPIA (`06 §5`) scores the data-protection subset consistently
(discrimination / proxy / opacity / instability ≈ Med×Med).

| ID | Hazard | Likelihood | Impact | Mitigation performed | Residual | Owner |
|----|--------|-----------|--------|-----------|----------|-------|
| R1 | Proxy discrimination via `personal_status`/`foreign_worker`/`age` | **Med** | High | Fairness audit done (`11`) — **NO mitigation applied**; model still trains on the sex proxy + age | **Med** (disparities directional but **not significant at n=300**; monitored, not mitigated) | **Provider** (Art. 10(2)(f)) |
| R2 | Unfaithful explanations shown to officers/applicants | **Low–Med** | High | Faithfulness benchmark + ROAR (`10`): explanations faithful under a **validated** metric | **Low–Med** (faithfulness is **metric-dependent**; a naive signed score would mislabel) | Provider |
| R3 | Covariate drift degrades performance post-deployment | **Med** | Med | **Not performed** — drift / post-market monitoring is a **Gap** (`13 §E`) | **Med–High** (no drift monitoring) | **Provider** (build) / **Deployer** (Art. 26/72) |
| R4 | Adversarial gaming of visible feature weights | **Low–Med** | Med | **Not performed** — gaming probe is a **Gap** | **Med** (unassessed) | Provider |
| R5 | Data-quality defects (completeness/errors) | **Med** | Med | DQ profile done (`data_quality.json`, Art. 10(3)); GMSC 96/98 sentinels + age errors cleaned | **Low–Med** (documented + handled) | Provider |
| R6 | Over-reliance on automation (human oversight fails) | **Med** | High | Human-in-the-loop **design only** — effectiveness **not tested** (`13 §E`) | **High** (the load-bearing safeguard is unverified; robustness `12` shows a 13.3% near-threshold population) | Deployer (Art. 14/22) |

*Residuals reflect that several mitigations are **assessed but not remediated** (self-assessment); the
open items (drift/PMM, gaming probe, oversight-effectiveness) are in the dossier gap register `13 §E`.*

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
