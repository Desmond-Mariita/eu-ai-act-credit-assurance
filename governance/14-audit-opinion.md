# 14 — Audit Opinion

**Engagement:** AI-assurance assessment of a high-risk credit-scoring model against the EU AI Act +
GDPR criteria pinned in `00-scenario.md §4`.
**Target of Evaluation:** the SHA-pinned LightGBM model `models/lightgbm.txt`
(`1456b07f…`, `00-scenario.md §5`).
**Assessor:** the author (Desmond Mariita), in a methodology-demonstration capacity.
**Date:** 2026-07-05. **Report basis:** dossier `13`, findings `10`–`12`, `06`, `08`, evidence index
`09`, pre-registration `HYPOTHESES.md` (tag `prereg-v1`).

## 1. Nature and level of assurance
This is a **self-assessment providing NO formal assurance and NO conformity declaration.** The assessor
built the audited model *and* performed the assessment, so **independence is not satisfied** (audit
independence in the assurance sense — cf. Annex VII / ISAE-3000). Method and code were hardened through internal and **external
model-reviewer** gauntlets — a quality control, **not** human independent review. Accordingly this is a
**limited, qualified statement of findings and readiness**, not "reasonable" or "limited assurance" in
the ISAE-3000 sense. A stronger opinion is conditional on the items in §5.

## 2. Scope
**In scope:** the model's predictive performance, **explanation faithfulness**, **fairness**,
**robustness**, and **recourse**, plus the governance evidence and conformity-readiness mapping (`13`).
**Out of scope** (`00-scenario.md §6`): live production operation, data provenance/collection
lawfulness, security posture, and any system other than the audited model on German Credit (+ a Give
Me Some Credit generalization check).

## 3. Summary of findings (each internally + externally reviewed to consensus)
1. **Faithfulness — favourable, nuanced.** Under a **validated** direction-agnostic movement metric,
   **both TreeSHAP and LIME are faithful** (TreeSHAP > LIME *under this metric*); a **retrain-based
   ROAR anchor (8 splits) independently corroborates that both are faithful** (0.108 / 0.111 vs 0.036
   random, both beat the floor on every split) — though ROAR finds the two **indistinguishable**, so
   the ordering is metric-dependent. The pre-registered *signed*
   on-manifold test is an **underpowered null**, not evidence of unfaithfulness. The core result
   **replicated on a second dataset.** *Faithfulness is metric-dependent — claims must state the regime
   and sign convention.*
2. **Fairness — no significant disparity, but flags.** Group disparities **consistently disadvantage**
   women and younger applicants (the foreign-worker comparison is **not inferable**, n=15) but **none
   is statistically significant at n=300**
   (sex FPR permutation p=0.086). The model **trains on a sex proxy (personal-status) and age
   directly.** *Monitoring + mitigation-study flags, not established gaps.*
3. **Robustness — moderate.** 3.6–16.3% of decisions flip under small input noise; a **13.3%
   near-threshold** population is noise-sensitive. *An Art. 15 monitoring concern.*
4. **Recourse — broadly available.** **94% [89.1, 96.8]** of declined applicants have actionable loan-term
   recourse; a **~6% infeasible core** rests on non-actionable factors and requires a genuine
   explanation. *(A popular off-the-shelf CF tool under-reported this by ~20 points — a cautionary
   note for auditors.)*

## 4. Opinion
On the dimensions assessed and against the pinned criteria, **on the evidence gathered the assessor
identified no indication that the model is unfit** on faithfulness, fairness, robustness, or recourse —
indeed the explanations are demonstrably faithful and recourse is broadly available — **subject to** the fairness
monitoring flags, the near-threshold robustness population, and the ~6% recourse-infeasible core being
addressed by the deployer. **This is NOT an opinion that the system is EU-AI-Act-compliant or
conformant.** Conformity readiness is **partial** (`13 §F`): several high-risk obligations are **Gap**
(QMS, logging, cybersecurity, post-market monitoring, DPIA-as-real, CE/registration) and the
assessment is **not independent**. No opinion is expressed on those.

## 5. Conditions for a stronger opinion
(a) **≥1 independent human external review** with sign-off (currently absent — the load-bearing gap);
(b) closure of the §F Gap obligations; (c) larger-sample fairness re-run (subgroup power);
(d) real-deployment logging, human-oversight-effectiveness testing, and post-market monitoring.

## 6. Material limitations
Self-assessment (not independent); single model; mainline dataset + one generalization check; n≈300
test (low subgroup power); robustness noise synthetic; recourse model-side; public research data (no
real data subjects). Full register: `13 §E`.

## 7. Integrity attestation
The faithfulness **hypotheses (H1–H4) and the pre-registered signed comprehensiveness method** were
fixed **before results**; the **absolute-movement metric and the ROAR anchor were post-hoc additions**
(disclosed in `10 §Deviations`). Pre-registration (`HYPOTHESES.md`, signed
tag `prereg-v1`, OpenTimestamps — **Bitcoin-confirmed, block 956533**). Every quantitative claim traces
to a hashed metrics artifact and a one-command reproduction (`08`, `09`); governance documents are
content-addressed at the v1 freeze. This document is committed to
the public repository under version control; its integrity rests on the git history and the signed
pre-registration tag, **not** on any claim of independence.
