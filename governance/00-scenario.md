# 00 — Target of Evaluation, Scenario & Criteria (ToE)

**Status:** Phase 0.5 charter (written before the instrument freeze and any results).
**Nature:** a **self-assessment / methodology demonstration — NOT an independent audit** (see §7).

This document pins *what is evaluated* and *against what yardstick*, before any model is trained or
any result is produced. It is the fixed reference for every downstream artifact.

## 1. Target of Evaluation (ToE)
The AI system under audit is a **binary credit-scoring model** used at loan origination:

- **The system = the trained model + its decision threshold + its input feature contract.** A
  LightGBM classifier outputs `P(bad)`; a cost-sensitive threshold converts it to an
  accept/decline recommendation.
- **In scope:** the model's predictive behaviour, its explanations' faithfulness, fairness across
  protected/proxy attributes, robustness/drift, and the governance evidence around it.
- **Out of scope (system boundary):** upstream data *collection* and lineage, downstream loan
  servicing, the bank's IT security posture, and live production operation. The **EBM** trained
  alongside is a **challenger/validation comparator only** — not the deployed system.

**Intended purpose:** assist a human loan officer's origination decision (decision support), not
fully-automated lending (a full GDPR Art. 22 trigger analysis / DPIA — `06-gdpr-dpia.md` — is
**deferred / not produced**; see dossier `13` §B & §E).

## 2. Operating scenario (fictional EU bank)
`applicant → model P(bad) → cost-sensitive threshold → human reviewer → decision → appeal path`

- **Decision threshold:** cost-sensitive, using German Credit's official **5:1** cost matrix
  (a false "good" costs 5× a false "bad") → threshold on `P(bad)` = `cost_fp / (cost_fn + cost_fp)`.
- **Human oversight:** a loan officer reviews the model's recommendation before the decision issues;
  applicants may appeal (logged).
- **Logging:** each decision records inputs, score, threshold, recommendation, reviewer action.

## 3. Roles
- **Provider** (builds/places the system on the market): the bank's model team — here, the author,
  acting in a methodology-demonstration capacity.
- **Deployer** (uses it in the course of business): the bank's lending unit.
Both roles are simulated; obligations are assessed as *readiness*, not real conformity.

## 4. Criteria — the yardstick (version-pinned)
- **EU AI Act** — Annex III **§5(b)** high-risk credit-scoring; Arts. **6(2)** (Annex III
  classification) & **6(3)** (exemption analysed & rejected), **9, 10, 11, 12, 13, 14, 15, 16, 17,
  26, 27** (FRIA), **72, 73, 86**; **Annex IV** (technical documentation) & **Annex VI** (internal-
  control route). **Application date for stand-alone Annex III systems: 2 December 2027** *if* the
  Digital Omnibus is adopted; the **statutory baseline is 2 August 2026** (Art. **113**) absent it.
  Art. **111** separately governs systems already placed on the market. *AI Act consolidated-text
  version to be dated on freeze; verify against the final Official Journal text.*
- **GDPR** — Arts. **5** (principles), **6/9** (lawful basis / special-category — only where
  actually processed or proxy-inferred), **13–15** ("meaningful information"), **22(3)** (automated-
  decision safeguards — subject to the Art. 22 trigger analysis), **35** (DPIA).
- **Standards/frameworks (edition-pinned):** **ISO/IEC 42001:2023** (AIMS), **NIST AI RMF 1.0
  (2023)** + Generative-AI Profile, **BCBS 239 (Jan 2013)** (risk-data aggregation & reporting).

## 5. Model version & dataset snapshot (pinned on freeze)
- **Model:** LightGBM (audited) — artifact `models/lightgbm.txt`, **SHA256
  `1456b07fb365dcbc91d6b139c51e1269976509249bdb8d7d4b4cd8e48ab5eab5`** (deterministic; frozen at
  plan Task 8). Stratified holdout (seed 0, leakage check passed): **test AUROC 0.769, Brier 0.178**,
  cost-sensitive threshold **1/6**; EBM challenger AUROC 0.794. Full metrics: `metrics/models.json`.
- **Datasets:** German Credit (UCI Statlog, CC BY 4.0) mainline + Give Me Some Credit (generalization
  check) — **snapshot SHA256s recorded in `data_manifest.sha256` (plan Task 3)** and referenced here.

## 6. Scoping-exclusion statement (we express NO opinion on)
- Live production monitoring, incident handling, or model performance after deployment.
- Real-world data provenance, collection lawfulness, or data-subject consent (public research data).
- Production security, access control, or infrastructure.
- The suitability of the 5:1 cost matrix for any real lender (used as given by the dataset).
- Any system other than the single audited LightGBM model on these datasets.

## 7. Independence & conflict-of-interest disclosure
This is a **self-assessment**: the author built the audited model *and* performs the assessment.
True independence is therefore **not** satisfied. Mitigations: (a) **blind pre-registration** of
method + hypotheses before results (`HYPOTHESES.md`, tagged); (b) a **provider/auditor code
boundary** (`src/credit_assurance/system/` vs `audit/`); (c) **≥1 external peer review** with
sign-off before release (plan Task 24). The "externally reviewed" claim is only made if (c) is met.
