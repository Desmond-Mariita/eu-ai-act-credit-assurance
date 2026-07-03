# EU AI Act Credit-Assurance — Explanation-Faithfulness Audit

A **pre-registered** audit of a high-risk **credit-scoring** model's *explanation faithfulness*,
wrapped in an **EU AI Act + GDPR** governance evidence pack.

> **Status:** pre-registered (`prereg-v1`); **results phase in progress.**
> **Nature:** a **self-assessment / methodology demonstration — NOT an independent audit**
> (the author built the audited model *and* performs the assessment; see
> [`governance/00-scenario.md`](governance/00-scenario.md) §7). Scope is *behavioural,
> perturbation-faithfulness* — **not** a proof of "true" faithfulness.

## The question
When a bank shows a loan officer or a declined applicant an "explanation" of a credit decision, is
that explanation **faithful** — i.e. do the features it calls important actually drive the model's
output? This repo audits **TreeSHAP** vs **LIME** on a **LightGBM** credit model (with **random** and
**label-shuffled** negative controls), and asks what it means for **EU AI Act Art. 86 / GDPR
Arts. 13–15** if production explanations are *not* faithful.

## Pre-registration & integrity
The method, directional hypotheses, and acceptance thresholds were fixed **before any model was
trained or any result produced** — see [`HYPOTHESES.md`](HYPOTHESES.md), committed and bound by the
signed tag [`prereg-v1`](../../releases/tag/prereg-v1) with an OpenTimestamps proof
(`HYPOTHESES.md.ots`). The measurement instrument was frozen first. This makes the
"did-not-tune-to-pass" claim publicly falsifiable. Proof trail:
[`governance/09-evidence-index.md`](governance/09-evidence-index.md).

## What's built
- **Audit charter** — Target of Evaluation, criteria, roles, scoping exclusions, COI disclosure
  ([`governance/`](governance/)).
- **Data documentation (Art. 10)** — manifest + verifier, a data-quality/**error** profile
  (`data_quality.json`), feature-group map, dictionary, CDE register, bias-representation note.
- **Frozen instrument** — a feature-group-aware perturber (baseline / marginal / on-manifold
  conditional, standardised kNN) and pure faithfulness metrics
  ([`src/credit_assurance/`](src/credit_assurance/)), unit-tested.

Every layer above was hardened through independent internal + external review before commit.

## What's coming (the results phase)
Train the LightGBM model + EBM challenger → **run the pre-registered benchmark** → ROAR anchor →
counterfactual reason codes + robustness → the governance dossier → a **signed audit opinion** →
external peer review → a **finding-first** rewrite of this README.

## Reproduce
```bash
uv venv && uv pip install -e ".[dev]"   # core + dev (numpy/pandas/scikit-learn/pytest/ruff)
uv run ruff check src/ tests/
uv run pytest -q
```
Data acquisition: German Credit is fetched by code; **Give Me Some Credit** needs a one-time manual
Kaggle download of `cs-training.csv` into `data/` (verified by hash via
`python scripts/05_manifest_and_dq.py verify`). The Phase-2 model stack installs with `.[models]`.

## Honest caveats
Perturbation-faithfulness only (never "true"); simulated conformity *readiness*, not a real
conformity assessment; self-assessment, not an independent audit; German Credit is small/old — the
*vehicle, not the headline*, with Give Me Some Credit as a generalization check.

## Licence
MIT (see [`LICENSE`](LICENSE)). Datasets under their own terms (German Credit CC BY 4.0; GMSC per
Kaggle terms) — raw data is git-ignored, not redistributed.
