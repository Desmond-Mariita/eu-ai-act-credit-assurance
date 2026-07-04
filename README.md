# EU AI Act Credit-Assurance — a pre-registered, reproducible AI audit

A **pre-registered, end-to-end AI-assurance audit** of a high-risk **credit-scoring** model, against
the **EU AI Act + GDPR**. Explanation faithfulness, fairness, robustness, and recourse — each with a
concrete, reproducible finding, wrapped in an Annex IV governance evidence pack.

> **Status:** v1.0 findings complete. **Self-assessment — NOT an independent audit** (the author built
> the audited model *and* assessed it). The one load-bearing open item is **independent human
> review**; a model-reviewer gauntlet is a quality control, not a substitute. Perturbation-faithfulness
> only, not "true" faithfulness.

## Headline findings (each internally + externally reviewed to consensus)
1. **Explanation faithfulness is metric-dependent — and under a *validated* metric, explanations are
   faithful.** Under a direction-agnostic movement metric (validated: a random control sits at the
   floor), **both TreeSHAP and LIME beat random**; a retrain-based **ROAR** anchor (8 splits)
   independently corroborates this. The pre-registered *signed* on-manifold test is an **underpowered
   null** (sign cancellation), *not* evidence of unfaithfulness. Replicated on a second dataset.
   → *Faithfulness claims must state the perturbation regime and sign convention.*
2. **Fairness: disparities disadvantage the worse-off groups, but none is significant at n=300**
   (sex FPR permutation p=0.086); the model **trains on a sex proxy and age directly**. → *Monitoring
   flags, not established gaps.*
3. **Robustness: moderate** — 3.6–16.3% of decisions flip under small input noise; a **13.3%
   near-threshold** population is fragile.
4. **Recourse: 94% [89.1, 96.8]** of declined applicants have actionable loan-term recourse; a **~6%
   infeasible core** rests on non-actionable factors. *(A popular off-the-shelf counterfactual tool
   under-reported recourse by ~20 points — a cautionary note for auditors.)*

Full write-ups: [`governance/10`](governance/10-faithfulness-findings.md) ·
[`11`](governance/11-fairness-findings.md) · [`12`](governance/12-robustness-and-recourse.md).
**Signed opinion:** [`governance/14-audit-opinion.md`](governance/14-audit-opinion.md).
**Conformity-readiness dossier (Annex IV + article matrix):**
[`governance/13`](governance/13-conformity-dossier.md).

## What makes it credible
- **Pre-registration:** the method + hypotheses (H1–H4) were fixed **before results** —
  [`HYPOTHESES.md`](HYPOTHESES.md), signed tag [`prereg-v1`](../../releases/tag/prereg-v1),
  **OpenTimestamps Bitcoin-confirmed** (block 956533). The absolute-movement metric + ROAR were
  disclosed post-hoc additions.
- **Adversarial review:** every finding was hardened through internal (Claude) + external
  (Gemini/Codex/DeepSeek) model-reviewer gauntlets — which caught, and I corrected, a semantic LIME
  bug, over-read fairness CIs, a counterfactual search artifact, and more.
- **Full traceability:** every quantitative claim → a hashed artifact + a one-command reproduction
  ([`governance/08`](governance/08-traceability-matrix.md), [`09`](governance/09-evidence-index.md)).

## Reproduce
```bash
uv venv && uv pip install -e ".[dev]"      # core + tests
uv run ruff check src/ tests/ && uv run pytest -q
uv pip install -e ".[models]"              # model/explainer stack (Phase 2)
python scripts/10_train.py                 # audited model (deterministic; SHA pinned)
python scripts/30_faithfulness.py          # faithfulness benchmark
python scripts/{40_fairness,50_robustness,60_reason_codes,70_roar}.py
```
Data: German Credit is fetched by code; **Give Me Some Credit** needs a one-time Kaggle download of
`cs-training.csv` into `data/`.

## Honest limitations
Self-assessment (not independent); single model; mainline dataset + one generalization check; n≈300
test (low subgroup power); robustness noise synthetic; recourse model-side. Conformity readiness is
**partial** — several obligations (QMS, logging, cybersecurity, post-market monitoring, a real DPIA,
CE/registration) are **Gap**; the DPIA and FRIA are template/deployer-side. Full register:
[`governance/13 §E`](governance/13-conformity-dossier.md).

## Licence
MIT (see [`LICENSE`](LICENSE)). Datasets under their own terms; raw data is git-ignored, not
redistributed.
