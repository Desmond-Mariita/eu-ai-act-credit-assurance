# EU AI Act Credit-Assurance — a pre-registered, reproducible AI audit

A **pre-registered, end-to-end AI-assurance audit** of a high-risk **credit-scoring** model, against
the **EU AI Act + GDPR**. Explanation faithfulness, fairness, robustness, and recourse — each with a
concrete, reproducible finding, wrapped in an Annex IV governance evidence pack.

> **Status:** v1.1.2 (findings complete + deep-audit residual remediation + code-quality retrofit;
> passed a unanimous 4-model audit gauntlet — see CHANGELOG).
> **Self-assessment — NOT an independent audit** (the author built
> the audited model *and* assessed it). The load-bearing gap for *independence* is **human external
> review** (the model-reviewer gauntlets are a quality control, not a substitute); conformity readiness
> is otherwise **partial**, with further conditions (QMS, logging, larger-n fairness, post-market
> monitoring — see [opinion §5](governance/14-audit-opinion.md)). Perturbation-faithfulness only, not
> "true" faithfulness.

## Headline findings (each hardened through internal + external *model-reviewer* gauntlets)
1. **Explanation faithfulness is metric-, sign-, and regime-dependent — and TreeSHAP is the stronger
   explainer.** Under an *exploratory* direction-agnostic movement metric (clean random control at the
   floor — a calibration check), **both TreeSHAP (+0.106) and LIME (+0.090, group-valid) beat the
   floor**, and a retrain-based **ROAR** anchor confirms both explainers' *global* rankings are strongly
   predictive. But it's nuanced: under the pre-registered *signed* metric **TreeSHAP is inconclusive and
   LIME separates below the floor** (not a uniform "underpowered null"), and at baseline only TreeSHAP is
   faithful. The core movement result **replicated** on a second dataset.
   → *Faithfulness claims must state the metric, sign convention, and regime.*
2. **Fairness: disparities disadvantage the worse-off groups, but none is significant at n=300**
   (sex FPR within-negatives permutation p=0.091, Fisher 0.096; age omnibus p=0.39); the model **trains
   on a sex proxy and age directly**. → *Monitoring flags, not established gaps.*
3. **Robustness: moderate** — ≈4–7% of decisions flip under *small* input noise (rising to 16.7% at
   the largest synthetic noise tested); a **13.3% near-threshold** population is fragile.
4. **Recourse: 87.4% [81.2, 91.8]** of declined applicants have an actionable loan-term **reduction** to
   acceptance. The remaining **12.6% (19/151) is *reduction-infeasible*** — a fact about the two-feature
   action set, **not** a claim that these declines rest on non-actionable factors: **4 of the 19 flip to
   accept by *increasing* credit amount** (perverse model non-monotonicity — a model-risk finding), and
   15 do not flip under any single-feature move. *(An initial off-the-shelf counterfactual search was
   superseded by an exhaustive, direction-constrained integer grid; the two are not apples-to-apples, so
   no under-reporting magnitude/cause is claimed — exhaustive search remains the sound way to establish
   infeasibility.)*

Full write-ups: [`governance/10`](governance/10-faithfulness-findings.md) ·
[`11`](governance/11-fairness-findings.md) · [`12`](governance/12-robustness-and-recourse.md).
**Signed opinion:** [`governance/14-audit-opinion.md`](governance/14-audit-opinion.md).
**Conformity-readiness dossier (Annex IV + article matrix):**
[`governance/13`](governance/13-conformity-dossier.md).

## What makes it credible
- **Pre-registration:** the method + hypotheses (H1–H4) were fixed **before results** —
  [`HYPOTHESES.md`](HYPOTHESES.md), signed tag [`prereg-v1`](https://github.com/Desmond-Mariita/eu-ai-act-credit-assurance/releases/tag/prereg-v1),
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
python scripts/00_data.py                  # fetch + write data/ (German Credit; data/ is gitignored)
python scripts/10_train.py                 # audited model (deterministic; SHA pinned)
python scripts/30_faithfulness.py          # faithfulness benchmark (German Credit)
python scripts/06_gmsc_prep.py             # GMSC prep (needs data/cs-training.csv — see below)
python scripts/30_faithfulness.py --dataset gmsc   # generalization check
for s in 40_fairness 50_robustness 60_reason_codes 70_roar; do python scripts/$s.py; done
```
Data: `00_data.py` **fetches German Credit and writes `data/`** (gitignored, so this step is required
on a fresh clone); **Give Me Some Credit** needs a one-time Kaggle download of `cs-training.csv` into
`data/` before `06_gmsc_prep.py`.

For byte-identical reproduction, `uv.lock` pins exact transitive versions — install from it with
`uv sync --frozen --extra models --extra dev` (CI enforces `uv lock --check`). The `uv pip install`
lines above are the quick path; the lock is authoritative.

## Honest limitations
Self-assessment (not independent); single model; mainline dataset + one generalization check; n≈300
test (low subgroup power); robustness noise synthetic; recourse model-side. Conformity readiness is
**partial** — several obligations (QMS, logging, cybersecurity, post-market monitoring, a real DPIA,
CE/registration) are **Gap**; the DPIA and FRIA are template/deployer-side. Full register:
[`governance/13 §E`](governance/13-conformity-dossier.md).

## Licence
MIT (see [`LICENSE`](LICENSE)). Datasets under their own terms; raw data is git-ignored, not
redistributed.
