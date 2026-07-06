# 15 — Independent Review (template — to be completed by the reviewer)

> **How to use:** work top to bottom. Each section is tagged with the **competency** it needs. Fill in
> the *Reviewer finding* / *Verdict* cells. A section you are not qualified to attest → mark **"not
> assessed (out of competency)"** rather than passing it. The final sign-off is only as strong as the
> competency + independence declared in §0.

> **How to submit this review (keep it independent):**
> 1. **Fork** this public repo to your own GitHub account (no write access needed).
> 2. On your fork, complete this file — or add `governance/15-<yourname>-review.md` — and, ideally,
>    **sign your commit** (`git commit -S…`, GPG or SSH) so it is cryptographically attributable to you.
> 3. Open a **Pull Request** to `main`. The author will **merge without editing your words** (any
>    disagreement is handled as PR discussion, not an edit) — this preserves your independence and
>    leaves a permanent, timestamped, attributable record.
> Alternatives: a GitHub **Issue** (faster, less formal). *Avoid* being added as a repo collaborator —
> a fork + PR keeps you cleanly arm's-length. Use a real/verifiable GitHub identity.

## 0. Reviewer & independence declaration *(reviewer fills in)*
- **Reviewer:** __________________  **Role / affiliation:** __________________  **Date:** __________
- **Independence** (tick one):
  - [ ] **Independent** — no involvement in building the audited model or this audit; no conflict.
  - [ ] **Author self-review / internal QA** — I built part of the model/audit (this does **not** close
    the Task-24 independence gap; record as internal QA).
  - [ ] Other relationship: __________________
- **Competency I can attest** (tick all that apply): [ ] ML / statistics  [ ] EU AI Act & GDPR
  [ ] software / reproducibility
- **Time spent:** ______  **Environment:** OS ______, Python ______, commit reviewed `________`.

## 1. Reproducibility — *competency: software/reproducibility (any technical reviewer)*
Fresh clone; `uv venv && uv pip install -e ".[dev,models]"`; run the scripts; confirm the printed
numbers match. (`data/` is gitignored — run `python scripts/00_data.py` **first** to fetch German
Credit; GMSC needs a one-time Kaggle `cs-training.csv` in `data/` before `06_gmsc_prep.py`.) Tick
"matches?" against the expected values:

| Command | Expected (headline) | Matches? |
|---|---|---|
| `python scripts/00_data.py`; `06_gmsc_prep.py`; `05_manifest_and_dq.py` | writes `data/*.parquet`; `05` (default) rebuilds the DQ profile + **verifies** the committed manifest fail-closed (maintainer-only `freeze` re-baselines it) | ☐ |
| `python scripts/10_train.py` | AUROC **0.769**, Brier 0.178, model SHA `1456b07f…` | ☐ |
| `python scripts/30_faithfulness.py` | abs TreeSHAP **+0.106 [.094,.118]**, LIME **+0.090 [.080,.100]** (group-valid); signed: TS inconclusive, LIME below floor | ☐ |
| `python scripts/40_fairness.py` | sex FPR-diff CI includes 0, **perm p≈0.091 / Fisher 0.096**; age omnibus p≈0.39 | ☐ |
| `python scripts/50_robustness.py` | flip **3.8%** (eps .05) → **16.7%** (eps .5); near-thr **13.3%** | ☐ |
| `python scripts/60_reason_codes.py` | recourse **87.4% [81.2,91.8]** (reductions only); **12.6% reduction-infeasible**, of which **4/19 flip via *increase*** | ☐ |
| `python scripts/70_roar.py` | TreeSHAP 0.108±0.017 ≈ LIME 0.109±0.021, both ≫ random 0.036 | ☐ |
| `uv run pytest -q` / `ruff check` | 43 passed / clean (ruff enforces the D+ANN code standard) | ☐ |
| `git verify-tag prereg-v1`; `ots verify HYPOTHESES.md.ots` | signed; Bitcoin-attested (block 956533) | ☐ |

**Reviewer note:** _______________________________________________

## 2. Claims ↔ evidence traceability — *competency: any (with the map)*
Pick ≥5 headline numbers from `README`/`14` and confirm each resolves via `08-traceability-matrix.md`
→ `metrics/*.json` (hashes in `09`). Any claim that does **not** resolve is a finding.
**Reviewer note:** _______________________________________________

## 3. Method soundness — *competency: ML / statistics (required)*
- **Faithfulness:** is the movement metric honestly framed as **exploratory / calibration-checked** (NOT
  "validated") — clean control at the floor, lenient-bar caveat disclosed, ROAR as corroboration? Is the
  signed result correctly framed (**TreeSHAP inconclusive; LIME *below* floor** — not a uniform null, not
  "unfaithful")? Is the GMSC replication scoped to the *core* result (signed result is dataset-specific)?
- **ROAR:** multi-split, retrain-based, correct global-importance aggregation; is "TreeSHAP ≈ LIME"
  (ordering metric-specific) the right read?
- **Fairness:** are disparities correctly reported as **not significant at n=300** using the **signed**
  CIs / permutation test — and is the note that folded DP/EO CIs are *not* a significance test correct?
- **Recourse:** is the grid search genuinely **exhaustive** (integer, pinned model, reductions only), so the
  **12.6% is *reduction*-infeasibility** (an action-set fact) rather than a solver artifact — and is the
  perverse-increase breakdown (4/19 flip by *increasing* a feature) reported instead of a causal
  "non-actionable factors" claim?
- **Robustness:** is the perturbation model + bootstrap CI reasonable; noise framed as *synthetic*?

**Verdict (this section):** sound / issues below / not assessed — _____________________

## 4. Regulatory mapping — *competency: EU AI Act & GDPR (required)*
- Is the **Annex IV** map + **article matrix** (`13 §A/§B`) accurate, and the **Annex VI / Art. 43(2)**
  internal-control route correct for Annex III §5(b)?
- Is the **DPIA** (`06`) sound as a template (Art. 35 trigger, Art. 22 analysis, Art. 9 proxy note,
  missing 35(2)/(9) flagged)?
- Are the **statuses** (Ready/Partial/Gap/Scope) honest — nothing marked better than reality?

**Verdict (this section):** sound / issues below / not assessed — _____________________

## 5. Overclaim & honesty — *competency: careful reader*
Does the **opinion (`14`)** stay within "no assurance / not conformity / not independent"? Is
"TreeSHAP > LIME" always scoped to the movement metric? Is the **gap register (`13 §E`)** truthful and
complete? Any sentence a regulator/auditor would reject?
**Reviewer note:** _______________________________________________

## 6. Findings log *(reviewer)*
| # | Section | Finding | Severity (Blocker/Major/Minor) | Required change |
|---|---|---|---|---|
| 1 |  |  |  |  |

## 7. Overall verdict & sign-off
- **Verdict:** ☐ Accept  ☐ Accept with changes (log above)  ☐ Reject
- **Statement:** _______________________________________________
- **Signature / git-signed commit / date:** __________________

> **On the "independently reviewed" claim:** it may be asserted in `14`/`13`/`README` **only** if §0
> declares an *Independent* reviewer with the relevant *competency* and the verdict is Accept (or
> Accept-with-changes once resolved). A self-review/internal-QA completion strengthens quality but
> leaves the Task-24 independence gap **open** and must be recorded as such.
