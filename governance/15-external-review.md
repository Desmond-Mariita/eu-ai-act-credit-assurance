# 15 — Independent Review (template — to be completed by the reviewer)

> **How to use:** work top to bottom. Each section is tagged with the **competency** it needs. Fill in
> the *Reviewer finding* / *Verdict* cells. A section you are not qualified to attest → mark **"not
> assessed (out of competency)"** rather than passing it. The final sign-off is only as strong as the
> competency + independence declared in §0.

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
numbers match. (German Credit is fetched by code; GMSC needs a one-time Kaggle `cs-training.csv` in
`data/`.) Tick "matches?" against the expected values:

| Command | Expected (headline) | Matches? |
|---|---|---|
| `python scripts/10_train.py` | AUROC **0.769**, Brier 0.178, model SHA `1456b07f…` | ☐ |
| `python scripts/30_faithfulness.py` | abs TreeSHAP **+0.106 [.094,.118]**, LIME **+0.076 [.066,.086]**, clean control at floor | ☐ |
| `python scripts/40_fairness.py` | sex FPR-diff CI includes 0, **perm p≈0.086** | ☐ |
| `python scripts/50_robustness.py` | flip **3.6%** (eps .05) → **16.3%** (eps .5); near-thr **13.3%** | ☐ |
| `python scripts/60_reason_codes.py` | recourse **94.0% [89.1,96.8]**, infeasible ~6% | ☐ |
| `python scripts/70_roar.py` | TreeSHAP 0.108±0.017 ≈ LIME 0.111±0.016, both ≫ random 0.036 | ☐ |
| `uv run pytest -q` / `ruff check` | 30 passed / clean | ☐ |
| `git verify-tag prereg-v1`; `ots verify HYPOTHESES.md.ots` | signed; Bitcoin-attested (block 956533) | ☐ |

**Reviewer note:** _______________________________________________

## 2. Claims ↔ evidence traceability — *competency: any (with the map)*
Pick ≥5 headline numbers from `README`/`14` and confirm each resolves via `08-traceability-matrix.md`
→ `metrics/*.json` (hashes in `09`). Any claim that does **not** resolve is a finding.
**Reviewer note:** _______________________________________________

## 3. Method soundness — *competency: ML / statistics (required)*
- **Faithfulness:** is the movement metric **validated** (clean/random control sits at the floor)? Is
  the signed-null correctly called *underpowered* (not "unfaithful")? Is the GMSC replication scoped to
  the *core* result (signed null is dataset-specific)?
- **ROAR:** multi-split, retrain-based, correct global-importance aggregation; is "TreeSHAP ≈ LIME"
  (ordering metric-specific) the right read?
- **Fairness:** are disparities correctly reported as **not significant at n=300** using the **signed**
  CIs / permutation test — and is the note that folded DP/EO CIs are *not* a significance test correct?
- **Recourse:** is the grid search genuinely **exhaustive** (integer, pinned model), so ~6% is true
  infeasibility not a solver artifact?
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
