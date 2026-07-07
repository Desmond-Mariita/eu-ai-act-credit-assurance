# Changelog

## v1.1.1 — 2026-07-07 (post-audit polish)

v1.1 was re-audited by a fresh **4-model gauntlet** (Codex + DeepSeek both full-execution; GLM + Gemini
partial), returning a **unanimous PASS-WITH-RESERVATIONS** with **no Blocker and no sustained Major** —
both full-execution auditors byte-reproduced the entire pipeline and re-derived every headline statistic;
the fail-closed manifest, group-valid LIME, decoupled RNG, and ruff D+ANN gate were all confirmed by
execution. This patch closes the only surviving (Minor) findings and **retires the model-audit gauntlet**
for this flagship (converged; the remaining condition is independent HUMAN review, `governance/15`):
- **Digital Omnibus date now source-pinned** (`00-scenario.md §4`) — the "2 Dec 2027" Annex III date was
  correct (Council final approval 29 June 2026) but uncited; added the Council/Consilium reference.
- **Stale `v1.0` helper text refreshed to `v1.1.1`** in `.allowed_signers`, `08`, `09`, `02` (verify-tag
  examples + content-addressing labels + status).
- **Robustness stdout precision** aligned to the JSON (`.4f`).

Dismissed with evidence: Gemini's "Art. 86 is wrong / Art. 26 should be Art. 29" — verified false against
the AI Act (Art. 86 = *Right to Explanation of Individual Decision-Making*; Art. 26 = *Obligations of
Deployers*; Confidentiality is Art. 78). No code, model, or metric artifact changed in v1.1.1.

## v1.1 — 2026-07-06 (deep-audit residual remediation + code-quality retrofit)

A second, deeper independent-audit round (Codex full-execution + DeepSeek + Gemini, reconciled) returned
**PASS-WITH-RESERVATIONS** — no numerical headline mismatch, full pipeline byte-reproducible — but
surfaced residuals now remediated (R1–R7 below). This release ALSO carries a repo-wide **code-quality
retrofit** (Google-style docstrings + PEP 484 type hints + LaTeX-above-formulas + invariants,
CI-enforced via ruff `D`+`ANN`), verified behaviourally inert (only the three Part-A metrics changed).
The batch was itself **reviewed pre-commit** by a fresh 3-model panel → Codex found + I fixed a
manifest fail-open path, a stale evidence-index hash, an ambiguous Wilson-CI formula, and residual
"underpowered null" wording → Codex confirmation pass returned **APPROVE-WITH-NITS** (nit fixed). Cut as
a NEW signed tag `v1.1` — the public `v1.0` tag is left untouched (no moving a published tag).
**Still model-assisted QA, NOT independent human assurance**; the `governance/15` gap remains the
load-bearing condition. No reproduced statistic moved except the three explicitly re-run below.

- **R1 — recourse causal overclaim corrected.** The ~13% "infeasible core" was wrongly attributed to
  "non-actionable factors." It is *reduction*-infeasibility (an action-set fact): of 19/151, **4 flip to
  accept by *increasing* credit amount** (perverse model non-monotonicity — a deployer model-risk item)
  and 15 do not flip under any single-feature move. `60_reason_codes.py` now reports the breakdown;
  README/opinion/dossier/`12` reworded. Recourse rate itself unchanged (87.4%).
- **R2 — GMSC traceability closed.** The GMSC generalization model is now **persisted + hash-pinned**
  (`models/lightgbm_gmsc.txt`, verified in-script), and the derived `data/gmsc.parquet` is now covered by
  `data_manifest.sha256`. The manifest workflow is **fail-closed**: `verify` (default/CI) never mutates
  the manifest; only an explicit maintainer `freeze` re-baselines it (no more silent drift-blessing).
- **R3 — stale cross-document narratives reconciled.** GMSC metrics `interpretation` is now derived from
  its own H1 (was a verbatim copy of German's, which mis-stated the GMSC signed result); `08` FAITH-2
  reframed off "underpowered null"; data-dictionary "imputed at load" corrected (NaN preserved →
  train-split imputation); `foreign_worker` named as a model input. Semantic cross-doc tests added.
- **R4 — GDPR Art. 35 wording made conditional.** 35(2) DPO advice "where a DPO is designated"; 35(9)
  data-subject views "where appropriate" (was overstated as unconditionally mandatory).
- **R5 — robustness RNG streams decoupled.** Per-eps perturbation and bootstrap now use independent
  `SeedSequence`-spawned streams, so flip rates no longer depend on bootstrap count. Re-ran; point
  estimates shifted ≤0.7pp (3.6–16.3% → 3.8–16.7%).
- **R6 — tag verification made self-contained.** Ships `.allowed_signers`; `git -c
  gpg.ssh.allowedSignersFile=.allowed_signers verify-tag prereg-v1 v1.0` now verifies out-of-box, and CI
  verifies the signed tags.
- **R7 — DiCE claim scoped down.** Removed the unverified causal "off-the-shelf tool conflated
  search-failure with infeasibility / under-reported" claim (no apples-to-apples v1.0 evidence); kept the
  neutral "superseded by exhaustive grid" note.

Re-ran: `50_robustness` (R5), `60_reason_codes` (R1), `30_faithfulness` both datasets (R2 model pin + R3
interpretation; German byte-identical). Evidence index + integrity tests re-frozen in lockstep.

## v1.0 — amended 2026-07-06 (post-audit remediation)

After **internal + external model-assisted QA** (four models: internal Claude auditor + external
DeepSeek, Gemini, Codex), a full-execution audit returned a FAIL with substantive findings. **These are
model reviews, NOT independent human assurance** — the load-bearing independent-review gap
(`governance/15`) remains open. The `v1.0` release tag was **re-cut on the remediated commit**. The
amendments correct methodology and overclaims; the core finding (TreeSHAP faithful under a movement
diagnostic, corroborated by ROAR; GMSC replication) held.

Corrections:
- **LIME made group-valid.** The earlier `categorical_features` fix still sampled one-hot dummies
  independently (~99.9% of neighbourhoods off-manifold). LIME now runs in the **label-encoded logical
  space** with a re-expanding predict wrapper, so every sampled neighbour is a valid one-hot
  (100%). Re-ran German faithfulness + ROAR.
- **Recourse direction-constrained.** The grid counted perverse *increases* (borrow more/longer) as
  recourse. Restricted to actionable **reductions** (shorter term / lower amount); recourse recomputed
  (≈94% → ≈87%). Coupling/affordability constraints disclosed as unmodelled.
- **Signed result stated precisely.** Reframed from a uniform "underpowered null": TreeSHAP is
  inconclusive; LIME separated below the floor under the pre-registered signed estimand.
- **Movement metric = exploratory.** Dropped the "validated" construct claim — the clean control is a
  calibration check (same family as the floor) and the confounded control clears the bar; corroborated
  by ROAR, not asserted as validated.
- **ROAR scoped to global ranking utility**, not proof of local-explanation faithfulness.
- **Fairness statistics corrected.** FPR permutation now permutes within actual negatives (+1
  correction) with a Fisher-exact cross-check and an omnibus test for age; sex FPR p ≈ 0.096 (was a
  miscomputed 0.086). Conclusion (no disparity significant at 0.05) unchanged.
- **GMSC leakage fixed** (median imputation now fit on the train split only) + GMSC model AUROC stored
  as a hashed artifact (was orphaned). GDPR Art. 13–15 wording separated from optional model-side
  recourse. Environment re-locked (`uv.lock`), LICENSE completed, package version aligned to 1.0.0,
  stale cross-references cleared, honesty wording corrected throughout.

The pre-registration (`prereg-v1`, `HYPOTHESES.md`, OpenTimestamps Bitcoin-confirmed block 956533) is
**unchanged** and independent of this release tag.

## v1.0 — 2026-07-05 (initial + first amendment)

Initial public release: pre-registered self-assessment audit of a high-risk EU AI Act credit-scoring
model (faithfulness, fairness, robustness, recourse) with a governance/conformity pack, signed opinion,
and Bitcoin-timestamped pre-registration. A same-day amendment restored a full-scale faithfulness
artifact after a smoke-test file had been committed, and added a CI integrity guard.
