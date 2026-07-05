# Changelog

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
