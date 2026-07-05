# Changelog

## v1.0 — amended 2026-07-05 (post-audit corrections)

The `v1.0` release tag was **re-cut on the corrected commit** after an independent audit by four models
(internal Claude + external DeepSeek, Gemini, Codex). The amendments do **not** change the audit's
conclusions — the central faithfulness result held and, where re-run, strengthened — they correct a
packaging defect, fix the LIME baseline properly, and finish a number cascade.

- **A1 (Blocker).** The committed `metrics/faithfulness_german_credit.json` had been overwritten by a
  debug `n_eval=2` smoke run and shipped, contradicting its own headline (`validated:false`). Re-ran the
  full **n=300** benchmark (now `bb9bdfae` = EV-009) and added `tests/test_metrics_integrity.py`
  (CI-enforced) so any smoke-scale **or hash-drifted** metrics artifact fails the build.
- **LIME correctness.** `make_lime_explainer` now passes `categorical_features` (the one-hot dummy
  indices), so LIME samples them as categorical rather than off-manifold continuous. Re-ran faithfulness
  (German Credit + GMSC) and ROAR; the finding held and LIME strengthened slightly (abs **+0.082** vs
  +0.076; ROAR LIME **0.113** ≈ TreeSHAP 0.108). New hashes: EV-011 `f3cc4f48`, EV-015 `04664b46`.
- **Wording / hygiene.** Opinion "demonstrably faithful" → "shown faithful … corroborated by ROAR";
  governance-doc integrity reframed to git blob-SHA content-addressing at the release tag; small-n
  fairness CI-vs-permutation divergence noted; GMSC prose figures reconciled to the n=300 re-run;
  reproduce path prepends data generation; risk register finalised; unused deps pruned.

Because the tag was moved, a clone pinned to the *previous* `v1.0` differs — use the current tag. The
pre-registration (`prereg-v1`, `HYPOTHESES.md`, OpenTimestamps Bitcoin-confirmed block 956533) is
**unchanged** and independent of this release tag.

## v1.0 — 2026-07-05 (initial)

Initial public release: pre-registered self-assessment audit of a high-risk EU AI Act credit-scoring
model — explanation faithfulness, fairness, robustness, recourse — with a governance/conformity pack,
signed audit opinion, and Bitcoin-timestamped pre-registration.
