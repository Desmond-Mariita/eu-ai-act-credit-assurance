# 10 — Explanation-Faithfulness Findings

**Status:** results (produced AFTER pre-registration `prereg-v1`; hardened through repeated internal +
external model-review/audit rounds — a full-execution audit drove the group-valid LIME fix and the
honest metric reframing here).
**Self-assessment**, not an independent audit (see `00-scenario.md` §7). Primary metric:
**signed AOPC comprehensiveness** of top-k logical-group erasure (pre-registered); plus an **exploratory
direction-agnostic absolute-movement** diagnostic (calibration-checked, corroborated by ROAR). Full
numbers: `metrics/faithfulness_german_credit.json`.
Reproduce: `python scripts/30_faithfulness.py` (seed 0, n=300 test, n_perms=40, m=25; audited model
SHA256 in `00-scenario.md` §5). Perturbation = **approximate conditional (kNN-donor) imputation**
(residual off-manifoldness measured, not eliminated).

## Headline
**Whether these credit-model explanations register as "faithful" depends on the *metric*, the *sign
convention*, AND the *perturbation regime* — not just the model.** Under a **direction-agnostic
movement** diagnostic (the clean random control sits at its floor — a *calibration* check, not
construct validation), both **TreeSHAP and LIME beat the floor**, and a retrain-based **ROAR** anchor
confirms both explainers' *global* rankings are strongly predictive. But the picture is nuanced: under
the **pre-registered signed** metric (conditional regime) **TreeSHAP is inconclusive** (diff-vs-floor
CI spans 0) while **LIME separates *below* the floor** (erasing LIME's top features moves `P(bad)` the
wrong way on average) — so this is **NOT a uniform "underpowered null"**; and at **baseline (leverage)
only TreeSHAP** clears the bar. **Net: TreeSHAP is the more consistent explainer; LIME registers as
faithful only under the movement / ROAR view.** Governance caution: faithfulness claims must name their
metric, sign convention, and regime — a single score can mislabel an explainer in either direction.

## Pre-registered hypotheses — reported as observed
| # | Pre-registered claim | Result | |
|---|---|---|---|
| **H1** | TreeSHAP faithful (signed, conditional) | **Refuted** — diff-vs-floor 0.006, CI [−0.012, 0.025] | ✗ |
| **H2** | LIME less faithful than TreeSHAP | **Broadly supported** — TreeSHAP > LIME under movement (both datasets); at baseline & under the signed metric only TreeSHAP is faithful (LIME below floor); ROAR *global* rankings **indistinguishable** | ✓~ |
| **H3** | Negative control lands at the floor | **Holds within resolution — FRAGILE** (see below) | ~ |
| **H4** | OOD conditional < marginal < baseline | **Refuted** — baseline lowest (5.04) < conditional (5.45) < marginal (5.70) | ✗ |

Two predictions refuted, one supported, one fragile. Reported as-is; no adverse finding manufactured.

**H3 fragility (disclosed):** the pre-registered label-shuffled control's *signed* diff-vs-floor is
`0.0125, CI [−0.0001, 0.0255]` — it includes 0 by one ten-thousandth, so H3 "holds" only within
estimation noise, and it **does beat the absolute floor** (below). A LightGBM on permuted labels still
splits high-information features (top-5 Jaccard vs the real ranking ~0.30 vs 0.16 random), so this
control carries a mild real signal. The clean shuffled-SHAP control (a truly random ranking) is the
robust negative control and sits at the floor under every metric.

## Results — signed AOPC comprehensiveness (mean; "faithful" = diff-vs-floor CI excludes 0 AND ≥ 2·bootstrap SE)
| Regime | TreeSHAP | LIME | clean control | label-shuffled | floor | faithful |
|---|---|---|---|---|---|---|
| conditional (approx. kNN-donor) | +0.010 | −0.026 | +0.001 | +0.017 | +0.004 | none (LIME **below** floor: −0.030 [−0.046, −0.014]) |
| marginal | −0.016 | −0.050 | −0.006 | +0.004 | −0.002 | none |
| baseline (off-manifold leverage) | +0.105 | +0.048 | +0.032 | +0.044 | +0.031 | **TreeSHAP only** |

## Results — absolute (movement) metric, conditional regime (diff vs the absolute floor)
| Ranking | diff-vs-abs-floor | 95% CI | faithful? | role |
|---|---|---|---|---|
| TreeSHAP | **+0.106** | [0.094, 0.118] | **True** | explainer |
| LIME (group-valid) | **+0.090** | [0.080, 0.100] | **True** | explainer |
| **clean shuffled-SHAP** | +0.003 | [−0.005, 0.012] | **False** | **calibration: random ranking → at floor** |
| label-shuffled model | +0.044 | [0.034, 0.053] | True | confounded control (mild real signal — clears the bar) |

**Instrument calibration (NOT construct validation):** the clean random control sits at the floor
under the signed, absolute, and baseline tests — a *calibration* check that a genuinely random ranking
scores ~0. It is **not** validation against a ground-truth-faithful explanation, and the confounded
label-shuffled control *does* clear the absolute bar (so that bar is lenient). The absolute-movement
result is therefore **exploratory**, corroborated by the retrain-based ROAR anchor (below) — not
asserted as a "validated" faithfulness metric.

- **LIME multi-seed (0,1,2) top-5 stability = 0.80** (group-valid LIME in label-encoded space):
  moderately, not perfectly, stable.

## Deviations from pre-registration (full list in the metrics JSON)
1. **LIME fixes (three iterations, disclosed).** (i) `discretize_continuous=False` → `discretize=True`
   (raw sensitivities → instance contributions; removed a spurious stability = 1.0). (ii) Passing
   `categorical_features` still sampled the one-hot dummies *independently* — a full-execution audit
   showed **~99.9% of LIME neighbourhoods were off-manifold** (invalid multi-/zero-hot combinations).
   (iii) **Group-valid LIME:** LIME now runs in the **label-encoded logical space** with a re-expanding
   predict wrapper, so every sampled neighbour is a valid one-hot (verified 100%). The group-valid
   re-run keeps LIME faithful under the movement metric (abs **+0.090**) but shows it is the *weaker*
   explainer (below the floor under the signed metric; not faithful at baseline). GMSC (all-numeric) is
   unaffected.
2. **Signed → absolute.** The pre-registered metric is signed; a direction-agnostic absolute variant
   (+ its own floor, + both controls) was added post-hoc after review showed sign cancellation drives
   the signed conditional null.
3. **H3 control.** Reported as-is on the pre-registered label-shuffled control (fragile, see above); a
   clean shuffled-SHAP control was added as the robust post-hoc negative control.
4. **Floor variance.** The random floor now uses a per-instance seed (the earlier fixed seed
   understated variance and once made the label-shuffled control appear to beat the signed floor).

## Limitations (do not over-read)
- **Baseline is a global-leverage test**, not local on-distribution faithfulness (median/modal
  replacement rewards attributions pointing at high-main-effect features).
- **Perturbation** is *approximate* conditional kNN-donor imputation; residual off-manifoldness is
  measured (OOD diagnostic), not eliminated. The OOD figure is a diagnostic (TreeSHAP top-3 erasure,
  first 30 instances), **not** a regime characterisation.
- **Power:** n≈300, high per-instance variance; H2 is a mean-ordering check (no paired-CI materiality
  threshold). Subgroup generalisation pending; second-dataset (GMSC) generalisation **done** (above,
  subsampled to 6k). Single model per dataset. **Self-assessment, not independent.**

## Generalization check — Give Me Some Credit (GMSC)
Second dataset, structurally different: **all-numeric** (10 features, no one-hot groups), **imbalanced**
(~6.7% default), subsampled to 6,000 for a tractable conditional-kNN donor pool (807 past-due `96/98`
sentinels + `age<18` cleaned per the DQ profile; median imputation fit on the **train split only**,
`scripts/06_gmsc_prep.py` + `30_faithfulness.py`). LightGBM **test AUROC 0.827** (now stored in the
metrics JSON, EV-011 — a traceable value; the GMSC model is a generalization-check model, not the
audited German-Credit ToE); benchmark n=300, K=5, n_perms=40. Full numbers: `metrics/faithfulness_gmsc.json`.

**Replicates (the core finding holds on a different dataset):**
- **Absolute metric calibration-checked** — clean control at floor (−0.001, CI [−0.005, 0.002], not faithful).
- **Both explainers faithful under movement** — TreeSHAP +0.019 [0.014, 0.024], LIME +0.014 [0.011,
  0.017]; **TreeSHAP > LIME** (H2).
- **Baseline** — both faithful, clean control at floor. Clean control sits at floor under every
  metric/regime → the instrument is calibrated across datasets. *(NB: LIME is stronger on GMSC than on
  German Credit — the group-valid one-hot handling is the German-Credit-specific hard case.)*

**Differs (dataset-specific — and it is the *fragile* pieces that differ):**
- **Signed on-manifold test has power on GMSC** — TreeSHAP is faithful under the pre-registered signed
  metric (H1 *supported* here), unlike German Credit's underpowered null. GMSC's all-numeric features
  suffer less sign-cancellation than German Credit's one-hot groups.
- **H3 refuted on GMSC** — the label-shuffled control beats the *signed* floor (holds-fragile on German
  Credit). The confounded control's behaviour is dataset-specific; the clean control is the robust one.
- **LIME stability higher** (0.92 vs 0.80; fewer features, all numeric).

**Takeaway:** the headline — faithfulness is metric-dependent, both explainers beat the floor under the
(exploratory) movement metric with **TreeSHAP the more consistent**, and a single signed on-/off-manifold
score is unreliable — **replicates**. The pieces that do NOT replicate (German Credit's signed null, the
label-shuffled confound, LIME's below-floor signed result) are precisely the fragile / hard-case ones,
underscoring that faithfulness verdicts are metric- **and** dataset-sensitive.

## ROAR anchor (retrain-based cross-check, 8 splits)
An independent **RemOve-And-Retrain** check (Hooker et al. 2019): remove the top-k logical features by
each explainer's global importance, **retrain**, and measure test AUROC — over **8 stratified splits**
(mean ± sd), which avoids the off-manifold artefact of perturbation-only tests. Mean AUROC drop
(**larger = more faithful**; `metrics/roar_german_credit.json`, EV-015):

| ranking | mean AUROC drop (8 splits) |
|---|---|
| TreeSHAP | 0.108 ± 0.017 |
| LIME | 0.109 ± 0.021 |
| model gain | 0.098 ± 0.015 |
| random | 0.036 ± 0.011 |

Both explainers' **global rankings** degrade accuracy **~3× more than random on every one of the 8
splits** → they identify genuinely predictive feature *groups*. **Scope:** ROAR speaks to **global
feature-ranking utility**, not to the faithfulness of individual *local* explanations, and (with the
same 1,000 rows reused across correlated splits and an 8-sample floor) its "distinguishable" test is a
heuristic `|mean diff| > 2·SE`, not a formal significance test. On that heuristic ROAR does **not**
distinguish TreeSHAP from LIME (difference −0.001, TreeSHAP wins 4/8) — so the **TreeSHAP > LIME
ordering seen under the movement/baseline/signed views is metric-specific**, not universal for global
ranking. This **corroborates the "both faithful" result
via an independent retrain-based methodology**; the explainer *ordering* is itself metric-dependent.

## Governance implication (EU AI Act Art. 86 / GDPR Arts. 13–15)
"Meaningful information about the logic involved" presupposes the explanation is *faithful*. This
audit shows faithfulness is **conditional on how it is measured**: **TreeSHAP** carries genuine ordering
information across the movement, ROAR, and leverage views; **LIME** does under the movement / global-
ranking view (though it falls below the floor under the signed metric) — both beat a **calibrated
random-ranking control** (at the floor) — **but** a naive signed on-/off-manifold comprehensiveness score
would have mislabelled TreeSHAP as unfaithful. A deployer presenting these explanations to a loan officer or applicant
should (a) state the perturbation regime and direction convention under which faithfulness was
established, and (b) not treat a single comprehensiveness score as sufficient evidence — which is
itself the governance-relevant finding.
