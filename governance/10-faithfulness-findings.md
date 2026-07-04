# 10 — Explanation-Faithfulness Findings

**Status:** results (produced AFTER pre-registration `prereg-v1`; hardened through 2 internal + 3
external model reviews + 1 internal + 3 external confirmation reviews before finalising).
**Self-assessment**, not an independent audit (see `00-scenario.md` §7). Primary metric:
**signed AOPC comprehensiveness** of top-k logical-group erasure (pre-registered); plus a **validated,
direction-agnostic absolute-movement** metric. Full numbers: `metrics/faithfulness_german_credit.json`.
Reproduce: `python scripts/30_faithfulness.py` (seed 0, n=300 test, n_perms=40, m=25; audited model
SHA256 in `00-scenario.md` §5). Perturbation = **approximate conditional (kNN-donor) imputation**
(residual off-manifoldness measured, not eliminated).

## Headline
**Whether these credit-model explanations register as "faithful" depends on the faithfulness
*metric*, not just the model.** Under a **direction-agnostic movement** metric — **validated** (a
random-ranking control sits at its floor) — both **TreeSHAP and LIME are faithful** even under
approximate conditional perturbation. Under the **pre-registered signed** metric, the same
perturbation yields a **null** — no ranking (real or control) separates from a random floor — because
erasing risk-increasing vs protective features moves `P(bad)` in opposite directions and the signed
average cancels. This is an **underpowered-null / metric-design result, not evidence that the
explanations are unfaithful.** The governance-relevant caution: **faithfulness claims must name their
perturbation regime AND sign convention**, and a single signed off-/on-manifold score can mislabel a
genuinely informative explainer.

## Pre-registered hypotheses — reported as observed
| # | Pre-registered claim | Result | |
|---|---|---|---|
| **H1** | TreeSHAP faithful (signed, conditional) | **Refuted** — diff-vs-floor 0.006, CI [−0.012, 0.025] | ✗ |
| **H2** | LIME less faithful than TreeSHAP | **Supported** (mean ordering under both metrics; not a paired-CI materiality test) | ~ |
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
| conditional (approx. kNN-donor) | +0.010 | −0.011 | +0.001 | +0.017 | +0.004 | none |
| marginal | −0.016 | −0.040 | −0.006 | +0.004 | −0.002 | none |
| baseline (off-manifold leverage) | +0.105 | +0.106 | +0.032 | +0.044 | +0.031 | TreeSHAP, LIME |

## Results — absolute (movement) metric, conditional regime (diff vs the absolute floor)
| Ranking | diff-vs-abs-floor | 95% CI | faithful? | role |
|---|---|---|---|---|
| TreeSHAP | **+0.106** | [0.094, 0.118] | **True** | explainer |
| LIME | **+0.076** | [0.066, 0.086] | **True** | explainer |
| **clean shuffled-SHAP** | +0.003 | [−0.005, 0.012] | **False** | **validates the metric (random → at floor)** |
| label-shuffled model | +0.044 | [0.034, 0.053] | True | confounded control (mild real signal, now detectable) |

**Instrument validation:** the clean random control sits at the floor under the **signed**,
**absolute**, and **baseline** tests — so a real explainer "beating the floor" is a genuine
faithfulness signal, not an artifact of the metric. The absolute metric is therefore *validated*, not
merely asserted.

- **LIME multi-seed (0,1,2) top-5 stability = 0.79** (LIME default `discretize=True`): moderately, not
  perfectly, stable.

## Deviations from pre-registration (full list in the metrics JSON)
1. **LIME fix.** The first run used `discretize_continuous=False`, under which LIME weights are raw
   scaled sensitivities, not instance contributions. Corrected to `discretize=True` and re-run (also
   removed a spurious stability = 1.0).
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
sentinels + `age<18` cleaned per the DQ profile; `scripts/06_gmsc_prep.py`). LightGBM **AUROC 0.829**;
benchmark n=200, K=5, n_perms=30. Full numbers: `metrics/faithfulness_gmsc.json`.

**Replicates (the core finding holds on a different dataset):**
- **Absolute metric validated** — clean control at floor (−0.004, CI [−0.008, −0.001], not faithful).
- **Both explainers faithful under movement** — TreeSHAP +0.019 [0.014, 0.026], LIME +0.014 [0.010,
  0.018]; **TreeSHAP > LIME** (H2).
- **Baseline** — both faithful, clean control at floor. Clean control sits at floor under every
  metric/regime → the instrument is robust across datasets.

**Differs (dataset-specific — and it is the *fragile* pieces that differ):**
- **Signed on-manifold test has power on GMSC** — TreeSHAP is faithful under the pre-registered signed
  metric (H1 *supported* here), unlike German Credit's underpowered null. GMSC's all-numeric features
  suffer less sign-cancellation than German Credit's one-hot groups.
- **H3 refuted on GMSC** — the label-shuffled control beats the *signed* floor (holds-fragile on German
  Credit). The confounded control's behaviour is dataset-specific; the clean control is the robust one.
- **LIME stability higher** (0.91 vs 0.79; fewer features).

**Takeaway:** the headline — faithfulness is metric-dependent, both explainers are faithful under a
*validated* movement metric (TreeSHAP > LIME), and a single signed on-/off-manifold score is
unreliable — **replicates**. The pieces that do NOT replicate (the signed null, the label-shuffled
confound) are precisely the fragile ones, underscoring that faithfulness verdicts are metric- **and**
dataset-sensitive.

## ROAR anchor (retrain-based cross-check)
An independent **RemOve-And-Retrain** check (Hooker et al. 2019): remove the top-k logical features by
each explainer's global importance, **retrain**, and measure test AUROC — which avoids the off-manifold
artefact of perturbation-only tests. Mean AUROC drop over k=1..10 (**larger = more faithful**; base
AUROC 0.769; `metrics/roar_german_credit.json`, EV-015):

| ranking | mean AUROC drop |
|---|---|
| TreeSHAP | **0.108** |
| LIME | **0.104** |
| model gain | 0.093 |
| random | 0.042 |

Both explainers degrade accuracy **~2.5× more than random** → they identify genuinely predictive
features, and **TreeSHAP ≥ LIME** (consistent with H2). This **corroborates the movement-metric
finding via an independent retrain-based methodology** — strengthening confidence that the "both
faithful, TreeSHAP > LIME" result is not an artefact of any single test.

## Governance implication (EU AI Act Art. 86 / GDPR Arts. 13–15)
"Meaningful information about the logic involved" presupposes the explanation is *faithful*. This
audit shows faithfulness is **conditional on how it is measured**: the audited SHAP/LIME explanations
*do* carry genuine ordering information (they beat a validated random control under movement and
leverage tests, and TreeSHAP > LIME), **but** a naive signed on-/off-manifold comprehensiveness score
would have mislabelled them. A deployer presenting these explanations to a loan officer or applicant
should (a) state the perturbation regime and direction convention under which faithfulness was
established, and (b) not treat a single comprehensiveness score as sufficient evidence — which is
itself the governance-relevant finding.
