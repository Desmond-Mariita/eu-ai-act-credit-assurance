# 10 — Explanation-Faithfulness Findings

**Status:** results (produced AFTER pre-registration `prereg-v1`; reviewed by 2 internal + 3 external
model reviewers before writing). **Self-assessment**, not an independent audit (see `00-scenario.md`
§7). Metric: **signed AOPC comprehensiveness** of top-k logical-group erasure (primary, pre-
registered) + a **direction-agnostic absolute-movement** diagnostic. Full numbers: `metrics/
faithfulness_german_credit.json`. Reproduce: `python scripts/30_faithfulness.py` (seed 0, n=300 test,
n_perms=40, m=25; audited model SHA256 in `00-scenario.md` §5).

## Headline
**Whether these credit-model explanations register as "faithful" depends on the faithfulness
*metric*, not just the model.** Under a **direction-agnostic movement** metric, both **TreeSHAP and
LIME are clearly faithful** even under on-manifold perturbation. Under the **pre-registered signed**
metric, on-manifold perturbation yields a **null** — no ranking (real or control) beats a random
floor — because erasing risk-increasing vs protective features moves `P(bad)` in opposite directions
and the signed average cancels. This is an **underpowered-null / metric-design result, not evidence
that the explanations are unfaithful.** The negative controls behave (land at the floor), so the
instrument is validated; the practical caution for deployers is that **faithfulness claims must name
their perturbation regime and sign convention.**

## Pre-registered hypotheses — reported as observed
| # | Pre-registered claim | Result | |
|---|---|---|---|
| **H1** | TreeSHAP faithful (signed, on-manifold) | **Refuted** — diff-vs-floor 0.005, CI [−0.011, 0.023] | ✗ |
| **H2** | LIME less faithful than TreeSHAP | **Supported** — LIME < TreeSHAP under both signed and absolute metrics | ✓ |
| **H3** | Negative control lands at the floor | **Holds** — label-shuffled control not significantly above floor (+0.017, CI incl. 0) | ✓ |
| **H4** | OOD conditional < marginal < baseline | **Refuted** — baseline lowest (5.04) < conditional (5.45) < marginal (5.70) | ✗ |

Two of four predictions were wrong. They are reported as-is; no adverse finding was manufactured.

## Results (AOPC comprehensiveness vs random floor; "faithful" = diff CI excludes 0 AND ≥ 2·bootstrap SE)
| Regime / metric | TreeSHAP | LIME | clean control | label-shuffled | floor | faithful? |
|---|---|---|---|---|---|---|
| conditional — **signed** (primary) | +0.010 | −0.011 | +0.001 | +0.017 | +0.004 | none |
| conditional — **absolute (movement)** | **+0.106** `[.094,.118]` | **+0.076** `[.066,.086]` | — | — | (abs floor) | **TreeSHAP, LIME** |
| baseline — signed (off-manifold leverage) | **+0.105** | **+0.106** | +0.032 | +0.044 | +0.031 | **TreeSHAP, LIME** |
| marginal — signed | −0.016 | −0.040 | −0.006 | +0.004 | −0.002 | none |

- **Instrument validation:** the clean shuffled-SHAP control sits at/below the floor in every regime;
  the pre-registered label-shuffled control is not significantly above it. Real explainers separate
  from controls under the absolute and baseline tests.
- **LIME multi-seed (0,1,2) top-5 stability = 0.79** (with LIME's default `discretize=True`); LIME is
  moderately, not perfectly, stable.

## Deviations from pre-registration (full list in the metrics JSON)
1. **LIME fix.** The first run used `discretize_continuous=False`, under which LIME weights are raw
   scaled sensitivities, not instance contributions. Corrected to `discretize=True` and re-run; this
   also removed a spurious stability = 1.0.
2. **Signed → absolute.** The pre-registered metric is signed; a direction-agnostic absolute variant
   (+ its own random floor) was added post-hoc for the conditional regime after review showed sign
   cancellation drives the signed null.
3. **H3 control.** The pre-registered label-shuffled control is reported as-is (it holds once floor
   variance is estimated per-instance). A clean shuffled-SHAP control is an added post-hoc diagnostic.

## Limitations (do not over-read)
- **Baseline is a global-leverage test**, not local on-distribution faithfulness: median/modal
  replacement rewards attributions pointing at high-main-effect features.
- The **absolute metric's own clean-control check** (clean control at the abs floor) is not yet run —
  a stated next step.
- **OOD** figures are a diagnostic (TreeSHAP top-3 erasure, first 30 instances), not a regime
  characterisation; do not infer on/off-manifoldness from them.
- **Power:** n≈300, high per-instance variance; subgroup and dataset generalisation (Give Me Some
  Credit) are pending. Single model, single dataset. Self-assessment, not independent.

## Governance implication (EU AI Act Art. 86 / GDPR Arts. 13–15)
"Meaningful information about the logic involved" presupposes the explanation is *faithful*. This
audit shows faithfulness is **conditional on how it is measured**: a deployer presenting SHAP/LIME
explanations to a loan officer or applicant should (a) state the perturbation regime and direction
convention under which faithfulness was established, and (b) not treat a single off-manifold
comprehensiveness score as sufficient evidence. The audited explanations *do* carry genuine ordering
information (they beat random controls under movement and leverage tests) — but the naive on-manifold
signed test would have mislabelled them, which is itself the governance-relevant finding.
