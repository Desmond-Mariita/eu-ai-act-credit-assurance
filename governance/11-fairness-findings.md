# 11 — Fairness Findings

**Status:** results (audited LightGBM, German Credit). **Self-assessment.** EU AI Act **Art. 10(2)(f-g)**
(examination for bias) + **Art. 15** (accuracy/robustness) + GDPR (special-category-adjacent proxies).
Group metrics at the **cost-sensitive 5:1 threshold** (`P(bad) > 1/6` → decline), test n=300, seed 0.
Numbers: `metrics/fairness_german_credit.json` (`scripts/40_fairness.py`). Neutral report.

## Headline
The model shows **notable group disparities, all disadvantaging the historically worse-off group** —
women, foreign workers, and younger applicants are declined more often. The disparities are **material
but partly reflect base-rate differences in this dataset and are amplified by the aggressive 5:1
decline threshold**; several reference subgroups are **too small for confident conclusions**. Under
Art. 10 these must be documented and their justification/necessity examined by the deployer — which is
the point of this section.

## Disparities (selection_rate = fraction DECLINED; "bad" == 1 == declined)
| Attribute | Group (n) | decline rate | TPR (recall of bad) | FPR (good wrongly declined) | accuracy | DP diff | EO diff |
|---|---|---|---|---|---|---|---|
| **sex** | female (100) | 0.57 | 0.77 | **0.47** | 0.61 | **0.10** | **0.12** |
|  | male (200) | 0.47 | 0.79 | 0.35 | 0.69 | | |
| **foreign_worker** | yes (285) | 0.52 | 0.77 | 0.40 | 0.65 | **0.25** | **0.25** |
|  | no (15) | 0.27 | 1.00 | 0.15 | 0.87 | | |
| **age_band** | <25 (55) | **0.64** | 0.80 | 0.50 | 0.64 | **0.24** | **0.35** |
|  | 25-39 (160) | 0.51 | 0.84 | 0.39 | 0.68 | | |
|  | 40-59 (78) | 0.40 | 0.67 | 0.32 | 0.68 | | |
|  | 60+ (7) | 0.57 | 0.50 | 0.67 | 0.43 | | |

## Interpretation (honest)
- **Sex:** the clearest well-powered disparity — women are declined ~10 pts more and, critically, have
  a **higher false-positive rate** (0.47 vs 0.35): *good* female applicants are wrongly declined more
  often. This is not explained away by base rates alone (recall/TPR is similar across sexes), so it is
  a genuine equalised-odds gap worth mitigation consideration.
- **Foreign-worker & age:** large point disparities, but the advantaged reference groups are tiny
  (foreign=no n=15; 60+ n=7), so the magnitudes are **low-confidence**. The monotone age trend
  (younger → more declines) is consistent across the better-powered bands (<25 > 25-39 > 40-59).
- **Threshold effect:** the 5:1 cost matrix pushes an aggressive overall decline rate (~0.50), which
  *amplifies* absolute group gaps. A less aggressive threshold would shrink them (fairness–cost
  trade-off the deployer must own).
- **Data vs model:** German Credit's base "bad" rates differ by group; part of each disparity is
  inherited from the training distribution (Art. 10 data-governance issue) rather than created by the
  model. Separating the two rigorously (e.g., conditional on legitimate risk factors) is future work.

## Limitations
Single stratified split (no CI on the disparities); small subgroups (foreign=no, 60+, <25) → indicative
only; proxies (sex derived from personal-status; no direct protected labels); the 5:1 threshold is
taken as given; **no mitigation applied** — this is assessment, not remediation. Self-assessment, not
independent.

## Governance implication (Art. 10(2)(f-g), Art. 15, GDPR)
Art. 10 requires examining datasets for possible biases. This audit **documents** material,
consistently-directioned disparities (notably the female equalised-odds gap) and their caveats. A
deployer should (a) determine whether each disparity is *justified and necessary* for the legitimate
aim, (b) assess mitigation (e.g., Fairlearn reductions / threshold-per-group / reweighing) against the
cost trade-off, and (c) re-examine the **training data** base rates as the likely upstream source.
This section expresses **no opinion** that the model is or is not "fair" — it provides the evidence
Art. 10 requires the provider/deployer to act on.
