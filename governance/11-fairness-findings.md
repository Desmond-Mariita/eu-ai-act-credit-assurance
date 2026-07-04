# 11 — Fairness Findings

**Status:** results (the AUDITED, SHA-pinned LightGBM, German Credit). **Self-assessment.** EU AI Act
**Art. 10(2)(f-g)** (examination for bias) + **Art. 15**. The attributes are protected under **EU
non-discrimination law** (equal-treatment) — **not** GDPR Art. 9 special-category data (sex and age
are not Art. 9 categories). Group metrics at the **cost-sensitive 5:1 threshold** (`P(bad) > 1/6` →
decline), test n=300, seed 0, **2000-sample bootstrap CIs**. Numbers: `metrics/fairness_german_credit.json`
(`scripts/40_fairness.py`). Neutral report.

## Headline
The model's decisions show disparities that **consistently disadvantage the historically worse-off
group** — women, foreign workers, and younger applicants are declined more — **but at n=300 none is
statistically distinguishable from zero** on the valid (signed) test. These are **directional flags
for monitoring and a mitigation trade-off study, not established gaps.** Two things to be careful
about: (1) the model **trains on personal-status (a sex proxy) and age directly**, so these protected
attributes are *model inputs* — a bias source Art. 10 requires flagging; (2) the folded DP/EO
statistics can *look* significant but are not (see below). This section provides the evidence Art. 10
obliges the provider/deployer to examine; it expresses **no opinion** that the model is or isn't "fair".

## Disparities (decisions at 5:1; selection_rate = fraction DECLINED; bad==1)
| Attribute | Group (n) | base-rate bad | decline | FPR | AUROC | inferable? |
|---|---|---|---|---|---|---|
| **sex** | female (100) | 0.34 | 0.57 | 0.47 | 0.74 | yes |
|  | male (200) | 0.28 | 0.47 | 0.35 | 0.78 | yes |
| **foreign_worker** | yes (285) | — | 0.52 | 0.40 | — | yes |
|  | no (15) | — | 0.27 | 0.15 | — | **no (n<30)** |
| **age_band** | <25 (55) | — | 0.64 | 0.50 | — | yes |
|  | 25-39 (160) | — | 0.51 | 0.39 | — | yes |
|  | 40-59 (78) | — | 0.40 | 0.32 | — | yes |
|  | 60+ (7) | — | 0.57 | 0.67 | — | **no (n<30)** |

## Statistical significance (this is the correction the review demanded)
| Attribute | DP diff [folded CI] | EO diff [folded CI] | **valid signed test** |
|---|---|---|---|
| sex | 0.10 [0.01, 0.22] | 0.12 [0.03, 0.27] | **FPR diff (♀−♂) 95% CI includes 0** (≈[−0.03, 0.26]); decline-diff CI includes 0; permutation p≈0.10 |
| foreign_worker | 0.25 [0.03, 0.47] | 0.25 [0.18, 0.82] | **decline diff (yes−no) 95% CI includes 0** ([−0.02, 0.47]) |
| age_band | 0.24 [0.12, 0.64] | 0.35 [0.23, 0.90] | multi-group; folded only — treat as exploratory |

**Read this carefully:** DP and EO are **non-negative max-difference** statistics, so their bootstrap
CIs are **biased away from 0** and their apparent exclusion of 0 is **not** a valid significance test.
The **signed pairwise differences** (which *can* be negative) are the valid test, and for both binary
attributes **they include 0**. So: the disparities are real as **point estimates** and consistent in
direction, but **not statistically significant** at this sample size.

## Interpretation (honest)
- **Sex** is the largest well-powered *point* estimate — women face a ~10-pt higher decline rate and a
  higher FPR (0.47 vs 0.35, i.e. good female applicants wrongly declined more) — **but it is not
  significant** (signed FPR-diff CI spans 0, p≈0.10; the FPR rests on only 66 true-good women).
  Threshold-independent AUROC mildly corroborates slightly worse ranking for women (0.74 vs 0.78).
  Base rates differ (female 0.34 vs male 0.28 bad), so *some* decline-rate gap is expected under a
  single threshold on calibrated scores; the FPR/EO view conditions on the true label and would, if
  significant, indicate more than base rates — here it is not.
- **Foreign-worker & age:** point disparities are large but the advantaged reference groups are tiny
  (foreign=no n=15; 60+ n=7) → **not inferable**, excluded from any conclusion. The monotone age trend
  holds across the inferable bands (<25 > 25-39 > 40-59) but its DP/EO are folded-only.
- **Threshold:** the aggressive 5:1 decline rate (~0.50) inflates absolute gaps; a different threshold
  **may change** them (not demonstrated here — no threshold sweep).

## Limitations
Single stratified split (CIs from bootstrap, not repeated splits); small subgroups suppressed as
non-inferable; sex is a **proxy from personal-status, which conflates marital status** (A92 = female
div/sep/**married**); age cutpoints arbitrary; the 5:1 threshold taken as given; **no mitigation
applied** — assessment, not remediation; no full base-rate/model decomposition or threshold sweep.
Self-assessment, not independent.

## Governance implication (Art. 10(2)(f-g), Art. 15)
Art. 10 requires examining datasets/models for bias. This audit documents disparities that are
**directionally consistent but not statistically significant** at n=300, and flags that the model
**uses personal-status and age as inputs**. A deployer should (a) treat these as **monitoring flags**
and re-run at larger n, (b) study whether excluding/mitigating the protected inputs (Fairlearn
reductions / per-group thresholds / feature removal) is warranted against the cost trade-off, and (c)
examine the **training-data base rates** as a likely upstream source. No "fair/unfair" opinion is
expressed — this is the evidence Art. 10 requires the provider/deployer to act on.
