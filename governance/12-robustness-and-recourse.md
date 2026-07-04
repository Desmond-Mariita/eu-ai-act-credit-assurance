# 12 — Robustness & Recourse (Reason Codes)

**Status:** results (the AUDITED, SHA-pinned LightGBM, German Credit; hardened through 2 internal +
3 external reviews). **Self-assessment.** Robustness → EU AI Act **Art. 15**; recourse/reason-codes →
**GDPR Arts. 13–15**. Decisions at the deployed **5:1 / P(bad) > 1/6** threshold, test split seed 0.
Numbers: `metrics/robustness_german_credit.json`, `metrics/reason_codes_german_credit.json`
(`scripts/50_robustness.py`, `scripts/60_reason_codes.py`). Neutral report.

## Robustness (Art. 15) — decision stability under input noise
Continuous features (loan duration, credit amount, age) perturbed by Gaussian noise = `eps ×
(feature train-std)`, clipped to the train range and rounded (ordinals + one-hots left intact); 10
draws/instance; fraction of accept/decline decisions that flip vs unperturbed, with a 95% instance-
bootstrap CI:

| eps (× feature std) | decision-flip rate | 95% CI | mean \|ΔP(bad)\| |
|---|---|---|---|
| 0.05 | 3.6% | [2.3, 5.0] | 0.025 |
| 0.10 | 6.9% | [5.2, 8.6] | 0.041 |
| 0.20 | 10.6% | [8.4, 13.0] | 0.067 |
| 0.50 | 16.3% | [14.0, 18.8] | 0.110 |

- **13.3% of applicants sit within 0.05 of the decision threshold** — the fragile band where flips
  concentrate.
- **Reading:** under the **chosen synthetic perturbation levels** (`eps` is a fraction of each
  feature's spread — *not* calibrated to real error logs), a small perturbation flips **≈4–7% of
  decisions**, rising to ~16% at large noise. The model is moderately stable; the sizeable near-
  threshold population means a non-trivial share of decisions are sensitive to small input changes —
  a documented Art. 15 concern, not a failure.

## Recourse / reason codes (GDPR Arts. 13–15)
For each of the **151 declined test applicants**, an **exhaustive integer grid/line search** over the
genuinely actionable, independently-negotiable loan terms — **loan duration** and **credit amount**
(every valid integer in the train range) — scored by the **pinned model**, asks whether any change
reaches the **deployed accept** (`P(bad) ≤ 1/6`). (The 94% is grid-resolution-stable: coarse, dense,
and every-integer grids give identical results, since the tree model is a step function.) (Installment rate is excluded — it is coupled to amount/
duration/income, not independently actionable; age/residence/dependents are immutable/protected;
one-hots are never touched.)

| Metric | Value | 95% CI (Wilson) |
|---|---|---|
| **Have loan-term recourse** (reach P ≤ 1/6) | **94.0%** | [89.1, 96.8] |
| **No loan-term recourse** (truly infeasible) | **6.0%** | [3.2, 10.9] |
| Recourse via a **single** feature change | 73.5% | — |
| Mean actionable features changed | 1.22 | — |
| Reason-code frequency (range-normalised) | loan_duration 86, credit_amount 87 | — |

- **Recourse is broadly available and simple:** 94% of declined applicants could reach acceptance by
  changing loan terms, and 73.5% by changing **just one** — **roughly evenly split between shortening
  the loan term and reducing the amount** (e.g. 60 → 9 months, or 2 124 → 1 479 DM). (The minimal
  change is chosen by a range-normalised distance, so months and DM are comparable — a raw-unit
  comparison spuriously favours duration.)
- **A genuine (small) infeasible core:** ~**6%** cannot reach acceptance by *any* loan-term change —
  their decline rests on **non-actionable creditworthiness factors** (history, purpose, account
  status). For them, "meaningful information about the logic" (Art. 13–15) must explain *why*, since
  no loan-term recourse exists.
- **Methodological caution (a reusable audit lesson):** an initial **DiCE random-search** approach
  found recourse for only **72.5%** (a superseded run — not recomputed in the current script) — a
  **~20-point under-report** vs the exhaustive grid, and it emitted **non-integer** loan terms (e.g.
  13.1 months). **Off-the-shelf counterfactual tools can
  conflate search-failure with infeasibility**; recourse infeasibility should be established by
  exhaustive/verified search on a valid feature grid, as done here.

## Limitations
Robustness: synthetic Gaussian noise (not calibrated to real data-quality evidence), continuous
features only (age perturbed as generic input noise), single split. Recourse: grid is exhaustive over
the **train range at integer resolution** for two features (2-D grid for the non-single-feature
cases); it is **model-side** — real-world feasibility (can an applicant actually borrow less or take a
shorter term?) and business rules are **not** assessed; targets `P(bad) ≤ 1/6` only. Self-assessment,
not independent.

## Governance implication (Art. 15, GDPR Arts. 13–15)
Art. 15: the model is moderately robust but has a **~13% near-threshold population** whose decisions
flip under small input noise — the deployer should monitor stability there and consider review/
abstention bands. GDPR 13–15: **actionable recourse exists for ~94% of declined applicants** (mostly a
shorter loan term) and should be surfaced as reason codes; the **~6% without loan-term recourse** need
a genuine explanation of the driving factors, not a false promise of recourse. No "compliant/robust
enough" opinion is expressed — this is the evidence Arts. 15 / 13–15 require the provider/deployer to
act on.
