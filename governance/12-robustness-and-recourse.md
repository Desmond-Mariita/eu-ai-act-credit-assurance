# 12 — Robustness & Recourse (Reason Codes)

**Status:** results (the AUDITED, SHA-pinned LightGBM, German Credit). **Self-assessment.** Robustness
→ EU AI Act **Art. 15**; recourse/reason-codes → **GDPR Arts. 13–15** ("meaningful information" +
actionability). Decisions at the deployed **5:1 / P(bad) > 1/6** threshold, test split seed 0.
Numbers: `metrics/robustness_german_credit.json`, `metrics/reason_codes_german_credit.json`
(`scripts/50_robustness.py`, `scripts/60_reason_codes.py`). Neutral report.

## Robustness (Art. 15) — decision stability under input noise
Numeric features perturbed by Gaussian noise = `eps × (feature train-std)`, clipped to the training
range and rounded (one-hot categoricals left intact); 10 draws/instance; fraction of accept/decline
decisions that flip vs unperturbed:

| eps (× feature std) | decision-flip rate | mean \|ΔP(bad)\| |
|---|---|---|
| 0.05 | 3.7% | 0.023 |
| 0.10 | 6.7% | 0.039 |
| 0.20 | 10.9% | 0.069 |
| 0.50 | 16.8% | 0.111 |

- **13.3% of applicants sit within 0.05 of the decision threshold** — the fragile band where flips
  concentrate.
- **Reading:** at realistic small measurement noise (5–10% of a feature's spread) **≈4–7% of
  decisions flip.** The model is moderately stable, but the sizeable near-threshold population means a
  non-trivial share of decisions are sensitive to small input changes — a documented Art. 15 concern,
  not a failure.

## Recourse / reason codes (GDPR Arts. 13–15)
DiCE counterfactuals for **80 declined applicants**, varying ONLY genuinely **actionable loan
parameters** — loan duration, credit amount, installment rate — with **age/residence/dependents
excluded (immutable/protected)** and one-hot categoricals frozen (valid). Recourse checked against the
**deployed 1/6 policy** (reach `P(bad) ≤ 1/6`), not DiCE's 0.5 argmax.

| Metric | Value |
|---|---|
| A counterfactual was found for | 100% of declined applicants |
| Flips the **model class** (P < 0.5) | **100%** |
| Reaches the **deployed accept** (P ≤ 1/6) | **72.5%** |
| **No loan-parameter recourse** under the 1/6 policy | **27.5%** |
| Mean actionable features changed (sparsest recourse) | 1.45 |
| Reason-code frequency | loan_duration 36, credit_amount 31, installment_rate 17 |

- **The load-bearing finding:** for **27.5% of declined applicants, adjusting loan duration/amount/
  installment cannot reach acceptance** under the deployed threshold — their decline rests on
  **non-actionable creditworthiness factors** (history, purpose, account status). For them, "meaningful
  information about the logic" (Art. 13–15) must explain *why*, since no simple recourse exists.
- **When recourse exists it is sparse and sensible** — typically one or two changes, most often
  **shorten the loan / reduce the amount** (e.g. credit 10 144 → 2 752 DM + installment 2 → 1.5).
- **Model-behaviour flag:** some counterfactuals are **non-monotone** (e.g. one applicant just above
  threshold is accepted by *increasing* the requested amount 1 444 → 4 898), i.e. the model is not
  monotone in amount — worth a provider sanity-check.

## Limitations
Robustness: synthetic Gaussian noise (not a real threat model), numeric features only, single split,
no CIs on flip rates. Recourse: DiCE `random` method (stochastic, seeded); the actionable-feature set
is a judgement call; rates are from n=80 without CIs (indicative); recourse is model-side — real-world
feasibility (can an applicant actually borrow less?) is not assessed; targets `P(bad) ≤ 1/6` only.
Self-assessment, not independent.

## Governance implication (Art. 15, GDPR Arts. 13–15)
Art. 15: the model is moderately robust but has a **~13% near-threshold population** whose decisions
flip under small input noise — the deployer should monitor stability there and consider abstention/
review bands. GDPR 13–15: **actionable recourse exists for ~73% of declined applicants** (mostly via
loan terms) and should be surfaced as reason codes; the **~27% without loan-parameter recourse** need
a genuine explanation of the driving factors, not a false promise of recourse. The non-monotone
behaviour is a flag for the provider. No "compliant/robust enough" opinion is expressed — this is the
evidence Arts. 15 / 13–15 require the provider/deployer to act on.
