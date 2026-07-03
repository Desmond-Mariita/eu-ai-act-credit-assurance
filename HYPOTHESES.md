# Pre-registration — Tabular Explanation-Faithfulness Audit

**Status:** PRE-REGISTERED. Written and committed **before any model is trained or any faithfulness
result is produced.** The measurement instrument was frozen first (cores + tests, commit `2521586`,
double-reviewed). This file is bound by a signed git tag `prereg-v1`, an OpenTimestamps `.ots`
proof, and its SHA256 recorded in `governance/09-evidence-index.md`.

**Scope:** an *investigation* of behavioural, perturbation-faithfulness — NOT a proof of "true"
faithfulness. Neutral reporting: whatever the data show is the result.

## 1. What is being tested
Whether the feature attributions from explainers applied to the audited **LightGBM** credit model
are *faithful* — i.e. whether erasing the features an explainer calls important actually moves the
model's `P(bad)` more than erasing random features.

- **Explainers under audit:** **TreeSHAP**, **LIME** (multi-seed).
- **Negative controls:** a **random-attribution** ranking and a **label-shuffled** attribution
  (fit on permuted labels). These are known-bad; the instrument must flag them as unfaithful.
- **Datasets:** German Credit (mainline) + Give Me Some Credit (generalization check).
- **Perturbation:** whole feature-**groups**; primary regime = **conditional** (on-manifold, kNN
  donor in standardised space); secondary = marginal, baseline. OOD distance reported per regime.

## 2. Operationalised definition of "faithful" (fixed before results)
For an explainer on a dataset, over the held-out test instances:
- **comprehensiveness(k)** = base `P(bad)` − `P(bad)` after erasing the top-k ranked groups
  (k = 1 … K, **K = ⌈50% of logical features⌉**, justified cap);
- aggregate = mean comprehensiveness AOPC with an **instance-level bootstrap** (B = 2000) CI;
- the **random floor** = expected comprehensiveness under random group orderings.

An explainer is declared **FAITHFUL** iff its mean AOPC **exceeds the random floor by ≥ 2 bootstrap
SE _and_ the bootstrap CI of (AOPC − floor) excludes 0**. Reported alongside: **sufficiency** (low =
good) and **LIME multi-seed stability** (an unstable explainer is unfaithful in a second sense).

## 3. Directional hypotheses (predicted before running)
- **H1 — TreeSHAP is faithful.** By its local-accuracy axiom on tree models, TreeSHAP's AOPC is
  *high* and clears the faithfulness bar on both datasets.
- **H2 — LIME < TreeSHAP.** LIME's AOPC is materially lower than TreeSHAP's (surrogate approximation
  + instability).
- **H3 — Negative controls land at the floor.** Random-attribution and label-shuffled attributions
  do **not** clear the bar (AOPC ≈ floor) — instrument validity.
- **H4 — OOD ordering:** mean OOD distance conditional < marginal < baseline.

## 4. Pre-specified headline (data-driven, honest whatever the result)
- **If H1 ∧ H2 hold:** *"The LIME-style explanations often shown to credit officers are markedly
  less faithful than exact Shapley values — a concrete governance reason to standardise on the
  faithful explainer."* (AI Act Art. 86 / GDPR Arts. 13–15 implication.)
- **If TreeSHAP also fails the bar:** report *"neither production explainer is perturbation-faithful
  on this model,"* with limitations.
- **If results contradict H1–H4:** report them as observed. **No adverse finding is manufactured;
  a clean result is published with limitations and residual risks.**

## 5. Reproducibility preconditions (part of the freeze)
- Inputs are finite **float64**; all seeds fixed; the **donor pool is the held-out train split**
  (no self-match leakage); the per-call RNG is derived from `(seed, group_ids, x)`.
- The instrument is frozen at commit `2521586`; this pre-registration at tag `prereg-v1`.
- Power note: mainline test n ≈ 300 → **subgroup faithfulness claims are caveated** as low-powered.

## 6. Explicitly NOT claimed
"True" or mechanistic faithfulness; any explainer's behaviour on models other than the audited
LightGBM; that the conditional regime is guaranteed on-manifold (it is *approximate*; residual OOD
is measured and reported).
