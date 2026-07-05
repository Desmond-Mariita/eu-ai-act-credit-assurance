# Data Documentation (EU AI Act Art. 10) — dictionary, CDE register, bias-representation note

**Status:** Phase 0.5 (plan Task 3). Numbers below are from `data_quality.json` / `feature_groups.json`
and the committed data snapshot (`data_manifest.sha256`). The data-quality profile is a
**completeness/error check (Art. 10(3))** and does **not** inform the pre-registered faithfulness
hypotheses.

## 0. Provenance & licensing
| Dataset | Source | Version / file | Licence / terms | Redistribution |
|---------|--------|----------------|-----------------|----------------|
| German Credit (mainline) | UCI Statlog (id 144) | `german_credit.parquet` (cleaned/encoded) | **CC BY 4.0** | raw git-ignored; snapshot hash published |
| Give Me Some Credit (generalization) | Kaggle "GiveMeSomeCredit" | `cs-training.csv` (raw) | per **Kaggle competition terms** | **not redistributed**; user acquires; verified locally by hash |

Raw files are excluded from git and verified against `data_manifest.sha256` (`python scripts/05_manifest_and_dq.py verify`).

## 1. Data dictionary — German Credit (Statlog, 20 attributes → 61 encoded columns)
Numeric attributes kept as-is; categoricals one-hot encoded (`AttributeN_value`); see `feature_groups.json`.

| Attr | Meaning | Type | Notes |
|------|---------|------|-------|
| 1 | Status of existing checking account | categorical (A11–A14) | strong predictor |
| 2 | Duration (months) | numeric | |
| 3 | Credit history | categorical (A30–A34) | |
| 4 | Purpose | categorical (A40–A410) | |
| 5 | Credit amount | numeric | |
| 6 | Savings account / bonds | categorical (A61–A65) | |
| 7 | Present employment since | categorical (A71–A75) | |
| 8 | Installment rate (% disposable income) | numeric | |
| 9 | Personal status & sex | categorical (A91–A94) | **derives `sex`** (protected/proxy) |
| 10 | Other debtors / guarantors | categorical (A101–A103) | |
| 11 | Present residence since | numeric | |
| 12 | Property | categorical (A121–A124) | |
| 13 | Age (years) | numeric | **protected attribute** |
| 14 | Other installment plans | categorical (A141–A143) | |
| 15 | Housing | categorical (A151–A153) | |
| 16 | # existing credits at this bank | numeric | |
| 17 | Job | categorical (A171–A174) | |
| 18 | # people liable for maintenance | numeric | |
| 19 | Telephone | categorical (A191–A192) | |
| 20 | Foreign worker | categorical (A201–A202) | **`foreign_worker`** (protected/proxy) |
| **y** | Target | binary | **1 = bad credit** (recoded from Statlog 1=good/2=bad) |

**Class balance:** 700 good (0) / 300 bad (1). **No missing values** in the cleaned parquet.

## 2. Data dictionary — Give Me Some Credit (10 numeric features)
Target `SeriousDlqin2yrs` (1 = serious delinquency in 2 yrs); features: revolving-utilization, age,
past-due counts (30-59/60-89/90+), debt ratio, monthly income, open credit lines, real-estate loans,
dependents. **Class balance:** 139,974 / 10,026 (**6.7 % positive** — imbalanced).
**Completeness (Art. 10(3)):** `MonthlyIncome` **29,731 missing** (~19.8 %); `NumberOfDependents`
**3,924 missing** (~2.6 %) — imputed at load; flagged as a documented data-quality risk (R5).
**Errors / anomalies (Art. 10(3)):** `age` has an implausible value (min = 0); the three past-due-count
columns each contain **269 sentinel values (96/98)** (likely data-entry codes); `RevolvingUtilization`
and `MonthlyIncome` show **extreme outliers** (max ≫ p99). Machine-recorded under `data_quality.json`
→ `anomalies`; these are not corrected at this stage, only documented (out-of-scope per ToE §6).

## 3. Critical Data Elements (CDE) register
Elements most material to the decision and/or carrying protected-attribute risk:

| CDE | Attribute | Why critical | Protected / proxy |
|-----|-----------|--------------|-------------------|
| Checking-account status | 1 | Dominant credit-risk signal | no |
| Credit history | 3 | Direct risk signal | no |
| Duration / Credit amount | 2 / 5 | Exposure sizing | no |
| Personal status & sex | 9 | Feeds decision **and** derives `sex` | **yes (proxy)** |
| Age | 13 | Predictor **and** protected attribute | **yes** |
| Foreign worker | 20 | Predictor **and** protected/proxy for national/ethnic origin | **yes (proxy)** |

## 4. Bias-representation note (Art. 10(2)(f))
Representation of protected/proxy groups in German Credit (n = 1,000):

| Attribute | Groups | Counts |
|-----------|--------|--------|
| Sex (from Attr. 9) | male / female | **690 / 310** (69 % / 31 %) |
| Foreign worker (Attr. 20) | foreign / non-foreign | **963 / 37** (96.3 % / 3.7 %) |
| Age band (Attr. 13; `<25 / 25-39 / 40-59 / 60+`, per `data._age_band`, shared with the fairness test) | <25 / 25-39 / 40-59 / 60+ | **149 / 552 / 248 / 51** |

**Representation risks:**
- **Severe skew in `foreign_worker`** (only 37 non-foreign) → any subgroup fairness/performance claim
  for the non-foreign group is **low-powered**; must be caveated (feeds risk R1, and the n≈300
  test-set power note pre-registered for faithfulness).
- **Sex imbalance** (2.2:1 male:female) → parity metrics reported with confidence intervals.
- `personal_status` couples marital status with sex — a documented **proxy-leakage** concern; handled
  under the protected/proxy-attribute policy (**not** GDPR Art. 9 special-category unless proxy-inferred).
