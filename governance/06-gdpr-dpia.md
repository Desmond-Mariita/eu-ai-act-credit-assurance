# 06 — Data Protection Impact Assessment (GDPR Art. 35)

**Status & nature:** a **template DPIA for the *simulated* deployment** (a fictional EU bank,
`00-scenario.md §2`). **No real personal data is processed** — German Credit is public research data;
this demonstrates the Art. 35 assessment methodology, it is not a DPIA for real processing.
**Self-assessment**, not independent (see `13 §E`). Draws its risk analysis from findings `10`–`12`.

## 1. Is a DPIA required? (Art. 35(1), 35(3)(a))
Yes. The processing is a **systematic and extensive evaluation of personal aspects based on automated
processing, including profiling**, on which **decisions producing legal or similarly significant
effects** (credit refusal) are based — the Art. 35(3)(a) trigger. It also meets EDPB WP248 criteria
(evaluation/scoring; automated decision with significant effect) — two suffice; no vulnerability of the
data subjects is asserted.

## 2. Systematic description of the processing (illustrative)
- **Data subjects / categories:** loan applicants; financial/behavioural attributes (account status,
  credit history, duration, amount, employment, housing), plus **personal-status/sex** and **age**.
- **Purpose & legal basis (Art. 6):** assess creditworthiness at origination — candidates: Art.
  6(1)(c) (legal obligation — responsible-lending/creditworthiness checks, often the strongest),
  6(1)(b) (steps prior to contract — EDPB Guidelines 2/2019 read this narrowly for scoring), and/or
  (f) legitimate interest; **to be fixed by the deployer**.
- **Flow (`00 §2`):** applicant → model `P(bad)` → cost-sensitive threshold → **human loan-officer
  review** → decision → logged → appeal path.
- **Special-category data (Art. 9):** **none processed as such.** **Sex is not an Art. 9 category**,
  but note the model **uses personal-status (a sex proxy) and age as inputs** — a proxy-inference and
  non-discrimination concern (see §5).

## 3. Art. 22 automated-decision trigger analysis
Art. 22(1) bites only on decisions **based *solely* on automated processing**. Here a loan officer
reviews the recommendation before the decision issues → **not solely automated *if* that review is
meaningful** (not a rubber-stamp). **Residual risk:** de-facto rubber-stamping would re-trigger
Art. 22. **Safeguards (Art. 22(3)):** right to obtain human intervention, to express a view, and to
contest — supported by the recourse/reason-codes (`12`) and the logged appeal path.

## 4. Necessity & proportionality (Art. 5, 6)
- **Minimisation (5(1)(c)):** the model uses **sex (via personal-status) and age** — the deployer
  should test whether excluding them materially harms accuracy; if not, their use is disproportionate.
  **A feature-ablation study (not yet performed) is required to establish this** — the recourse
  analysis (`12`) is about loan-term actionability and does **not** by itself prove minimisation.
- **Data accuracy (5(1)(d))** concerns the accuracy of the *personal data itself* (not model
  performance): the deployer must keep inputs accurate/up-to-date. Separately, model **predictive**
  accuracy (AUROC 0.769, Brier 0.178) and the **13% near-threshold** instability (`12`) bear on AI Act
  **Art. 15**, not GDPR 5(1)(d).
- **Purpose limitation / storage:** illustrative; to be set by the deployer.

## 5. Risks to the rights and freedoms of data subjects
| Risk | Source finding | Severity × likelihood | Note |
|---|---|---|---|
| **Discrimination / bias** (sex, age, foreign-worker) | `11` | Medium × Medium | Disparities directionally disadvantage worse-off groups but **none significant at n=300**; model **trains on a sex proxy + age**. |
| **Proxy protected-attribute inference (indirect discrimination)** | `11`, §2 | Medium × Medium | Sex inferred from marital status (personal-status) — a **non-Art.9** protected-attribute proxy → indirect-discrimination risk. |
| **Opacity / no meaningful explanation** | `10`, `12` | Medium × Low–Med | Explanations are **metric-dependently faithful** (TreeSHAP the more consistent); **~13% of declines have no actionable reduction-recourse** and need a genuine reason. |
| **Inaccuracy / instability at the margin** | `12` | Medium × Medium | 13% near-threshold; ~4–7% decision flips under small input noise. |
| **Loss of autonomy / automation over-reliance** | `00 §2` | High × Low | Mitigated only if human oversight is *effective* (untested). |

## 6. Measures to address the risks
Human-in-the-loop review (`00 §2`); **reason codes / recourse** surfaced to applicants (`12`);
ongoing **faithfulness, fairness, robustness monitoring** (this audit is the baseline); a logged
**appeal / contest** path; a recommendation to **re-evaluate use of the sex proxy and age** and to
apply **bias mitigation** if disparities persist at larger n (`11`). Pre-registration + external
model-reviewer gauntlets guard the assurance method's integrity.

## 7. Residual risk & prior consultation (Art. 36)
Residual risk after measures is assessed **medium** (bias + proxy-inference + margin-instability are
not eliminated, only monitored). If a real deployer cannot reduce it, **Art. 36 prior consultation
with the supervisory authority** may be required. **No opinion** is expressed that residual risk is
acceptable — that is the controller's decision.

## 8. Limitations
Template on public research data (no real data subjects, no real controller). **Conditional Art. 35
procedural elements NOT performed:** Art. **35(2)** DPO advice — required **where a DPO is designated**
(or otherwise obtained where the controller has one; a real deployer must first determine designation
under Art. 37); Art. **35(9)** data-subject views — to be sought **where appropriate**, subject to the
protection of commercial/public interests and security; and the controller identity / retention /
recipients / **Chapter V international-transfer** assessment (all deployer-specific). Legal bases and retention are illustrative; human-oversight effectiveness and the
Art. 22 "meaningfulness" question are **not empirically tested**; single dataset/model; self-assessment,
not a lawyer-reviewed DPIA.
