# 09 — Evidence Index (working papers)

**Status:** living evidence index — appended as artifacts are produced; governance docs
content-addressed by git at tag `v1.1.1` (see note below). Every figure, metric, and
claim referenced by the audit opinion must resolve to a stable **`EV-###`** ID with a content hash,
so a reviewer can re-perform it (traceability spine: `08-traceability-matrix.md`).

| EV-ID | Artifact (repo path) | SHA256 (prefix) | Produced by | Referenced in |
|-------|----------------------|-----------------|-------------|---------------|
| EV-001 | `data_manifest.sha256` (now covers `data/gmsc.parquet`) | `7b5f4568…` | Task 3 | data-dictionary §0; ToE §5 |
| EV-002 | `data_quality.json` | `76b952bd…` | Task 3 | data-dictionary §2; risk R5 |
| EV-003 | `feature_groups.json` | `3fe2ba1b…` | Task 3 | perturbation core (Task 6) |
| EV-004 | `governance/data-dictionary.md` | *at v1 freeze* | Task 3 | conformity pack (Art. 10) |
| EV-005 | `governance/00-scenario.md` (ToE) | *at v1 freeze* | Task 2 | audit opinion; traceability |
| EV-006 | `governance/02-risk-management-file.md` | *at v1 freeze* | Task 2/16 | conformity pack (Art. 9) |
| EV-007 | `models/lightgbm.txt` (audited model) | `1456b07f…` | Task 8 | ToE §5; model card (Art. 11) |
| EV-008 | `metrics/models.json` | `8ef65b5d…` | Task 8 | model card; conformity pack (Art. 15 accuracy) |
| EV-009 | `metrics/faithfulness_german_credit.json` | `ce746733…` | Task 10-11 | findings §Results; opinion (Art. 86) |
| EV-010 | `governance/10-faithfulness-findings.md` | *at v1 freeze* | Task 10-11 | audit opinion; conformity pack (Art. 86) |
| EV-011 | `metrics/faithfulness_gmsc.json` (+ `data/gmsc.parquet` `be036f0f…` via `scripts/06_gmsc_prep.py`; + pinned GMSC model `models/lightgbm_gmsc.txt` `1c1f70ea…`, deterministic, verified in-script) | `a6304b1a…` | Task 11 | findings §Generalization (incl. GMSC model AUROC) |
| EV-012 | `metrics/fairness_german_credit.json` (+ `governance/11-fairness-findings.md`) | `954d5bb6…` | Task 12 | conformity pack (Art. 10(2)(f-g), Art. 15) |
| EV-013 | `metrics/robustness_…json` + `metrics/reason_codes_…json` (+ `governance/12`) | `56fd9e31…` / `bacc5500…` | Task 13 | conformity pack (Art. 15; GDPR 13-15) |
| EV-014 | `governance/13-conformity-dossier.md` | *at v1 freeze* | Task 16 | Annex IV technical documentation; audit opinion |
| EV-015 | `metrics/roar_german_credit.json` (ROAR anchor) | `ea08b093…` | Task 11 supplement | `10-faithfulness-findings.md §ROAR` |
| EV-016 | `governance/06-gdpr-dpia.md` (template DPIA) | *at v1 freeze* | Task 13 supplement | dossier §B (GDPR 35); audit opinion |
| EV-017 | `governance/08-traceability-matrix.md` | *at v1 freeze* | Task 16 | traceability spine; audit opinion |
| EV-018 | `governance/15-external-review.md` (review template; awaiting reviewer) | *on completion* | Task 24 | independence sign-off (open) |

*Data + metrics artifacts carry SHA256 prefixes above (recompute via `sha256sum`). Living **governance
documents are content-addressed by git**: their blob SHA is frozen at tag `v1.1.1` — git inherently
content-addresses versioned text, so `git rev-parse v1.1.1:governance/<file>` yields the immutable hash
(no separate hashing needed; "*at v1 freeze*" marks these rows).*

## Pre-registration proof (filled at plan Task 4, verified at Task 26)
| Item | Value |
|------|-------|
| `HYPOTHESES.md` SHA256 | `8c4e2d7f05314680866f2c651db53129d31b2e2a2ae14ff7dfc59b7a2ec6b357` |
| `prereg-v1` tag SHA (SSH-signed) | `234a6ddecf986b191dda186b58d6e652798fcf85` |
| Commit SHA (pre-registration) | `73fa5f9461c7ed784dab3b9ed41e79de26fc70ab` |
| Public remote URL | https://github.com/Desmond-Mariita/eu-ai-act-credit-assurance (tag: `/releases/tag/prereg-v1`) |
| OpenTimestamps `.ots` verify output | `HYPOTHESES.md.ots` — **Bitcoin-confirmed** via `ots upgrade` (2026-07-05): `BitcoinBlockHeaderAttestation(956533)` (+ 956560) |
