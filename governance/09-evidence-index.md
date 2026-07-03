# 09 — Evidence Index (working papers)

**Status:** Phase 0.5 placeholder — appended to as artifacts are produced. Every figure, metric, and
claim referenced by the audit opinion must resolve to a stable **`EV-###`** ID with a content hash,
so a reviewer can re-perform it (traceability spine; see `08-traceability-matrix.md`).

| EV-ID | Artifact (repo path) | SHA256 (prefix) | Produced by | Referenced in |
|-------|----------------------|-----------------|-------------|---------------|
| EV-001 | `data_manifest.sha256` | `d2d14d26…` | Task 3 | data-dictionary §0; ToE §5 |
| EV-002 | `data_quality.json` | `76b952bd…` | Task 3 | data-dictionary §2; risk R5 |
| EV-003 | `feature_groups.json` | `3fe2ba1b…` | Task 3 | perturbation core (Task 6) |
| EV-004 | `governance/data-dictionary.md` | *at v1 freeze* | Task 3 | conformity pack (Art. 10) |
| EV-005 | `governance/00-scenario.md` (ToE) | *at v1 freeze* | Task 2 | audit opinion; traceability |
| EV-006 | `governance/02-risk-management-file.md` | *at v1 freeze* | Task 2/16 | conformity pack (Art. 9) |
| EV-007 | `models/lightgbm.txt` (audited model) | `1456b07f…` | Task 8 | ToE §5; model card (Art. 11) |
| EV-008 | `metrics/models.json` | *(committed)* | Task 8 | model card; conformity pack (Art. 15 accuracy) |
| EV-009 | `metrics/faithfulness_german_credit.json` | *(committed)* | Task 10-11 | findings §Results; opinion (Art. 86) |
| EV-010 | `governance/10-faithfulness-findings.md` | *at v1 freeze* | Task 10-11 | audit opinion; conformity pack (Art. 86) |

*Data artifacts are hashed now (prefixes above; full values recomputable via `sha256sum`); living
governance documents are content-addressed at the v1 release freeze (Task 26).*

## Pre-registration proof (filled at plan Task 4, verified at Task 26)
| Item | Value |
|------|-------|
| `HYPOTHESES.md` SHA256 | `8c4e2d7f05314680866f2c651db53129d31b2e2a2ae14ff7dfc59b7a2ec6b357` |
| `prereg-v1` tag SHA (SSH-signed) | `234a6ddecf986b191dda186b58d6e652798fcf85` |
| Commit SHA (pre-registration) | `73fa5f9461c7ed784dab3b9ed41e79de26fc70ab` |
| Public remote URL | https://github.com/Desmond-Mariita/eu-ai-act-credit-assurance (tag: `/releases/tag/prereg-v1`) |
| OpenTimestamps `.ots` verify output | `HYPOTHESES.md.ots` committed; stamped 2026-07-03, submitted to 4 calendars — **`ots upgrade`/`verify` after Bitcoin confirmation (Task 26)** |
