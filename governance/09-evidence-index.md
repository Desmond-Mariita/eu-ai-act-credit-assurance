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

*Data artifacts are hashed now (prefixes above; full values recomputable via `sha256sum`); living
governance documents are content-addressed at the v1 release freeze (Task 26).*

## Pre-registration proof (filled at plan Task 4, verified at Task 26)
| Item | Value |
|------|-------|
| `HYPOTHESES.md` SHA256 | *(Task 4)* |
| `prereg-v1` tag SHA | *(Task 4)* |
| Commit SHA (pre-registration) | *(Task 4)* |
| Public remote URL | *(Task 4 — requires the repo to be published)* |
| OpenTimestamps `.ots` verify output | *(Task 26 — asynchronous, after Bitcoin confirmation)* |
