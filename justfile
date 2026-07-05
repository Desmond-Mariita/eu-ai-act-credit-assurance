setup:
    uv venv && uv pip install -e ".[dev]"
setup-models:
    uv pip install -e ".[models]"
test:
    uv run ruff check src/ tests/ scripts/ && uv run pytest -q
data:
    uv run python scripts/00_data.py
    uv run python scripts/05_manifest_and_dq.py
    uv run python scripts/06_gmsc_prep.py
train:
    uv run python scripts/10_train.py
# full audit (needs the [models] extra + data/ present); scripts are numbered in run order
audit:
    uv run python scripts/30_faithfulness.py
    uv run python scripts/30_faithfulness.py --dataset gmsc
    uv run python scripts/40_fairness.py
    uv run python scripts/50_robustness.py
    uv run python scripts/60_reason_codes.py
    uv run python scripts/70_roar.py
