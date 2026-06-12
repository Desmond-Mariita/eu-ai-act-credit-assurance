setup:
    uv venv && uv pip install -e ".[dev]"
test:
    uv run pytest -q
data:
    uv run python scripts/00_data.py
train:
    uv run python scripts/10_train.py
audit:
    uv run python scripts/20_fairness.py && uv run python scripts/30_faithfulness.py && uv run python scripts/31_roar.py && uv run python scripts/50_robustness.py
figures:
    uv run python scripts/60_figures.py
notebooks:
    for nb in notebooks/*.py; do uv run jupytext --to ipynb "$nb" && uv run jupyter nbconvert --to html --execute "${nb%.py}.ipynb"; done
