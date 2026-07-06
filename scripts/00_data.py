"""scripts/00_data.py — fetch both datasets, write cleaned parquet to data/, print shapes.

For GMSC: handle missing CSV gracefully (Kaggle login required; user must place the file).
"""
from pathlib import Path
from credit_assurance.data import load_german_credit

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

# --- German Credit (mainline) ---
gc = load_german_credit()
X_gc = gc["X"].copy()
X_gc["y"] = gc["y"].values
out_path = DATA / "german_credit.parquet"
X_gc.to_parquet(out_path, index=False)
print(f"german_credit: {gc['X'].shape}")
print(f"  y classes: {sorted(gc['y'].unique().tolist())}")
print(f"  protected non-null counts:\n{gc['protected'].notnull().sum().to_string()}")
print(f"  X NaN count: {gc['X'].isna().sum().sum()}")
print(f"  written -> {out_path}")

# --- Give Me Some Credit: the audited benchmark parquet is produced by scripts/06_gmsc_prep.py
# (sentinel cleaning + stratified subsample; train-split-only imputation downstream). 00_data does NOT
# write data/gmsc.parquet — that would be a full-data, full-imputed snapshot inconsistent with the
# audited pipeline. Just report availability of the Kaggle CSV. ---
if (DATA / "cs-training.csv").exists():
    print("gmsc: cs-training.csv present -> run scripts/06_gmsc_prep.py to build data/gmsc.parquet")
else:
    print("\nGMSC not available (expected): place cs-training.csv in data/ (Kaggle) + run 06_gmsc_prep.py")
