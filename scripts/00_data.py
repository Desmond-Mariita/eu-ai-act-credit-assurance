"""scripts/00_data.py — fetch both datasets, write cleaned parquet to data/, print shapes.

For GMSC: handle missing CSV gracefully (Kaggle login required; user must place the file).
"""
from pathlib import Path
import pandas as pd
from credit_assurance.data import load_german_credit, load_gmsc

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

# --- Give Me Some Credit (generalization check; CSV is Kaggle-gated) ---
try:
    gmsc = load_gmsc()
    gmsc_path = DATA / "gmsc.parquet"
    gmsc_X = gmsc["X"].copy()
    gmsc_X["y"] = gmsc["y"].values
    gmsc_X.to_parquet(gmsc_path, index=False)
    print(f"gmsc: {gmsc['X'].shape}")
    print(f"  written -> {gmsc_path}")
except FileNotFoundError as e:
    print(f"\nGMSC not available (expected): {e}")
