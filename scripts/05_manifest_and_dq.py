"""scripts/05_manifest_and_dq.py — Phase 0.5 data manifest, verifier, data-quality profile,
and feature-group derivation. Pure pandas + hashlib + json; no network, no model.

Artifacts (repo root):
  data_manifest.sha256   per-file SHA256 of the data snapshot (sha256sum format)
  data_quality.json      Art. 10(3) completeness/error profile (missingness/bounds/uniques)
  feature_groups.json    logical feature -> encoded column indices (one-hot dummies grouped)

Usage:
  python scripts/05_manifest_and_dq.py          # (re)generate all artifacts
  python scripts/05_manifest_and_dq.py verify    # verify data files against the manifest
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# The data snapshot under audit (raw GMSC csv + the cleaned German-Credit parquet).
DATA_FILES = ["german_credit.parquet", "cs-training.csv"]
MANIFEST = ROOT / "data_manifest.sha256"
DQ_JSON = ROOT / "data_quality.json"
FG_JSON = ROOT / "feature_groups.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    lines = [f"{sha256(DATA / f)}  {f}" for f in DATA_FILES]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"manifest -> {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> bool:
    if not MANIFEST.exists():
        print("FAIL: no manifest; run without 'verify' first")
        return False
    ok = True
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split("  ", 1)
        got = sha256(DATA / name)
        status = "OK  " if got == want else "FAIL"
        ok &= got == want
        print(f"{status} {name}")
    print("manifest verify:", "OK" if ok else "MISMATCH")
    return ok


def _column_profile(df: pd.DataFrame) -> dict:
    out = {}
    for col in df.columns:
        s = df[col]
        rec = {
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_frac": round(float(s.isna().mean()), 6),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            rec["min"] = float(s.min())
            rec["max"] = float(s.max())
        out[col] = rec
    return out


def data_quality() -> dict:
    profiles = {}

    # German Credit (cleaned, encoded parquet; target column 'y')
    gc = pd.read_parquet(DATA / "german_credit.parquet")
    y = gc["y"]
    profiles["german_credit"] = {
        "source": "UCI Statlog German Credit (id 144), CC BY 4.0 — cleaned/encoded parquet",
        "n_rows": int(gc.shape[0]),
        "n_cols": int(gc.shape[1] - 1),
        "target": {"name": "y", "meaning": "1 = bad credit",
                   "classes": {str(k): int(v) for k, v in y.value_counts().sort_index().items()}},
        "columns": _column_profile(gc.drop(columns=["y"])),
    }

    # Give Me Some Credit (raw csv; target SeriousDlqin2yrs) — has real missingness
    gmsc = pd.read_csv(DATA / "cs-training.csv", index_col=0)
    yg = gmsc["SeriousDlqin2yrs"]
    profiles["gmsc"] = {
        "source": "Kaggle 'GiveMeSomeCredit' cs-training.csv (per Kaggle terms) — raw",
        "n_rows": int(gmsc.shape[0]),
        "n_cols": int(gmsc.shape[1] - 1),
        "target": {"name": "SeriousDlqin2yrs", "meaning": "1 = serious delinquency in 2 yrs",
                   "classes": {str(k): int(v) for k, v in yg.value_counts().sort_index().items()}},
        "columns": _column_profile(gmsc.drop(columns=["SeriousDlqin2yrs"])),
    }
    DQ_JSON.write_text(json.dumps(profiles, indent=2))
    print(f"data-quality -> {DQ_JSON.relative_to(ROOT)}")
    # Surface any completeness issues (Art. 10(3))
    for name, prof in profiles.items():
        miss = {c: r["missing"] for c, r in prof["columns"].items() if r["missing"] > 0}
        if miss:
            print(f"  {name}: columns with missing values -> {miss}")
    return profiles


def feature_groups() -> dict:
    """Group encoded columns into logical features: one-hot dummies `AttributeN_value` collapse
    to `AttributeN`; numeric columns are their own group. Excludes the target 'y'."""
    gc = pd.read_parquet(DATA / "german_credit.parquet").drop(columns=["y"])
    groups: dict[str, list[int]] = {}
    for idx, col in enumerate(gc.columns):
        logical = col.split("_", 1)[0] if "_" in col else col
        groups.setdefault(logical, []).append(idx)
    # invariant checks: full coverage, no overlap
    covered = [i for cols in groups.values() for i in cols]
    assert sorted(covered) == list(range(gc.shape[1])), "feature-group coverage/overlap invariant violated"
    FG_JSON.write_text(json.dumps({"german_credit": groups}, indent=2))
    n_multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"feature-groups -> {FG_JSON.relative_to(ROOT)} "
          f"({len(groups)} logical features, {n_multi} one-hot groups)")
    return groups


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        sys.exit(0 if verify_manifest() else 1)
    write_manifest()
    data_quality()
    feature_groups()
    print("Phase 0.5 data artifacts written.")


if __name__ == "__main__":
    main()
