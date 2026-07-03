"""scripts/05_manifest_and_dq.py — Phase 0.5 data manifest, verifier, data-quality + error
profile, and feature-group derivation. Pure pandas + hashlib + json; no network, no model.

Artifacts (repo root):
  data_manifest.sha256   per-file SHA256 (`sha256sum -c`-compatible, repo-relative paths)
  data_quality.json      Art. 10(3) completeness + ERROR/anomaly profile
  feature_groups.json    logical feature -> {columns:[names], indices:[int]} (one-hot grouped)

Feature-group naming contract: one-hot columns are named `<logical>_<value>` (e.g. `Attribute1_A11`);
any column without `_` is a numeric singleton. Grouping validates full coverage and no overlap and
RAISES on violation (not `assert`, so it holds under `python -O`).

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
# The data snapshot under audit, as repo-relative paths (sha256sum-compatible).
DATA_FILES = ["data/german_credit.parquet", "data/cs-training.csv"]
MANIFEST = ROOT / "data_manifest.sha256"
DQ_JSON = ROOT / "data_quality.json"
FG_JSON = ROOT / "feature_groups.json"

SOURCES = {
    "german_credit": "UCI Statlog German Credit (id 144) — https://archive.ics.uci.edu/dataset/144 "
                     "— CC BY 4.0 — cleaned/encoded parquet",
    "gmsc": "Kaggle 'GiveMeSomeCredit' cs-training.csv — https://www.kaggle.com/c/GiveMeSomeCredit "
            "— per Kaggle competition terms — raw",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    missing = [f for f in DATA_FILES if not (ROOT / f).exists()]
    if missing:
        print(f"ERROR: acquire data first — missing: {missing}\n"
              f"  (Give Me Some Credit needs a manual Kaggle download of cs-training.csv into data/.)")
        sys.exit(2)
    lines = [f"{sha256(ROOT / f)}  {f}" for f in DATA_FILES]
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
        path = ROOT / name
        if not path.exists():
            print(f"MISSING {name}")
            ok = False
            continue
        got = sha256(path)
        print(f"{'OK  ' if got == want else 'FAIL'} {name}")
        ok &= got == want
    print("manifest verify:", "OK" if ok else "MISMATCH/MISSING")
    return ok


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


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
            nn = s.dropna()
            # guard all-null: keep JSON standard (null, not NaN)
            rec["min"] = float(nn.min()) if not nn.empty else None
            rec["max"] = float(nn.max()) if not nn.empty else None
            rec["p99"] = round(float(nn.quantile(0.99)), 4) if not nn.empty else None
        out[col] = rec
    return out


def _anomalies(df: pd.DataFrame) -> list[str]:
    """Art. 10(3) 'errors' — flag domain-implausible values, sentinels, and extreme outliers."""
    flags = []
    for c in _numeric_cols(df):
        s = df[c].dropna()
        if s.empty:
            continue
        neg = int((s < 0).sum())
        if neg:
            flags.append(f"{c}: {neg} negative value(s)")
        if "age" in c.lower():
            bad = int(((s < 18) | (s > 110)).sum())
            if bad:
                flags.append(f"{c}: {bad} implausible age(s) <18 or >110 (min={s.min():.0f})")
        if any(k in c for k in ("PastDue", "DaysLate", "30-59", "60-89", "90Days")):
            sent = int(s.isin([96, 98]).sum())
            if sent:
                flags.append(f"{c}: {sent} sentinel value(s) 96/98 (likely data-entry codes)")
        p99, mx = float(s.quantile(0.99)), float(s.max())
        if p99 > 0 and mx > 100 * p99:
            flags.append(f"{c}: max {mx:.0f} >> p99 {p99:.0f} — probable outlier/sentinel")
    return flags


def _profile(name: str, df: pd.DataFrame, target: str, target_meaning: str) -> dict:
    y = df[target]
    return {
        "source": SOURCES[name],
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1] - 1),
        "target": {"name": target, "meaning": target_meaning,
                   "classes": {str(k): int(v) for k, v in y.value_counts().sort_index().items()}},
        "columns": _column_profile(df.drop(columns=[target])),
        "anomalies": _anomalies(df.drop(columns=[target])),
    }


def data_quality() -> dict:
    profiles = {}
    gc_path, gmsc_path = ROOT / DATA_FILES[0], ROOT / DATA_FILES[1]
    if gc_path.exists():
        profiles["german_credit"] = _profile(
            "german_credit", pd.read_parquet(gc_path), "y", "1 = bad credit")
    else:
        profiles["german_credit"] = {"status": "absent"}
    if gmsc_path.exists():
        profiles["gmsc"] = _profile(
            "gmsc", pd.read_csv(gmsc_path, index_col=0),
            "SeriousDlqin2yrs", "1 = serious delinquency in 2 yrs")
    else:
        profiles["gmsc"] = {"status": "absent — manual Kaggle download required"}
    DQ_JSON.write_text(json.dumps(profiles, indent=2, allow_nan=False))
    print(f"data-quality -> {DQ_JSON.relative_to(ROOT)}")
    for nm, pr in profiles.items():
        for a in pr.get("anomalies", []):
            print(f"  [{nm}] anomaly: {a}")
    return profiles


def feature_groups() -> dict:
    """Group encoded columns into logical features (see the naming contract in the module docstring).
    Stores both column NAMES and indices so the artifact is self-describing (not parquet-order-dependent)."""
    out = {}
    for name, path, target in [("german_credit", ROOT / DATA_FILES[0], "y"),
                               ("gmsc", ROOT / DATA_FILES[1], "SeriousDlqin2yrs")]:
        if not path.exists():
            out[name] = {"status": "absent"}
            continue
        df = (pd.read_parquet(path) if path.suffix == ".parquet"
              else pd.read_csv(path, index_col=0)).drop(columns=[target])
        groups: dict[str, dict] = {}
        for idx, col in enumerate(df.columns):
            logical = col.split("_", 1)[0] if "_" in col else col
            g = groups.setdefault(logical, {"columns": [], "indices": []})
            g["columns"].append(col)
            g["indices"].append(idx)
        # invariant: full coverage, no overlap — RAISE (holds under python -O)
        covered = sorted(i for g in groups.values() for i in g["indices"])
        if covered != list(range(df.shape[1])):
            raise ValueError(f"{name}: feature-group coverage/overlap invariant violated")
        out[name] = groups
        n_multi = sum(1 for g in groups.values() if len(g["indices"]) > 1)
        print(f"feature-groups[{name}] -> {len(groups)} logical, {n_multi} one-hot groups")
    FG_JSON.write_text(json.dumps(out, indent=2))
    print(f"feature-groups -> {FG_JSON.relative_to(ROOT)}")
    return out


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        sys.exit(0 if verify_manifest() else 1)
    write_manifest()
    data_quality()
    feature_groups()
    print("Phase 0.5 data artifacts written.")


if __name__ == "__main__":
    main()
