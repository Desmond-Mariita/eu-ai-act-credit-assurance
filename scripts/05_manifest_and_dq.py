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
  python scripts/05_manifest_and_dq.py           # rebuild DQ + feature groups, then VERIFY (fail-closed)
  python scripts/05_manifest_and_dq.py verify     # verify integrity files against the manifest only
  python scripts/05_manifest_and_dq.py freeze     # MAINTAINER-ONLY: (re)write the manifest (re-baseline)

The manifest is committed; the default/verify paths never mutate it, so a `just data` run authenticates
the audited snapshot instead of silently re-blessing upstream drift (audit R2). Only an explicit
`freeze` re-baselines — do that deliberately, when the audited data snapshot legitimately changes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Raw/derived data sources profiled for data-quality + feature groups.
DATA_FILES = ["data/german_credit.parquet", "data/cs-training.csv"]
# The integrity manifest: EVERY load-bearing data artifact under audit, incl. the DERIVED GMSC snapshot
# (audit R2 — gmsc.parquet is EV-011's data input and must be verifier-covered). Pinned models are
# verified separately by their own sha256 (models.json for German; EV-011 + an in-script allclose guard
# in 30_faithfulness for GMSC), mirroring the existing German-model design.
MANIFEST_FILES = ["data/german_credit.parquet", "data/cs-training.csv", "data/gmsc.parquet"]
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
    """Stream a file through SHA256 in 1 MiB chunks (memory-bounded for large data files).

    Args:
        path: File to hash.

    Returns:
        The hex SHA256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    """MAINTAINER-ONLY re-freeze — the sole path that MUTATES the manifest. Reproduction must VERIFY,
    never re-freeze (regenerating from whatever files are present silently blesses upstream drift —
    audit R2). Absent integrity files are reported, not silently dropped."""
    absent = [f for f in MANIFEST_FILES if not (ROOT / f).exists()]
    if absent:   # REFUSE an incomplete freeze — a partial manifest would later pass a coverage check
        print(f"ERROR: refusing incomplete freeze — {len(absent)} required integrity file(s) absent: "
              f"{absent}\n  (a COMPLETE freeze needs the Kaggle GMSC data + scripts/06_gmsc_prep.py.)")
        sys.exit(2)
    lines = [f"{sha256(ROOT / f)}  {f}" for f in MANIFEST_FILES]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"manifest (re-frozen) -> {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> bool:
    """Fail-closed integrity check: a hash MISMATCH, a listed-but-MISSING file, or a present-but-
    UNTRACKED integrity file all fail. Never mutates the manifest."""
    if not MANIFEST.exists():
        print("FAIL: no manifest; a maintainer must run 'python scripts/05_manifest_and_dq.py freeze'")
        return False
    ok = True
    tracked = set()
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split("  ", 1)
        tracked.add(name)
        path = ROOT / name
        if not path.exists():
            print(f"MISSING {name}")            # fail-closed: a listed file that is gone is a FAIL
            ok = False
            continue
        got = sha256(path)
        print(f"{'OK  ' if got == want else 'FAIL'} {name}")
        ok &= got == want
    for f in MANIFEST_FILES:                     # coverage: EVERY required integrity file must be tracked
        if f not in tracked:                     # (fail-closed even if the file is also absent on disk)
            state = "present on disk" if (ROOT / f).exists() else "absent on disk"
            print(f"UNCOVERED {f} (required integrity file not in manifest; {state}) — re-freeze")
            ok = False
    print("manifest verify:", "OK" if ok else "MISMATCH/MISSING/UNCOVERED")
    return ok


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return the genuinely numeric (non-boolean) column names.

    Args:
        df: The dataframe to inspect.

    Returns:
        Names of numeric, non-bool columns.
    """
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _column_profile(df: pd.DataFrame) -> dict:
    """Per-column dtype / missingness / cardinality (+ numeric min/max/p99) profile.

    Args:
        df: The feature dataframe (target dropped).

    Returns:
        ``{column: {dtype, missing, missing_frac, n_unique, [min, max, p99]}}``.
    """
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
    """Flag Art. 10(3) 'errors': domain-implausible values, sentinel codes, and extreme outliers.

    Checks negatives, implausible ages (<18 or >110), past-due sentinel codes 96/98, and values whose
    max exceeds 100x the 99th percentile (probable sentinels/outliers).

    Args:
        df: The feature dataframe (target dropped).

    Returns:
        Human-readable anomaly strings (empty if none).
    """
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
    """Assemble the source/shape/target/columns/anomalies profile for one dataset.

    Args:
        name: Dataset key (``"german_credit"`` or ``"gmsc"``).
        df: The full dataframe including the target column.
        target: Target column name.
        target_meaning: Human description of the positive class.

    Returns:
        A JSON-serialisable data-quality profile dict.
    """
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
    """Build + write ``data_quality.json`` (Art. 10(3) completeness + error profile) for both datasets.

    Returns:
        The profiles dict (also written to disk); absent datasets are marked ``{"status": "absent"}``.
    """
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
    """CLI entry: ``verify`` (default, fail-closed), ``freeze`` (maintainer re-baseline), or rebuild DQ.

    Exits non-zero on a manifest verification failure so ``just data`` / CI gate on integrity.
    """
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "verify":
        sys.exit(0 if verify_manifest() else 1)
    if arg == "freeze":                       # maintainer-only: (re)write the integrity manifest
        write_manifest()
        data_quality()
        feature_groups()
        print("Phase 0.5 data artifacts (re-)frozen.")
        return
    # default (e.g. `just data`): rebuild DQ + feature groups, then VERIFY fail-closed — never re-freeze.
    data_quality()
    feature_groups()
    if MANIFEST.exists():
        print("(verifying committed manifest — use 'freeze' to intentionally re-baseline)")
        sys.exit(0 if verify_manifest() else 1)
    print("FAIL: no manifest — a maintainer must run 'python scripts/05_manifest_and_dq.py freeze'.")
    sys.exit(1)   # fail-closed: absence of the integrity manifest is not a pass


if __name__ == "__main__":
    main()
