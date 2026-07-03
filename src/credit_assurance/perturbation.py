# src/credit_assurance/perturbation.py
"""Feature-GROUP erasure regimes (the instrument-freeze contract).

A factory `make_perturber(donor_pool, feature_groups, regime, seed)` binds a **held-out** donor
pool (the training split) + the feature-group map + the regime + a base seed, then returns
`perturb(x, group_ids, m) -> PerturbResult`.

Reproducibility: `perturb` is **order-independent** — its RNG is derived deterministically per call
from `(seed, group_ids, x)`, so identical arguments always yield identical draws regardless of call
order or reuse (safe to cache/parallelise).

Whole feature-groups are erased atomically — a categorical's one-hot dummy columns move together,
never split — so no invalid one-hot row is ever produced (the factory validates the group map is a
full, disjoint partition of the columns). Regimes:

  baseline    : one fixed value per group — numeric median / modal one-hot category (deterministic).
  marginal    : each erased group sampled INDEPENDENTLY from donor rows (breaks the joint).
  conditional : APPROXIMATE on-manifold — all erased groups copied from the SAME donor drawn among
                x's nearest neighbours in the RETAINED columns (kNN conditional imputation). PRIMARY.

Distances (kNN neighbour selection and `ood_distance`) are computed in **standardised** space
(`StandardScaler` on the donor pool) so no column's raw scale dominates. `ood_distance` = mean
standardised nearest-neighbour distance of the perturbed rows to the donor pool; it is a *residual*
off-manifold diagnostic (conditional imputation is approximate, not a guarantee), reported so a
prediction drop can be read as importance vs off-manifold confusion. The donor pool MUST be held out
from the evaluated instances (else self-matches leak).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FeatureGroups = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class PerturbResult:
    X: np.ndarray          # (m, d) donor draws
    ood_distance: float    # mean standardised nearest-neighbour distance of X to the donor pool


Perturb = Callable[[np.ndarray, Sequence[str], int], PerturbResult]


def _validate_groups(feature_groups: FeatureGroups, d: int) -> None:
    seen: set[int] = set()
    for name, cols in feature_groups.items():
        if len(cols) == 0:
            raise ValueError(f"empty feature group {name!r}")
        for c in cols:
            if not (0 <= c < d):
                raise ValueError(f"group {name!r} index {c} out of bounds [0,{d})")
            if c in seen:
                raise ValueError(f"column index {c} appears in more than one feature group")
            seen.add(c)
    if len(seen) != d:
        raise ValueError(f"feature groups cover {len(seen)}/{d} columns — must be a full, disjoint partition")


def _erased_cols(group_ids: Sequence[str], feature_groups: FeatureGroups) -> list[int]:
    cols: list[int] = []
    for g in group_ids:
        cols.extend(feature_groups[g])
    return sorted(set(cols))


def _call_rng(seed: int, group_ids: Sequence[str], x: np.ndarray) -> np.random.Generator:
    """Deterministic per-call RNG from (seed, group_ids, x) — order-independent, reproducible."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())  # fixed little-endian for portability
    h.update(repr(sorted(group_ids)).encode())
    mix = int.from_bytes(h.digest()[:8], "little")
    return np.random.default_rng((int(seed) ^ mix) & 0xFFFFFFFFFFFFFFFF)


def make_perturber(donor_pool: np.ndarray, feature_groups: FeatureGroups,
                   regime: str = "conditional", seed: int = 0, k_neighbors: int = 50) -> Perturb:
    if regime not in {"baseline", "marginal", "conditional"}:
        raise ValueError(f"unknown regime {regime!r}")
    donor_pool = np.asarray(donor_pool, dtype=float)
    d = donor_pool.shape[1]
    _validate_groups(feature_groups, d)
    scaler = StandardScaler().fit(donor_pool)
    donor_scaled = scaler.transform(donor_pool)
    nn_full = NearestNeighbors(n_neighbors=1).fit(donor_scaled)

    def _ood(out: np.ndarray) -> float:
        dist, _ = nn_full.kneighbors(scaler.transform(out), n_neighbors=1, return_distance=True)
        return float(np.mean(dist))

    def _baseline_block(cols: list[int]) -> np.ndarray:
        block = donor_pool[:, cols]
        if len(cols) > 1:  # one-hot group -> the modal category vector (a real, valid category)
            uniq, counts = np.unique(block, axis=0, return_counts=True)
            return uniq[int(np.argmax(counts))]
        return np.array([float(np.median(block[:, 0]))])

    def perturb(x: np.ndarray, group_ids: Sequence[str], m: int) -> PerturbResult:
        if m < 1:
            raise ValueError("m must be >= 1")
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("x must be finite")
        groups = sorted(set(group_ids))          # canonical order -> permutation-invariant output
        cols = _erased_cols(groups, feature_groups)
        out = np.repeat(x[None, :], m, axis=0)
        if not cols:  # unperturbed -> report x's own (real) distance to the pool, not 0
            return PerturbResult(out, _ood(out))

        rng = _call_rng(seed, groups, x)
        if regime == "baseline":
            for g in groups:
                gc = list(feature_groups[g])
                out[:, gc] = _baseline_block(gc)
        elif regime == "marginal":
            for g in groups:
                gc = list(feature_groups[g])
                donors = rng.integers(0, len(donor_pool), size=m)
                out[:, gc] = donor_pool[np.ix_(donors, gc)]
        else:  # conditional (approximate on-manifold)
            keep = [j for j in range(d) if j not in cols]
            if keep:
                nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(donor_pool))).fit(donor_scaled[:, keep])
                x_keep = scaler.transform(x[None, :])[0, keep][None, :]
                nbr = nn.kneighbors(x_keep, return_distance=False)[0]
            else:
                nbr = np.arange(len(donor_pool))
            donors = rng.choice(nbr, size=m, replace=True)
            out[:, cols] = donor_pool[np.ix_(donors, cols)]  # all erased cols from the SAME donor row

        return PerturbResult(out, _ood(out))

    return perturb
