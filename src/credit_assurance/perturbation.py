# src/credit_assurance/perturbation.py
"""Feature-GROUP erasure regimes (the instrument-freeze contract).

A factory `make_perturber(donor_pool, feature_groups, regime, seed)` binds the training-split
donor pool + the feature-group map + the regime + the RNG seed, then returns a pure
`perturb(x, group_ids, m) -> PerturbResult`.

Whole feature-groups are erased atomically — a categorical's one-hot dummy columns are
erased/donated together, never split — so no invalid one-hot rows are ever fabricated. All three
regimes copy real donor column-blocks (or a real modal category), preserving encoding validity;
they differ only in HOW donors are chosen:

  baseline    : one fixed value per group — numeric median / the modal one-hot category
                (deterministic; the naive off-manifold default we critique).
  marginal    : each erased group sampled INDEPENDENTLY from donor rows (breaks the joint
                between groups; off-manifold contrast).
  conditional : on-manifold — all erased groups copied from the SAME donor drawn among x's
                nearest neighbours in the RETAINED columns (keeps the joint plausible). PRIMARY.

`ood_distance` (mean nearest-neighbour distance of the perturbed rows to the donor pool) is
returned so a reader can see whether a measured prediction drop reflects importance or
off-manifold confusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

FeatureGroups = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class PerturbResult:
    X: np.ndarray          # (m, d) donor draws
    ood_distance: float    # mean nearest-neighbour distance of X to the donor pool


Perturb = Callable[[np.ndarray, Sequence[str], int], PerturbResult]


def _erased_cols(group_ids: Sequence[str], feature_groups: FeatureGroups) -> list[int]:
    cols: list[int] = []
    for g in group_ids:
        cols.extend(feature_groups[g])
    return sorted(set(cols))


def make_perturber(donor_pool: np.ndarray, feature_groups: FeatureGroups,
                   regime: str = "conditional", seed: int = 0, k_neighbors: int = 50) -> Perturb:
    if regime not in {"baseline", "marginal", "conditional"}:
        raise ValueError(f"unknown regime {regime!r}")
    donor_pool = np.asarray(donor_pool, dtype=float)
    rng = np.random.default_rng(seed)
    nn_full = NearestNeighbors(n_neighbors=1).fit(donor_pool)

    def _baseline_block(cols: list[int]) -> np.ndarray:
        block = donor_pool[:, cols]
        if len(cols) > 1:  # one-hot group -> the modal category vector (a real, valid category)
            uniq, counts = np.unique(block, axis=0, return_counts=True)
            return uniq[int(np.argmax(counts))]
        return np.array([float(np.median(block[:, 0]))])

    def perturb(x: np.ndarray, group_ids: Sequence[str], m: int) -> PerturbResult:
        x = np.asarray(x, dtype=float)
        cols = _erased_cols(group_ids, feature_groups)
        if not cols:  # nothing to erase -> the point itself (on-manifold by definition)
            return PerturbResult(np.repeat(x[None, :], max(m, 1), axis=0), 0.0)
        out = np.repeat(x[None, :], m, axis=0)

        if regime == "baseline":
            for g in group_ids:
                gc = list(feature_groups[g])
                out[:, gc] = _baseline_block(gc)
        elif regime == "marginal":
            for g in group_ids:
                gc = list(feature_groups[g])
                donors = rng.integers(0, len(donor_pool), size=m)
                out[:, gc] = donor_pool[np.ix_(donors, gc)]
        else:  # conditional (on-manifold)
            keep = [j for j in range(donor_pool.shape[1]) if j not in cols]
            if keep:
                nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(donor_pool))).fit(donor_pool[:, keep])
                nbr = nn.kneighbors(x[keep][None, :], return_distance=False)[0]
            else:
                nbr = np.arange(len(donor_pool))
            donors = rng.choice(nbr, size=m, replace=True)
            out[:, cols] = donor_pool[np.ix_(donors, cols)]  # all erased cols from the SAME donor row

        ood = float(np.mean(nn_full.kneighbors(out, n_neighbors=1, return_distance=True)[0]))
        return PerturbResult(out, ood)

    return perturb
