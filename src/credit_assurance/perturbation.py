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
from sklearn.neighbors import NearestNeighbors      # kNN index (donor search) — standard utility
from sklearn.preprocessing import StandardScaler    # per-column standardisation — standard utility

# Logical feature name -> the tuple of encoded column indices that make up that group.
FeatureGroups = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class PerturbResult:
    """Result of one group-erasure call.

    Attributes:
        X: The (m, d) matrix of donor-perturbed rows.
        ood_distance: Mean standardised nearest-neighbour distance of ``X`` to the donor pool.
    """
    X: np.ndarray          # (m, d) donor draws
    ood_distance: float    # mean standardised nearest-neighbour distance of X to the donor pool


# perturb(x, group_ids, m) -> PerturbResult; the erasure operator returned by make_perturber.
Perturb = Callable[[np.ndarray, Sequence[str], int], PerturbResult]


def _validate_groups(feature_groups: FeatureGroups, d: int) -> None:
    """Assert the feature-group map is a full, disjoint partition of the ``d`` columns.

    Args:
        feature_groups: Logical name -> column-index tuple.
        d: Total number of encoded columns.

    Raises:
        ValueError: If any group is empty, an index is out of ``[0, d)``, a column is shared by two
            groups, or the groups do not cover all ``d`` columns.
    """
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
    """Flatten the requested logical groups to their sorted, de-duplicated encoded column indices.

    Args:
        group_ids: Logical group names to erase.
        feature_groups: Logical name -> column-index tuple.

    Returns:
        Sorted unique column indices spanned by ``group_ids``.
    """
    cols: list[int] = []
    for g in group_ids:
        cols.extend(feature_groups[g])
    return sorted(set(cols))


def _call_rng(seed: int, group_ids: Sequence[str], x: np.ndarray) -> np.random.Generator:
    r"""Deterministic per-call RNG from ``(seed, group_ids, x)`` — order-independent and reproducible.

    LaTeX: \text{state} = \text{seed} \oplus \big(\text{SHA256}(x_{\text{LE}} \,\|\, \text{sorted}(G))_{[0:8]}\big),
    i.e. the 64-bit low word of a content hash of the instance bytes and the sorted group names is
    XOR-mixed into the seed. Sorting the group names makes the draw invariant to argument order.

    Args:
        seed: Base integer seed bound by the factory.
        group_ids: Logical groups being erased on this call.
        x: The instance, shape (d,).

    Returns:
        A NumPy ``Generator`` seeded reproducibly for this exact call.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())  # fixed little-endian for portability
    h.update(repr(sorted(group_ids)).encode())
    mix = int.from_bytes(h.digest()[:8], "little")
    return np.random.default_rng((int(seed) ^ mix) & 0xFFFFFFFFFFFFFFFF)


def make_perturber(donor_pool: np.ndarray, feature_groups: FeatureGroups,
                   regime: str = "conditional", seed: int = 0, k_neighbors: int = 50) -> Perturb:
    """Build a frozen group-erasure operator bound to a held-out donor pool and a regime.

    Args:
        donor_pool: Held-out rows (training split), shape (n_donor, d), used as replacement values.
        feature_groups: Logical name -> column-index tuple (must partition all d columns).
        regime: One of ``"baseline"``, ``"marginal"``, ``"conditional"`` (see module docstring).
        seed: Base seed mixed per call by ``_call_rng``.
        k_neighbors: Neighbourhood size for the conditional regime's kNN donor draw.

    Returns:
        ``perturb(x, group_ids, m) -> PerturbResult`` erasing whole groups atomically.

    Raises:
        ValueError: If ``regime`` is unknown or ``feature_groups`` is not a full disjoint partition.
    """
    if regime not in {"baseline", "marginal", "conditional"}:
        raise ValueError(f"unknown regime {regime!r}")
    donor_pool = np.asarray(donor_pool, dtype=float)
    d = donor_pool.shape[1]
    _validate_groups(feature_groups, d)
    scaler = StandardScaler().fit(donor_pool)
    donor_scaled = scaler.transform(donor_pool)
    nn_full = NearestNeighbors(n_neighbors=1).fit(donor_scaled)

    def _ood(out: np.ndarray) -> float:
        r"""Mean standardised 1-NN distance of rows ``out`` to the donor pool.

        LaTeX: \text{ood}(out) = \frac{1}{m}\sum_{i} \min_{z \in \text{donor}} \lVert T(out_i) - T(z) \rVert_2,
        with T the donor-fitted standardisation. A residual off-manifold diagnostic, not a guarantee.
        """
        dist, _ = nn_full.kneighbors(scaler.transform(out), n_neighbors=1, return_distance=True)
        return float(np.mean(dist))

    def _baseline_block(cols: list[int]) -> np.ndarray:
        r"""Fixed replacement vector for a group: modal one-hot category, or numeric median.

        LaTeX: one-hot group -> \arg\max_{v} \#\{z : z_{cols}=v\} (modal real category); numeric
        singleton -> \operatorname{median}(donor_{:,col}).
        """
        block = donor_pool[:, cols]
        if len(cols) > 1:  # one-hot group -> the modal category vector (a real, valid category)
            uniq, counts = np.unique(block, axis=0, return_counts=True)
            return uniq[int(np.argmax(counts))]
        return np.array([float(np.median(block[:, 0]))])

    def perturb(x: np.ndarray, group_ids: Sequence[str], m: int) -> PerturbResult:
        """Erase ``group_ids`` from ``x`` under the bound regime, returning m perturbed rows + OOD.

        Args:
            x: Instance to perturb, shape (d,), must be finite.
            group_ids: Logical groups to erase (deduplicated + sorted internally for invariance).
            m: Number of perturbed rows to produce.

        Returns:
            ``PerturbResult`` with the (m, d) matrix and its mean standardised OOD distance.

        Raises:
            ValueError: If ``m < 1`` or ``x`` contains non-finite values.
        """
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
        else:  # conditional (approximate on-manifold): draw donors among x's kNN in RETAINED columns
            # LaTeX: let R = retained columns; N_k(x) = the k donors minimising
            # || T(x)_R - T(z)_R ||_2 ; draw d_1..d_m ~ Unif(N_k(x)); set out_i[cols] = donor_{d_i}[cols]
            # (ALL erased columns copied from the SAME donor row -> preserves their joint distribution).
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
