# src/credit_assurance/perturbation.py
"""Three feature-erasure regimes, each a factory returning a perturb(x, idxs)->(m,d) callable.

baseline    : replace with column median (off-manifold; the naive default we critique).
marginal    : replace with values sampled from the column's marginal (off-manifold).
conditional : on-manifold — replace with the erased-feature values of the x's nearest
              neighbours in the *retained* feature subspace (a kNN donor; keeps the joint
              distribution plausible). This is the primary regime.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from sklearn.neighbors import NearestNeighbors

def baseline_perturber(X: np.ndarray):
    med = np.median(X, axis=0)
    def perturb(x, idxs):
        y = x.copy().astype(float); y[list(idxs)] = med[list(idxs)]
        return y[None, :]
    return perturb

def marginal_perturber(X: np.ndarray, m: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    def perturb(x, idxs):
        out = np.repeat(x[None, :].astype(float), m, axis=0)
        for j in idxs:
            out[:, j] = rng.choice(X[:, j], size=m, replace=True)
        return out
    return perturb

def conditional_perturber(X: np.ndarray, m: int = 20, k_neighbors: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    def perturb(x, idxs):
        idxs = list(idxs)
        keep = [j for j in range(X.shape[1]) if j not in idxs]
        nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(X))).fit(X[:, keep])
        nbr = nn.kneighbors(x[keep][None, :], return_distance=False)[0]
        donors = rng.choice(nbr, size=m, replace=True)
        out = np.repeat(x[None, :].astype(float), m, axis=0)
        out[:, idxs] = X[np.ix_(donors, idxs)]
        return out
    return perturb
