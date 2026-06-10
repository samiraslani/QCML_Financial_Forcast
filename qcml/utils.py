"""
qcml/utils.py
-------------
Shared helpers for clustering, evaluation, and label alignment.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def best_kmeans(X: np.ndarray, k: int, n_init: int = 30) -> np.ndarray:
    """Run K-Means with multiple restarts and return the best label assignment."""
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=500, random_state=0)
    return km.fit_predict(X)


def align_labels(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """
    Permute predicted cluster indices to maximise agreement with y_true.

    Uses the Hungarian algorithm on the contingency matrix.
    Necessary because K-Means cluster numbering is arbitrary.
    """
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            cost[i, j] = int(np.sum((y_pred == i) & (y_true == j)))
    _, col = linear_sum_assignment(-cost)
    return np.array([col[label] for label in y_pred])


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Best-permutation accuracy after Hungarian alignment."""
    return float(np.mean(align_labels(y_true, y_pred, k) == y_true))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def cluster_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_emb: np.ndarray,
) -> dict:
    """
    Return ARI, NMI, and Silhouette score as a dictionary.

    Parameters
    ----------
    y_true : ground-truth integer labels
    y_pred : predicted cluster labels
    X_emb  : the embedding used for silhouette (could be raw or QCML)
    """
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Silhouette": silhouette_score(X_emb, y_pred)
        if len(np.unique(y_pred)) > 1
        else 0.0,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Accuracy, true-positive rate, true-negative rate for binary labels."""
    assert set(np.unique(y_true)) <= {0, 1}
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"Accuracy": acc, "TPR": tpr, "TNR": tnr}


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_moons_hd(n: int = 300, noise: float = 0.12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-moons embedded in 5D with noise dimensions."""
    rng = np.random.default_rng(seed)
    t0  = rng.uniform(0, np.pi, n // 2)
    t1  = rng.uniform(0, np.pi, n - n // 2)
    X0  = np.column_stack([np.cos(t0), np.sin(t0)])
    X1  = np.column_stack([1 - np.cos(t1), 0.5 - np.sin(t1)])
    X2d = np.vstack([X0, X1]) + rng.normal(0, noise, (n, 2))
    y   = np.array([0] * (n // 2) + [1] * (n - n // 2))
    return np.hstack([X2d, rng.normal(0, noise, (n, 3))]), y


def make_concentric_shells(n: int = 300, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two concentric hyperspherical shells in 5D (inner radius 1, outer radius 2)."""
    rng = np.random.default_rng(seed)
    d   = 5
    n0, n1 = n // 2, n - n // 2
    dirs0 = rng.standard_normal((n0, d)); dirs0 /= np.linalg.norm(dirs0, axis=1, keepdims=True)
    dirs1 = rng.standard_normal((n1, d)); dirs1 /= np.linalg.norm(dirs1, axis=1, keepdims=True)
    X = np.vstack([dirs0 * (1.0 + rng.normal(0, noise, (n0, 1))),
                   dirs1 * (2.0 + rng.normal(0, noise, (n1, 1)))])
    return X, np.array([0] * n0 + [1] * n1)


def make_xor(n: int = 300, noise: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """XOR pattern in 4D: label = sign(x0 * x1) > 0."""
    rng = np.random.default_rng(seed)
    X   = rng.uniform(-1, 1, (n, 4)) + rng.normal(0, noise, (n, 4))
    y   = (np.sign(X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


def make_spirals(n: int = 300, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two Archimedean spiral arms in 2D embedded in 4D with noise dimensions."""
    rng = np.random.default_rng(seed)
    n0, n1 = n // 2, n - n // 2
    t0  = np.linspace(0, 4 * np.pi, n0)
    t1  = np.linspace(0, 4 * np.pi, n1)
    X0  = np.column_stack([t0 * np.cos(t0), t0 * np.sin(t0)])
    X1  = np.column_stack([t1 * np.cos(t1 + np.pi), t1 * np.sin(t1 + np.pi)])
    X2d = np.vstack([X0, X1]) / (4 * np.pi)
    y   = np.array([0] * n0 + [1] * n1)
    return np.hstack([X2d + rng.normal(0, noise, (n, 2)),
                      rng.normal(0, noise, (n, 2))]), y


def make_gaussian_clusters(
    k: int,
    n_per_class: int,
    d: int,
    separation: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate k balanced Gaussian clusters in R^d.

    Returns
    -------
    X : shape (k * n_per_class, d) — standardised
    y : shape (k * n_per_class,)  — integer labels 0..k-1
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    Xp, yp = [], []
    for c in range(k):
        mean = rng.standard_normal(d)
        mean = mean / np.linalg.norm(mean) * separation * (c + 1)
        L = rng.standard_normal((d, d))
        cov = (L @ L.T) / d + 0.5 * np.eye(d)
        Xp.append(rng.multivariate_normal(mean, cov, size=n_per_class))
        yp.append(np.full(n_per_class, c, dtype=int))

    X_raw = np.vstack(Xp)
    y = np.concatenate(yp)
    perm = rng.permutation(len(y))
    X_raw, y = X_raw[perm], y[perm]
    return StandardScaler().fit_transform(X_raw), y
