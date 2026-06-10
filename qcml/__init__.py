"""
QCML — Quantum Cognition Machine Learning
==========================================
A quantum-inspired unsupervised representation learning framework.

Quick start
-----------
>>> from qcml.core import QCML
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np

>>> X = StandardScaler().fit_transform(your_data)  # (N, D) float array
>>> model = QCML(hilbert_dim=16, lr=3e-3, w=1.0, seed=42)
>>> model.fit(X, epochs=300, batch_size=32)
>>> Z = model.transform(X)   # embeddings on the unit sphere, shape (N, m)
"""

from .core import QCML, build_hamiltonian, ground_state, hf_gradient
from .utils import (
    best_kmeans,
    align_labels,
    cluster_accuracy,
    cluster_metrics,
    classification_metrics,
    make_gaussian_clusters,
    make_moons_hd,
    make_concentric_shells,
    make_xor,
    make_spirals,
)

__all__ = [
    "QCML",
    "build_hamiltonian",
    "ground_state",
    "hf_gradient",
    "best_kmeans",
    "align_labels",
    "cluster_accuracy",
    "cluster_metrics",
    "classification_metrics",
    "make_gaussian_clusters",
    "make_moons_hd",
    "make_concentric_shells",
    "make_xor",
    "make_spirals",
]
