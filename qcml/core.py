"""
qcml/core.py
------------
Quantum Cognition Machine Learning (QCML) — canonical implementation.

Reference
---------
"Quantum Cognition Machine Learning: Concepts, Foundations, and Notation" (2026).
Musaelian et al., Qognitive Inc. / Fields Institute.

Algorithm summary
-----------------
Each data point x in R^D is encoded as the ground state |psi_0(x)> of a
Hamiltonian built from D learned Hermitian observables {A_1, ..., A_D}:

    H(x) = sum_k (A_k - x_k * I)^2

The ground state minimises the loss:

    L(x) = lambda_0(H(x))
          = (1/2) * ||<A>_psi - x||^2  +  (1/2) * sum_k Var_psi(A_k)

Gradients are computed via the Hellmann-Feynman theorem:

    d(lambda_0)/d(A_k) = psi_0 (B_k psi_0)^T + (B_k psi_0) psi_0^T
    where B_k = A_k - x_k * I

Observables are updated with Adam and re-symmetrised after each step.
The output embedding Z = {|psi_0(x_t)>} lives on the unit sphere in R^m.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level quantum mechanics helpers
# ---------------------------------------------------------------------------

def build_hamiltonian(x: np.ndarray, observables: List[np.ndarray]) -> np.ndarray:
    """H(x) = sum_k (A_k - x_k I)^2  [m x m symmetric matrix]."""
    m = observables[0].shape[0]
    I = np.eye(m)
    H = np.zeros((m, m))
    for k, A in enumerate(observables):
        B = A - x[k] * I
        H += B @ B
    return H


def ground_state(
    x: np.ndarray,
    observables: List[np.ndarray],
) -> Tuple[float, np.ndarray]:
    """
    Return (lambda_0, |psi_0>) — smallest eigenvalue and eigenvector of H(x).

    Sign convention: the component with the largest absolute value is made
    positive, removing the global +/-1 phase ambiguity before downstream
    clustering.
    """
    vals, vecs = eigh(build_hamiltonian(x, observables))
    psi0 = vecs[:, 0].copy()
    psi0 *= np.sign(psi0[np.argmax(np.abs(psi0))])
    return float(vals[0]), psi0


def hf_gradient(
    x: np.ndarray,
    psi0: np.ndarray,
    k: int,
    observables: List[np.ndarray],
) -> np.ndarray:
    """
    Matrix gradient of lambda_0(H(x)) w.r.t. A_k (Hellmann-Feynman theorem).

    G_k = outer(psi0, B_k psi0) + outer(B_k psi0, psi0)
    where B_k = A_k - x_k I.

    G_k is symmetric, so Adam updates preserve Hermitian structure.
    """
    B = observables[k] - x[k] * np.eye(observables[k].shape[0])
    v = B @ psi0
    return np.outer(psi0, v) + np.outer(v, psi0)


def expected_values(
    psi0: np.ndarray,
    observables: List[np.ndarray],
) -> np.ndarray:
    """<psi0 | A_k | psi0> for each k.  Shape (D,)."""
    return np.array([float(psi0 @ A @ psi0) for A in observables])


def quantum_variances(
    psi0: np.ndarray,
    observables: List[np.ndarray],
) -> np.ndarray:
    """Var_psi(A_k) = <A_k^2> - <A_k>^2 for each k.  Shape (D,)."""
    return np.array([
        float(psi0 @ A @ A @ psi0) - float(psi0 @ A @ psi0) ** 2
        for A in observables
    ])


# ---------------------------------------------------------------------------
# QCML class
# ---------------------------------------------------------------------------

class QCML:
    """
    Quantum Cognition Machine Learning unsupervised representation learner.

    Parameters
    ----------
    hilbert_dim : int
        Hilbert-space dimension m. Observables are m x m symmetric matrices.
        Larger m gives richer representations at higher compute cost.
        Rule of thumb: m >= D for near-zero reconstruction error.
    lr : float
        Adam learning rate (default 3e-3 works well across all tested datasets).
    w : float in [0, 1]
        Variance weight. The loss is:
            L_w = lambda_0 - (1 - w) * sum_k Var_psi(A_k)
        w=1.0 : minimise full ground-state energy (default, max compression)
        w=0.0 : minimise bias only, maximise variance (max spread on sphere)
        w=0.25: empirically best for linear separability (see experiments)
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        hilbert_dim: int = 16,
        lr: float = 3e-3,
        w: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {w}")
        self.m = hilbert_dim
        self.lr = lr
        self.w = w
        self.rng = np.random.default_rng(seed)

        self.observables: List[np.ndarray] = []
        self.D: int = 0
        self.loss_history: List[float] = []

        self._adam_m: List[np.ndarray] = []
        self._adam_v: List[np.ndarray] = []
        self._adam_t: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_observables(self, D: int, X: np.ndarray) -> None:
        """
        Initialise observables with eigenvalues spanning each feature's range.
        This gives the gradient a strong starting signal rather than fighting
        a near-zero random initialisation.
        """
        self.D = D
        self.observables, self._adam_m, self._adam_v = [], [], []
        for k in range(D):
            lo, hi = float(X[:, k].min()), float(X[:, k].max())
            eigvals = np.linspace(lo, hi, self.m)
            Q, _ = np.linalg.qr(self.rng.standard_normal((self.m, self.m)))
            A = Q @ np.diag(eigvals) @ Q.T
            noise = self.rng.standard_normal((self.m, self.m)) * 0.01
            A = A + 0.5 * (noise + noise.T)
            self.observables.append(A)
            self._adam_m.append(np.zeros((self.m, self.m)))
            self._adam_v.append(np.zeros((self.m, self.m)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _adam_update(self, grads: List[np.ndarray]) -> None:
        """Apply one Adam step to all observables and re-symmetrise."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam_t += 1
        t = self._adam_t
        for k in range(self.D):
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * g ** 2
            m_hat = self._adam_m[k] / (1 - beta1 ** t)
            v_hat = self._adam_v[k] / (1 - beta2 ** t)
            self.observables[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            A = self.observables[k]
            self.observables[k] = 0.5 * (A + A.T)

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 300,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> "QCML":
        """
        Train observables via mini-batch Adam gradient descent.

        Parameters
        ----------
        X : array of shape (N, D) — standardised input features
        epochs : number of full passes over X
        batch_size : mini-batch size
        verbose : print loss every 50 epochs
        """
        N, D = X.shape
        if not self.observables:
            self._init_observables(D, X)

        for ep in range(epochs):
            perm = self.rng.permutation(N)
            ep_loss = 0.0

            for start in range(0, N, batch_size):
                batch = X[perm[start : start + batch_size]]
                grads = [np.zeros((self.m, self.m)) for _ in range(D)]
                b_loss = 0.0

                for x in batch:
                    lam0, psi0 = ground_state(x, self.observables)
                    if self.w < 1.0:
                        var_tot = float(quantum_variances(psi0, self.observables).sum())
                        b_loss += lam0 - (1.0 - self.w) * var_tot
                        for k in range(D):
                            g = hf_gradient(x, psi0, k, self.observables)
                            A = self.observables[k]
                            a_k = float(psi0 @ A @ psi0)
                            Ap = A @ psi0
                            g -= (1.0 - self.w) * (
                                np.outer(psi0, Ap) + np.outer(Ap, psi0)
                                - 2.0 * a_k * np.outer(psi0, psi0)
                            )
                            grads[k] += g
                    else:
                        b_loss += lam0
                        for k in range(D):
                            grads[k] += hf_gradient(x, psi0, k, self.observables)

                n = len(batch)
                self._adam_update([g / n for g in grads])
                ep_loss += b_loss

            self.loss_history.append(ep_loss)
            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1:4d}/{epochs}   loss = {ep_loss:12.4f}")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map X to ground-state embeddings on the unit sphere.

        Returns
        -------
        Z : array of shape (N, m)
        """
        return np.stack([ground_state(x, self.observables)[1] for x in X])

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Per-sample reconstruction MSE: (1/D) sum_k (<A_k> - x_k)^2.
        Shape (N,).
        """
        errs = []
        for x in X:
            _, psi0 = ground_state(x, self.observables)
            exp = expected_values(psi0, self.observables)
            errs.append(float(np.mean((exp - x) ** 2)))
        return np.array(errs)

    def expected_feature_values(self, X: np.ndarray) -> np.ndarray:
        """
        Predicted feature values <psi0 | A_k | psi0> for each sample.
        Shape (N, D).
        """
        rows = []
        for x in X:
            _, psi0 = ground_state(x, self.observables)
            rows.append(expected_values(psi0, self.observables))
        return np.array(rows)

    # ------------------------------------------------------------------
    # Feature diagnostics
    # ------------------------------------------------------------------

    def observable_eigenvalue_spread(self) -> np.ndarray:
        """
        Std of eigenvalues of each observable A_k.  Shape (D,).

        High spread means A_k has learned a wide range of measurement outcomes
        -> the feature is being used actively by the model.
        Low spread means A_k is nearly scalar -> the feature was effectively
        ignored (informative for identifying noise features).
        """
        return np.array([
            float(np.std(np.linalg.eigvalsh(A)))
            for A in self.observables
        ])

    def per_feature_quantum_variance(
        self,
        X: np.ndarray,
        n_samples: int = 200,
    ) -> np.ndarray:
        """
        Mean Var_psi(A_k) = <A_k^2> - <A_k>^2 per feature.  Shape (D,).

        Averaged over a random subsample of X for efficiency.
        """
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        var_sum = np.zeros(self.D)
        for x in X[idx]:
            _, psi0 = ground_state(x, self.observables)
            var_sum += quantum_variances(psi0, self.observables)
        return var_sum / len(idx)

    def per_feature_recon_mse(self, X: np.ndarray) -> np.ndarray:
        """Mean (<A_k> - x_k)^2 per feature k.  Shape (D,)."""
        X_hat = self.expected_feature_values(X)
        return np.mean((X_hat - X) ** 2, axis=0)

    def spectral_gaps(self, X: np.ndarray, n_samples: int = 120) -> np.ndarray:
        """
        lambda_1(H(x)) - lambda_0(H(x)) for a random subset of X.

        A large spectral gap indicates the ground state is well-separated from
        the first excited state, i.e., the embedding is robust to perturbations.
        """
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        gaps = []
        for x in X[idx]:
            vals = eigh(
                build_hamiltonian(x, self.observables),
                eigvals_only=True,
                subset_by_index=[0, 1],
            )
            gaps.append(float(vals[1] - vals[0]))
        return np.array(gaps)
