"""
qcml/core.py
------------
Quantum Cognition Machine Learning (QCML): canonical implementation.

Reference
---------
"Quantum Cognition Machine Learning: Concepts, Foundations, and Notation" (2026).
Musaelian et al., Qognitive Inc. / Fields Institute.

Algorithm summary
-----------------
Each data point x in R^D is encoded as the ground state |psi_0(x)> of a
Hamiltonian built from D learned Hermitian observables {A_1, ..., A_D}:

    H(x) = sum_k (A_k - x_k * I)^2

The ground state minimizes the loss:

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
    Return (lambda_0, |psi_0>): smallest eigenvalue and eigenvector of H(x).

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
        w=1.0 : minimize full ground-state energy (default, max compression)
        w=0.0 : minimize bias only, maximize variance (max spread on sphere)
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
        Initialize observables with eigenvalues spanning each feature's range.
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
        X : array of shape (N, D): standardized input features
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

    # ------------------------------------------------------------------
    # Quantum similarity and geometry
    # ------------------------------------------------------------------

    def fidelity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        QCML fidelity between two data points: |<psi0(x1)|psi0(x2)>|^2.

        Range [0, 1]. Values near 1 indicate quantum-mechanically similar
        data points; values near 0 indicate distinct quantum encodings.

        Reference: [Q3] Rosaler et al. 2025, Section 2.3
        """
        _, psi1 = ground_state(x1, self.observables)
        _, psi2 = ground_state(x2, self.observables)
        return float((psi1 @ psi2) ** 2)

    def fidelity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Pairwise QCML fidelity matrix F_ij = |<psi0(x_i)|psi0(x_j)>|^2.

        Shape (N, N). Computed efficiently via Z @ Z.T where Z = transform(X).
        Result is symmetric with ones on the diagonal.

        Reference: [Q3] Rosaler et al. 2025
        """
        Z = self.transform(X)   # (N, m), rows are unit vectors
        F = Z @ Z.T             # (N, N), inner products in [-1, 1]
        return F ** 2           # (N, N), fidelities in [0, 1]

    def quantum_metric(self, x: np.ndarray) -> np.ndarray:
        """
        D x D quantum metric tensor g at data point x.

            g_munu(x) = 2 * sum_{n>=1}
                <psi0|A_mu|psi_n> <psi_n|A_nu|psi0> / (E_n - E0)

        Computed efficiently as g = 2 * a @ G @ a.T where:
          - a[mu, :] = A_mu @ psi0       (D x m)
          - G = sum_{n>=1} outer(psi_n, psi_n) / (E_n - E0)  (m x m)

        The eigenvalue spectrum of the mean quantum metric (averaged over X)
        reveals the intrinsic dimension of the data manifold via a spectral
        gap at position d (the true dimension).

        NOTE: this is distinct from spectral_gaps(), which computes
        lambda_1 - lambda_0 of H(x) (a Hamiltonian energy gap, not a
        geometric object).

        Reference: [Q4] Candelori et al. 2025, [Q5] Abanov et al. 2025
        """
        vals, vecs = eigh(build_hamiltonian(x, self.observables))
        psi0 = vecs[:, 0]
        E0 = vals[0]

        # a_mu = A_mu @ psi0 for all mu  (D, m)
        a = np.array([A @ psi0 for A in self.observables])

        # Green's function G = sum_{n>=1} outer(psi_n, psi_n) / (E_n - E0)
        G = np.zeros((self.m, self.m))
        for n in range(1, self.m):
            dE = vals[n] - E0
            if dE < 1e-10:
                continue
            G += np.outer(vecs[:, n], vecs[:, n]) / dE

        return 2.0 * a @ G @ a.T   # (D, D)

    def mean_quantum_metric(
        self,
        X: np.ndarray,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        D x D quantum metric averaged over n_samples data points from X.

        Use the eigenvalues of this matrix to estimate intrinsic dimension:
        look for a spectral gap separating the top-d eigenvalues from the rest.

        Reference: [Q4] Candelori et al. 2025
        """
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
        G = np.zeros((self.D, self.D))
        for i in idx:
            G += self.quantum_metric(X[i])
        return G / len(idx)

    def commutativity(self) -> np.ndarray:
        """
        D x D matrix of Frobenius norms ||[A_j, A_k]||_F.

        Near-zero values indicate the observables nearly commute, which is
        the classical (k-means) limit. Large values indicate non-classical
        quantum correlations between features.

        As w -> 1, commutativity should decay toward zero.
        """
        D = self.D
        C = np.zeros((D, D))
        for j in range(D):
            for k in range(j + 1, D):
                comm = self.observables[j] @ self.observables[k] \
                     - self.observables[k] @ self.observables[j]
                val = float(np.linalg.norm(comm, 'fro'))
                C[j, k] = val
                C[k, j] = val
        return C


# ---------------------------------------------------------------------------
# Supervised QCML extensions
# ---------------------------------------------------------------------------

class QCMLRegressor(QCML):
    """
    QCML for regression via a learnable Hermitian output operator.

    Two-phase training
    ------------------
    Phase 1 (unsupervised):
        Call fit(X): standard Hamiltonian learning.

    Phase 2 (supervised):
        Call fit_output_operator(X, y): freezes observables and learns a
        Hermitian output operator B such that

            y_hat(x) = <psi0(x)|B|psi0(x)>

        minimizes MSE(y_hat, y) via Adam gradient descent.

    The decoupled strategy avoids backpropagating through the eigenvalue
    problem and is stable for all tested datasets.

    Gradient of MSE w.r.t. B (exact via Hellmann-Feynman):
        dL/dB = 2 (y_hat - y) |psi0><psi0|
    """

    def __init__(
        self,
        hilbert_dim: int = 16,
        lr: float = 3e-3,
        w: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(hilbert_dim, lr, w, seed)
        self.B: Optional[np.ndarray] = None
        self._B_adam_m: Optional[np.ndarray] = None
        self._B_adam_v: Optional[np.ndarray] = None
        self._B_adam_t: int = 0

    def _init_B(self, y_range: Tuple[float, float]) -> None:
        lo, hi = y_range
        eigs = np.linspace(lo, hi, self.m)
        Q, _ = np.linalg.qr(self.rng.standard_normal((self.m, self.m)))
        self.B = Q @ np.diag(eigs) @ Q.T
        self._B_adam_m = np.zeros((self.m, self.m))
        self._B_adam_v = np.zeros((self.m, self.m))

    def predict_one(self, x: np.ndarray) -> float:
        _, psi0 = ground_state(x, self.observables)
        return float(psi0 @ self.B @ psi0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression target for each row of X.  Shape (N,)."""
        return np.array([self.predict_one(x) for x in X])

    def fit_output_operator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> "QCMLRegressor":
        """
        Phase 2: learn B with frozen observables.

        Must be called after fit() has already trained the observables.
        """
        y = np.asarray(y, dtype=float)
        if self.B is None:
            self._init_B((float(y.min()), float(y.max())))

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for ep in range(epochs):
            perm = self.rng.permutation(len(X))
            ep_loss = 0.0

            for start in range(0, len(X), batch_size):
                idx = perm[start:start + batch_size]
                batch_X, batch_y = X[idx], y[idx]
                grad_B = np.zeros((self.m, self.m))
                b_loss = 0.0

                for xi, yi in zip(batch_X, batch_y):
                    _, psi0 = ground_state(xi, self.observables)
                    y_hat = float(psi0 @ self.B @ psi0)
                    res = y_hat - float(yi)
                    b_loss += res ** 2
                    grad_B += 2.0 * res * np.outer(psi0, psi0)

                n = len(batch_X)
                self._B_adam_t += 1
                t = self._B_adam_t
                g = grad_B / n
                self._B_adam_m = beta1 * self._B_adam_m + (1 - beta1) * g
                self._B_adam_v = beta2 * self._B_adam_v + (1 - beta2) * g ** 2
                mh = self._B_adam_m / (1 - beta1 ** t)
                vh = self._B_adam_v / (1 - beta2 ** t)
                self.B -= self.lr * mh / (np.sqrt(vh) + eps)
                self.B = 0.5 * (self.B + self.B.T)   # re-symmetrise
                ep_loss += b_loss

            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1:4d}/{epochs}   MSE = {ep_loss / len(X):.4f}")

        return self


class QCMLClassifier(QCML):
    """
    QCML for classification via quantum measurement operators.

    Two-phase training
    ------------------
    Phase 1 (unsupervised):
        Call fit(X): standard Hamiltonian learning.

    Phase 2 (supervised):
        Call fit_measurement_operators(X, y): freezes observables and learns
        C measurement vectors w_i (rows of matrix W, shape C x m) via
        cross-entropy loss on the quantum-measurement probabilities:

            p_i(x) = (w_i . psi0(x))^2 / sum_j (w_j . psi0(x))^2

    Physical interpretation
    -----------------------
    Each w_i is a measurement direction in Hilbert space. The squared
    projection (w_i . psi)^2 is the Born-rule probability for measuring
    outcome i given state |psi>. The renormalisation enforces p_i >= 0
    and sum_i p_i = 1 exactly without a softmax layer.

    Gradient of L = -log(scores[c] / S) w.r.t. w_i (exact):
        dL/dw[c]   = (1/S - 1/scores[c]) * 2 * dots[c] * psi0
        dL/dw[i]   = (1/S)               * 2 * dots[i] * psi0   (i != c)
    where scores[i] = (w_i . psi0)^2, S = sum(scores), c = true class.

    Reference: [Q1] Musaelian et al. 2024, Section 4
    """

    def __init__(
        self,
        hilbert_dim: int = 16,
        lr: float = 3e-3,
        w: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(hilbert_dim, lr, w, seed)
        self.W: Optional[np.ndarray] = None   # (C, m) measurement vectors
        self._W_adam_m: Optional[np.ndarray] = None
        self._W_adam_v: Optional[np.ndarray] = None
        self._W_adam_t: int = 0
        self.C: int = 0

    def _init_W(self, C: int) -> None:
        self.C = C
        raw = self.rng.standard_normal((self.m, self.m))
        Q, _ = np.linalg.qr(raw)
        self.W = Q[:C, :]          # first C rows of orthonormal basis
        self._W_adam_m = np.zeros((C, self.m))
        self._W_adam_v = np.zeros((C, self.m))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Quantum measurement probabilities.  Shape (N, C).
        Each row sums to exactly 1; all entries in [0, 1].
        """
        proba = []
        for x in X:
            _, psi0 = ground_state(x, self.observables)
            scores = (self.W @ psi0) ** 2    # (C,), always >= 0
            S = scores.sum() + 1e-12
            proba.append(scores / S)
        return np.array(proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels.  Shape (N,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_measurement_operators(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> "QCMLClassifier":
        """
        Phase 2: learn measurement vectors W with frozen observables.

        Must be called after fit() has already trained the observables.
        """
        y = np.asarray(y, dtype=int)
        C = int(y.max()) + 1
        if self.W is None:
            self._init_W(C)

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for ep in range(epochs):
            perm = self.rng.permutation(len(X))
            ep_loss = 0.0

            for start in range(0, len(X), batch_size):
                idx = perm[start:start + batch_size]
                batch_X, batch_y = X[idx], y[idx]
                grad_W = np.zeros((self.C, self.m))
                b_loss = 0.0

                for xi, yi in zip(batch_X, batch_y):
                    _, psi0 = ground_state(xi, self.observables)
                    dots = self.W @ psi0            # (C,)
                    scores = dots ** 2              # (C,), >= 0
                    S = scores.sum() + 1e-10

                    b_loss -= np.log(scores[yi] / S + 1e-10)

                    # Gradient of L = -log(scores[c]) + log(S)
                    dL_dscores = np.full(self.C, 1.0 / S)
                    dL_dscores[yi] -= 1.0 / (scores[yi] + 1e-10)

                    # dL/dw[i] = dL/dscores[i] * d(dots[i]^2)/dw[i]
                    #           = dL/dscores[i] * 2 * dots[i] * psi0
                    for i in range(self.C):
                        grad_W[i] += dL_dscores[i] * 2.0 * dots[i] * psi0

                n = len(batch_X)
                self._W_adam_t += 1
                t = self._W_adam_t
                g = grad_W / n
                self._W_adam_m = beta1 * self._W_adam_m + (1 - beta1) * g
                self._W_adam_v = beta2 * self._W_adam_v + (1 - beta2) * g ** 2
                mh = self._W_adam_m / (1 - beta1 ** t)
                vh = self._W_adam_v / (1 - beta2 ** t)
                self.W -= self.lr * mh / (np.sqrt(vh) + eps)
                ep_loss += b_loss

            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1:4d}/{epochs}   CE loss = {ep_loss / len(X):.4f}")

        return self
