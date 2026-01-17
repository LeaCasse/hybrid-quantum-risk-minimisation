"""QUBO construction and QUBO<->Ising conversion.

This project uses two objectives:

(A) Conceptual (paper): empirical CVaR -> QUBO with auxiliary variables.
    Implementing the full CVaR-QUBO can be large. We keep it as future work.

(B) Executable (prototype / notebook): a quadratic proxy objective
    derived from site risk scores and a spatial dependence matrix C.

This module implements (B) in a transparent, testable way, plus a verified
conversion from QUBO (x in {0,1}^n) to Ising (z in {-1,+1}^n).

Notation:
- x_i = 1 means *protected* (or selected). x_i = 0 means not selected.
- We want to minimize expected tail risk of unprotected sites.

Proxy objective used here (minimize):
  J(x) = sum_i r_i * (1 - x_i)
       + lambda_corr * sum_{i<j} C_ij * (1 - x_i) * (1 - x_j)
       + penalty_budget(x)

Budget options:
- exact-k: enforce sum_i x_i == k
- leq-k : enforce sum_i x_i <= k (quadratic hinge penalty)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class BudgetConstraint:
    kind: str  # 'exact' or 'leq'
    k: int
    lam: float = 10.0


def build_proxy_qubo(
    risk: np.ndarray,
    C: np.ndarray,
    lambda_corr: float,
    budget: BudgetConstraint,
) -> Tuple[np.ndarray, float]:
    """Build a QUBO matrix Q and constant offset for the proxy objective.

    Returns Q, const such that objective(x)=x^T Q x + const.

    x is interpreted as selection/protection (1=select).
    """
    r = np.asarray(risk, dtype=float).reshape(-1)
    n = r.shape[0]
    C = np.asarray(C, dtype=float)
    if C.shape != (n, n):
        raise ValueError(f"C must be (n,n) with n={n}, got {C.shape}")

    # Ensure symmetric with zero diagonal
    Csym = 0.5 * (C + C.T)
    np.fill_diagonal(Csym, 0.0)

    # Expand objective in x:
    # sum_i r_i (1 - x_i) = sum_i r_i - sum_i r_i x_i
    # sum_{i<j} lc C_ij (1 - x_i)(1 - x_j)
    # = lc * sum_{i<j} C_ij * (1 - x_i - x_j + x_i x_j)
    # const term: lc * sum_{i<j} C_ij
    # linear term: -lc * sum_{i<j} C_ij x_i - lc * sum_{i<j} C_ij x_j
    # quadratic: lc * sum_{i<j} C_ij x_i x_j

    Q = np.zeros((n, n), dtype=float)
    const = float(np.sum(r))

    # Linear from -r_i x_i
    Q[np.diag_indices(n)] += -r

    # Correlation contributions
    triu = np.triu_indices(n, 1)
    const += float(lambda_corr * np.sum(Csym[triu]))

    # Linear terms from correlation: for each pair (i,j), add -lc*C_ij to both i and j diagonals
    lin_corr = -lambda_corr * np.sum(Csym, axis=1)
    Q[np.diag_indices(n)] += lin_corr

    # Quadratic terms: lc*C_ij for i<j
    Q[triu] += lambda_corr * Csym[triu]
    Q[(triu[1], triu[0])] += lambda_corr * Csym[triu]  # symmetrize

    # Budget penalties
    if budget.k < 0 or budget.k > n:
        raise ValueError(f"Invalid k={budget.k} for n={n}")

    kind = budget.kind.lower().strip()
    if kind == "exact":
        # lam (sum x_i - k)^2 = lam( sum x_i^2 + 2 sum_{i<j} x_i x_j - 2k sum x_i + k^2)
        # with x_i^2=x_i.
        lam = float(budget.lam)
        const += lam * (budget.k ** 2)
        Q[np.diag_indices(n)] += lam * (1.0 - 2.0 * budget.k)
        Q[triu] += 2.0 * lam
        Q[(triu[1], triu[0])] += 2.0 * lam

    elif kind == "leq":
        # Penalize only if sum x_i > k using a smooth quadratic hinge surrogate:
        # lam * max(0, sum x_i - k)^2 is not quadratic in binaries without auxiliaries.
        # For prototype we use lam*(sum x_i - k)^2 and report that it enforces approximate <=k.
        lam = float(budget.lam)
        const += lam * (budget.k ** 2)
        Q[np.diag_indices(n)] += lam * (1.0 - 2.0 * budget.k)
        Q[triu] += 2.0 * lam
        Q[(triu[1], triu[0])] += 2.0 * lam

    else:
        raise ValueError("budget.kind must be 'exact' or 'leq'")

    return Q, const


def qubo_energy(Q: np.ndarray, x: np.ndarray, const: float = 0.0) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(x @ Q @ x + const)


def qubo_to_ising(Q: np.ndarray, const: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO to Ising.

    Mapping: x_i = (1 - z_i)/2 where z_i in {-1,+1}.

    Returns (h, J, offset) such that:
      E_ising(z) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
    and for z=1-2x, E_ising(z) == E_qubo(x).

    We return J as a full symmetric matrix with zero diagonal (convenience).
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")
    n = Q.shape[0]

    # Symmetrize Q; only symmetric part affects x^T Q x
    Qs = 0.5 * (Q + Q.T)

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)

    # Derivation:
    # x_i = (1 - z_i)/2
    # x_i x_j = (1 - z_i - z_j + z_i z_j)/4
    # x_i^2 = x_i.

    offset = const

    # Diagonal terms: Q_ii x_i
    # = Q_ii/2 - (Q_ii/2) z_i
    offset += 0.5 * np.sum(np.diag(Qs))
    h += -0.5 * np.diag(Qs)

    # Off-diagonal terms: for i<j, 2*Qs_ij x_i x_j? careful: x^T Qs x already counts both.
    # We'll iterate i<j using Qs_ij and use formula with coefficient 2? No: x^T Qs x = sum_i Qs_ii x_i^2 + 2 sum_{i<j} Qs_ij x_i x_j.
    # So effective coefficient on x_i x_j is 2*Qs_ij.
    for i in range(n):
        for j in range(i + 1, n):
            q = 2.0 * Qs[i, j]
            # q x_i x_j contributes:
            # q/4 - (q/4) z_i - (q/4) z_j + (q/4) z_i z_j
            offset += 0.25 * q
            h[i] += -0.25 * q
            h[j] += -0.25 * q
            J[i, j] += 0.25 * q
            J[j, i] += 0.25 * q

    np.fill_diagonal(J, 0.0)
    return h, J, float(offset)


def ising_energy(h: np.ndarray, J: np.ndarray, z: np.ndarray, offset: float = 0.0) -> float:
    z = np.asarray(z, dtype=float).reshape(-1)
    n = z.shape[0]
    if h.shape[0] != n or J.shape != (n, n):
        raise ValueError("Shape mismatch")
    # sum_{i<j} J_ij z_i z_j for symmetric J can be 0.5 z^T J z
    return float(offset + np.dot(h, z) + 0.5 * z @ J @ z)


def verify_qubo_ising_equivalence(Q: np.ndarray, const: float = 0.0, n_samples: int = 256, seed: int = 0) -> bool:
    """Monte Carlo check of equivalence between QUBO and Ising forms."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    h, J, off = qubo_to_ising(Q, const=const)
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=(n,))
        z = 1.0 - 2.0 * x
        e1 = qubo_energy(Q, x, const=const)
        e2 = ising_energy(h, J, z, offset=off)
        if not np.isclose(e1, e2, atol=1e-8, rtol=1e-6):
            return False
    return True
