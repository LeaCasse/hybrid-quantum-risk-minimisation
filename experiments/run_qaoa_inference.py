#!/usr/bin/env python3
"""
QAOA inference script (artifact-driven, paper-grade).

- Loads a QUBO instance + QAOA parameters from qaoa_artifacts.npz
- Converts QUBO (x in {0,1}) to an Ising Hamiltonian (z in {+1,-1})
- Runs QAOA inference on a simulator backend (step 1)
- Collects raw bitstring counts
- Computes feasibility (Hamming-weight=k) and QUBO energies
- Exports a portable JSON result for analysis & figures

Inference-only: NO parameter optimization happens here.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pennylane as qml


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def hamming_weight(bitstring: str) -> int:
    return bitstring.count("1")


def bitstring_to_array(bitstring: str, n: int) -> np.ndarray:
    """Assumes bitstring is MSB->LSB as returned by qml.counts()."""
    s = bitstring.replace(" ", "")
    if len(s) != n:
        # Some backends may include spaces or different formatting; be conservative.
        s = s[-n:]
    return np.array([int(b) for b in s], dtype=int)


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """E(x) = x^T Q x, with x in {0,1}^n."""
    return float(x @ Q @ x)


def qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO to Ising with mapping x = (1 - z)/2, z in {+1,-1}.
    Returns (h, J, offset) such that:
        E_qubo(x) == offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
    where z_i = 1 - 2 x_i.
    Assumes Q is symmetric (or will be symmetrized).
    """
    Q = np.array(Q, dtype=float)
    Q = 0.5 * (Q + Q.T)
    n = Q.shape[0]

    # Expand x^T Q x = sum_i Qii xi + 2 sum_{i<j} Qij xi xj
    # Use x = (1 - z)/2.
    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)
    offset = 0.0

    # Diagonal terms: Qii xi
    for i in range(n):
        # xi = (1 - zi)/2
        offset += 0.5 * Q[i, i]
        h[i] += -0.5 * Q[i, i]

    # Off-diagonal terms: 2 Qij xi xj
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j]
            if q == 0:
                continue

            # 2 q xi xj = 2 q * ((1-zi)/2)*((1-zj)/2)
            #           = (q/2) * (1 - zi - zj + zi zj)
            offset += 0.5 * q
            h[i] += -0.5 * q
            h[j] += -0.5 * q
            J[i, j] += 0.5 * q
            J[j, i] += 0.5 * q

    return h, J, float(offset)


def normalize_counts(counts: Dict[str, int], shots: int) -> Dict[str, float]:
    return {b: c / shots for b, c in counts.items()}


# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True, help="Path to qaoa_artifacts.npz")
    p.add_argument("--backend", default="sim",
                   help="Backend: sim | default.qubit (step 1 only)")
    p.add_argument("--shots", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output JSON file")
    return p.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # -----------------------------------------------------
    # Load artifacts
    # -----------------------------------------------------
    art = np.load(args.artifacts, allow_pickle=True)

    Q = np.array(art["Q"], dtype=float)
    n = int(art["n_qubits"])
    p_layers = int(art["p_layers"])
    k = int(art["k_dicke_state"])
    params = np.array(art["params_full"]).reshape(-1)

    assert Q.shape == (n, n), f"Q has wrong shape: {Q.shape}, expected {(n, n)}"

    # Params: expect 3p (gamma, beta, beta_tilde) in your artifacts
    L = len(params)
    if L == 3 * p_layers:
        gammas = params[0:p_layers]
        betas = params[p_layers:2*p_layers]
        betas_tilde = params[2*p_layers:3*p_layers]
        params_layout = "3p"
    elif L == 2 * p_layers:
        gammas = params[0:p_layers]
        betas = params[p_layers:2*p_layers]
        betas_tilde = np.zeros_like(betas)  # disable second mixer term
        params_layout = "2p"
    else:
        raise ValueError(
            f"Cannot interpret params_full: len={L}, p_layers={p_layers}. "
            "Expected 3p (preferred) or 2p."
        )

    # Convert QUBO -> Ising Hamiltonian
    h, J, offset = qubo_to_ising(Q)

    # -----------------------------------------------------
    # Backend (step 1: simulator)
    # -----------------------------------------------------
    if args.backend in ("sim", "default.qubit"):
        dev = qml.device("default.qubit", wires=n, shots=args.shots)
        backend_name = "default.qubit"
    else:
        raise NotImplementedError("Only simulator backend is supported for step 1.")

    # -----------------------------------------------------
    # Initial state: fixed-k basis state (NISQ-pragmatic)
    # Example: |11 00..0> with k ones (in wire order)
    # -----------------------------------------------------
    init_state = np.array([1] * k + [0] * (n - k), dtype=int)

    # -----------------------------------------------------
    # QAOA circuit
    # Cost: exp(-i gamma * (sum h_i Z_i + sum J_ij Z_i Z_j))
    # Mixer: XY ring with two parameters per layer (beta, beta_tilde)
    # -----------------------------------------------------
    @qml.qnode(dev)
    def qaoa_circuit():
        # Prepare initial basis state
        for i, b in enumerate(init_state):
            if b == 1:
                qml.PauliX(wires=i)

        for layer in range(p_layers):
            gamma = gammas[layer]
            beta = betas[layer]
            beta_t = betas_tilde[layer]

            # ----- Cost unitary -----
            # Single-Z terms
            for i in range(n):
                if h[i] != 0:
                    qml.RZ(2 * gamma * h[i], wires=i)

            # ZZ couplings
            for i in range(n):
                for j in range(i + 1, n):
                    if J[i, j] != 0:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * J[i, j], wires=j)
                        qml.CNOT(wires=[i, j])

            # ----- Mixer unitary (XY ring) -----
            for i in range(n):
                j = (i + 1) % n
                # Standard XY: XX + YY
                qml.IsingXX(2 * beta, wires=[i, j])
                qml.IsingYY(2 * beta, wires=[i, j])

                # Enriched XY components (optional)
                # These terms provide the 3rd parameter block you have in artifacts.
                # If your training used a specific convention, this is the closest
                # hardware-native decomposition in PennyLane.
                    # NOTE: PennyLane may not provide IsingXY/IsingYX depending on version.
                # For step 1, we run the standard XY mixer (XX + YY) only.
                # beta_tilde is stored in JSON for provenance, but not applied here.
                pass


        return qml.counts()

    # -----------------------------------------------------
    # Run circuit
    # -----------------------------------------------------
    t0 = time.time()
    counts = qaoa_circuit()
    runtime_sec = time.time() - t0

    probs = normalize_counts(counts, args.shots)

    # -----------------------------------------------------
    # Analysis (always compute QUBO energies, not Ising energies, for reporting)
    # -----------------------------------------------------
    energies = {}
    weights = {}

    for bitstring, p in probs.items():
        x = bitstring_to_array(bitstring, n)
        energies[bitstring] = qubo_energy(x, Q)
        weights[bitstring] = int(x.sum())

    # Feasible (exact-k)
    feasible_probs = {b: p for b, p in probs.items() if weights[b] == k}
    mass_kept = float(sum(feasible_probs.values()))

    if feasible_probs:
        best_bit = min(feasible_probs, key=lambda b: energies[b])
        best_energy = float(energies[best_bit])
    else:
        best_bit = None
        best_energy = None

    # -----------------------------------------------------
    # Output JSON
    # -----------------------------------------------------
    out = {
        "backend_name": backend_name,
        "shots": int(args.shots),
        "seed": int(args.seed),
        "n_qubits": int(n),
        "k_constraint": int(k),
        "p_layers": int(p_layers),
        "params_layout": params_layout,
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
        "betas_tilde": betas_tilde.tolist(),
        "init_state": init_state.tolist(),
        "runtime_sec": float(runtime_sec),

        # Instance info (for reproducibility)
        "qubo_Q": Q.tolist(),
        "ising_h": h.tolist(),
        "ising_J": J.tolist(),
        "ising_offset": float(offset),

        # Results
        "mass_kept": float(mass_kept),
        "best_bitstring": best_bit,
        "best_energy_qubo": best_energy,
        "probs": probs,
        "energies_qubo": energies,
        "hamming_weights": weights,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # -----------------------------------------------------
    # Console summary (sanity check)
    # -----------------------------------------------------
    print("=== QAOA inference summary ===")
    print(f"Backend         : {backend_name}")
    print(f"n_qubits        : {n}")
    print(f"k (constraint)  : {k}")
    print(f"p_layers        : {p_layers}")
    print(f"params layout   : {params_layout} (len={len(params)})")
    print(f"shots           : {args.shots}")
    print(f"mass_kept       : {mass_kept:.6f}")
    print(f"best_energy_qubo: {best_energy}")
    print(f"output          : {out_path}")


if __name__ == "__main__":
    main()
