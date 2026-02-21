#!/usr/bin/env python3
"""
Bruteforce exact-k optimum for the QUBO E(x) = x^T Q x with x in {0,1}^n and sum(x)=k.

Outputs a JSON file containing:
- E_opt, x_opt (bitstring and index set)
- top_m solutions (sorted)
- some sanity stats

Usage:
  python experiments/bruteforce_optimum.py --artifacts artifacts/qaoa_artifacts.npz --out runs/classical_ref.json --topm 10
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True, help="Path to qaoa_artifacts.npz")
    p.add_argument("--out", required=True, help="Output JSON file")
    p.add_argument("--topm", type=int, default=10, help="Store top-M solutions")
    return p.parse_args()


def qubo_energy_from_indices(Q: np.ndarray, idx) -> float:
    """
    Compute x^T Q x where x is the indicator vector of subset idx.
    Equivalent to sum_{i in idx} sum_{j in idx} Q[i,j].
    """
    s = 0.0
    for i in idx:
        for j in idx:
            s += Q[i, j]
    return float(s)


def idx_to_bitstring(idx, n: int) -> str:
    x = ["0"] * n
    for i in idx:
        x[i] = "1"
    return "".join(x)


def main():
    args = parse_args()

    art = np.load(args.artifacts, allow_pickle=True)
    Q = np.array(art["Q"], dtype=float)
    n = int(art["n_qubits"])
    k = int(art["k_dicke_state"])

    # Symmetrize defensively (QUBO usually assumed symmetric)
    Q = 0.5 * (Q + Q.T)

    # Enumerate all exact-k bitstrings
    sols = []
    for idx in combinations(range(n), k):
        E = qubo_energy_from_indices(Q, idx)
        sols.append((E, idx))

    sols.sort(key=lambda t: t[0])

    E_opt, idx_opt = sols[0]
    bit_opt = idx_to_bitstring(idx_opt, n)

    topm = min(args.topm, len(sols))
    top_list = []
    for r in range(topm):
        E, idx = sols[r]
        top_list.append({
            "rank": r + 1,
            "energy_qubo": float(E),
            "indices": list(idx),
            "bitstring": idx_to_bitstring(idx, n)
        })

    out = {
        "n_qubits": n,
        "k_constraint": k,
        "objective": "minimize x^T Q x subject to sum(x)=k",
        "E_opt_qubo": float(E_opt),
        "x_opt_bitstring": bit_opt,
        "x_opt_indices": list(idx_opt),
        "num_feasible": int(len(sols)),
        "top_solutions": top_list,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== Bruteforce optimum (exact-k) ===")
    print(f"n={n}, k={k}, feasible count={len(sols)}")
    print(f"E_opt_qubo={E_opt}")
    print(f"x_opt={bit_opt} (indices={list(idx_opt)})")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
