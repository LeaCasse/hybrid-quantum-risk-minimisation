#!/usr/bin/env python3
"""
run_qaoa_inference_qbraid_ionq.py

Artifact-driven QAOA inference via qBraid -> IonQ (QASM2 route).

- Loads QAOA artifacts from qaoa_artifacts.npz (Q, n, k, p, params_full)
- Builds a QASM2-friendly QAOA circuit in PennyLane (only H,S,RZ,CX,X)
- Converts tape -> OpenQASM 2.0 with qBraid transpiler
- (Optionally) strips measurements from QASM2 (recommended for IonQ)
- Submits 1 circuit to qBraid device (default: ionq_simulator)
- Collects raw counts and exports a JSON compatible with your paper pipeline

Outputs:
- runs/run_qbraid_ionq_sim.json (same structure as run_sim_paper.json but from qBraid counts)
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pennylane as qml

from qbraid.runtime import QbraidProvider
from qbraid.transpiler.conversions.pennylane import pennylane_to_qasm2


# -------------------------
# Utils
# -------------------------

def strip_measurements_from_qasm2(qasm: str) -> str:
    lines = []
    for ln in qasm.splitlines():
        if "measure" in ln:
            continue
        if ln.strip().startswith("creg "):
            # keep creg; harmless, but can remove if you prefer
            lines.append(ln)
            continue
        lines.append(ln)
    return "\n".join(lines)


def hamming_weight(bitstring: str) -> int:
    return bitstring.count("1")


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    return float(x @ Q @ x)


def bitstring_to_array(bitstring: str, n: int, order: str) -> np.ndarray:
    """
    order = 'msb0' : bitstring[0] is qubit-0 (q[0])   (no reversal)
    order = 'lsb0' : bitstring[-1] is qubit-0         (reverse)
    """
    s = bitstring.strip()
    # some backends might return with spaces or prefixes; be defensive
    s = "".join([c for c in s if c in "01"])
    if len(s) != n:
        # If backend returns shorter strings, left-pad with zeros
        if len(s) < n:
            s = s.zfill(n)
        else:
            s = s[-n:]
    if order == "msb0":
        bits = [int(c) for c in s]
    elif order == "lsb0":
        bits = [int(c) for c in s[::-1]]
    else:
        raise ValueError("order must be 'msb0' or 'lsb0'")
    return np.array(bits, dtype=int)


# -------------------------
# QASM-friendly two-qubit evolutions
# -------------------------

def exp_iXX(theta, wires):
    """Implements exp(-i * theta * X⊗X) up to global phase, using only QASM2 gates."""
    a, b = wires
    qml.Hadamard(a); qml.Hadamard(b)
    qml.CNOT(wires=[a, b])
    qml.RZ(2 * theta, wires=b)
    qml.CNOT(wires=[a, b])
    qml.Hadamard(a); qml.Hadamard(b)


def exp_iYY(theta, wires):
    """Implements exp(-i * theta * Y⊗Y) up to global phase, QASM2-friendly."""
    a, b = wires

    # U = S† H  => apply H then S† (because circuit applies left-to-right)
    qml.Hadamard(wires=a)
    qml.PhaseShift(-np.pi/2, wires=a)  # S†

    qml.Hadamard(wires=b)
    qml.PhaseShift(-np.pi/2, wires=b)  # S†

    # ZZ interaction via CNOT-RZ-CNOT
    qml.CNOT(wires=[a, b])
    qml.RZ(2 * theta, wires=b)
    qml.CNOT(wires=[a, b])

    # U† = H S  => apply S then H (left-to-right)
    qml.PhaseShift(np.pi/2, wires=a)   # S
    qml.Hadamard(wires=a)

    qml.PhaseShift(np.pi/2, wires=b)   # S
    qml.Hadamard(wires=b)




def apply_cost_layer(Q, gamma):
    """
    Paper pipeline cost layer used previously (diagonal RZ + ZZ terms via CNOT-RZ-CNOT).
    This is kept consistent with your existing artifacts/params.
    """
    n = Q.shape[0]
    # diagonal
    for i in range(n):
        if Q[i, i] != 0:
            qml.RZ(2 * gamma * float(Q[i, i]), wires=i)
    # off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            qij = float(Q[i, j])
            if qij != 0:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * qij, wires=j)
                qml.CNOT(wires=[i, j])


def apply_xy_ring_mixer(n, beta_x, beta_y):
    """XY ring mixer using QASM2-friendly decompositions."""
    for i in range(n):
        j = (i + 1) % n
        if beta_x != 0:
            exp_iXX(beta_x, wires=[i, j])
        if beta_y != 0:
            exp_iYY(beta_y, wires=[i, j])


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True, help="Path to qaoa_artifacts.npz")
    p.add_argument("--device-id", default="ionq_simulator",
                   help="qBraid device id (e.g., ionq_simulator)")
    p.add_argument("--shots", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/run_qbraid_ionq_sim.json")
    p.add_argument("--strip-measure", action="store_true",
                   help="Strip measure statements from QASM2 before submission (recommended).")
    p.add_argument("--bitorder", default="msb0", choices=["msb0", "lsb0"],
                   help="Interpretation of returned bitstrings when mapping to x (affects energies).")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    art = np.load(args.artifacts, allow_pickle=True)
    Q = np.array(art["Q"], dtype=float)
    n = int(art["n_qubits"])
    p_layers = int(art["p_layers"])
    k = int(art["k_dicke_state"])
    params = np.array(art["params_full"], dtype=float).ravel()

    # Defensive symmetrization (won't change x^TQx if Q already symmetric)
    Q = 0.5 * (Q + Q.T)

    # params layout: accept 2p or 3p
    if len(params) == 2 * p_layers:
        gammas = params[:p_layers]
        betas_x = params[p_layers:]
        betas_y = params[p_layers:]  # same
        params_layout = "2p (shared betas)"
    elif len(params) == 3 * p_layers:
        gammas = params[:p_layers]
        betas_x = params[p_layers:2 * p_layers]
        betas_y = params[2 * p_layers:3 * p_layers]
        params_layout = "3p (beta_x, beta_y)"
    else:
        raise ValueError(f"params_full length {len(params)} not compatible with 2p or 3p (p={p_layers})")

    # Initial basis state with k ones: |11..100..0>
    init_state = np.array([1] * k + [0] * (n - k), dtype=int)

    # Build export QNode on default.qubit to get tape -> QASM2
    dev_export = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(dev_export)
    def export_circuit():
        # prepare |11..100..0>
        for i, b in enumerate(init_state):
            if b == 1:
                qml.PauliX(wires=i)

        # QAOA layers
        for layer in range(p_layers):
            apply_cost_layer(Q, gammas[layer])
            apply_xy_ring_mixer(n, betas_x[layer], betas_y[layer])

        # return probs to force measurement insertion in qasm2; we will strip if needed
        return qml.probs(wires=list(range(n)))

    _ = export_circuit()

    # get tape robustly
    tape = None
    for attr in ("tape", "qtape", "_tape"):
        if hasattr(export_circuit, attr):
            tape = getattr(export_circuit, attr)
            break
    if tape is None:
        raise RuntimeError("Could not access QNode tape for QASM2 conversion.")

    qasm2 = pennylane_to_qasm2(tape)
    if args.strip_measure:
        qasm2 = strip_measurements_from_qasm2(qasm2)

    # Submit to qBraid device
    provider = QbraidProvider()
    device = provider.get_device(args.device_id)

    t0 = time.time()
    run_ret = device.run([qasm2], shots=args.shots)
    runtime_sec = time.time() - t0
    
    # --- Normalize qBraid return types across versions ---
    # Possible returns:
    #   - Job
    #   - [Job]
    #   - Result
    #   - [Result]
    obj = run_ret
    
    # If list, take first element (we submitted 1 circuit)
    if isinstance(obj, list):
        if len(obj) == 0:
            raise RuntimeError("device.run returned an empty list.")
        obj = obj[0]
    
    # If Job-like, fetch result
    if hasattr(obj, "result") and callable(getattr(obj, "result")):
        result = obj.result()
    else:
        # Already a result-like object
        result = obj
    
    # Extract counts
    if not hasattr(result, "data"):
        raise RuntimeError(f"Result object has no .data attribute. type={type(result)}")
    counts_obj = result.data.get_counts()
    
    # counts_obj can be:
    #  - dict {bitstring: count}  (single circuit)
    #  - list[dict]              (batch)
    #  - dict {int: dict}        (indexed batch)
    # Handle all cases robustly.
    
    if counts_obj is None:
        raise RuntimeError("get_counts() returned None.")
    
    # Case 1: list of dicts
    if isinstance(counts_obj, list):
        if len(counts_obj) == 0:
            raise RuntimeError("get_counts() returned an empty list.")
        counts = counts_obj[0]
    
    # Case 2: dict
    elif isinstance(counts_obj, dict):
        # If values are ints -> already counts dict
        if all(isinstance(v, (int, np.integer)) for v in counts_obj.values()):
            counts = counts_obj
        else:
            # Maybe indexed dict: {0: {...}, 1: {...}}
            if 0 in counts_obj and isinstance(counts_obj[0], dict):
                counts = counts_obj[0]
            else:
                # fall back: take first dict-like value
                first = next(iter(counts_obj.values()))
                if not isinstance(first, dict):
                    raise RuntimeError(f"Unrecognized get_counts() dict structure: {counts_obj}")
                counts = first
    
    else:
        raise RuntimeError(f"Unrecognized get_counts() return type: {type(counts_obj)}")
    
    # Final sanity
    if not isinstance(counts, dict) or len(counts) == 0:
        raise RuntimeError(f"Counts malformed or empty. type={type(counts)}, counts={counts}")

    # normalize to probabilities
    total = sum(counts.values())
    probs = {b: c / total for b, c in counts.items()}

    # compute energies + weights
    energies = {}
    weights = {}
    for b, p in probs.items():
        x = bitstring_to_array(b, n, order=args.bitorder)
        energies[b] = qubo_energy(x, Q)
        weights[b] = int(x.sum())

    feasible = {b: p for b, p in probs.items() if weights[b] == k}
    mass_kept = float(sum(feasible.values()))

    if feasible:
        best_bit = min(feasible, key=lambda b: energies[b])
        best_energy = float(energies[best_bit])
    else:
        best_bit = None
        best_energy = None

    out = {
        "backend_name": f"qbraid:{args.device_id}",
        "shots": int(args.shots),
        "seed": int(args.seed),
        "n_qubits": int(n),
        "k_constraint": int(k),
        "p_layers": int(p_layers),
        "params_layout": params_layout,
        "init_state": init_state.tolist(),
        "runtime_sec": float(runtime_sec),
        "mass_kept": mass_kept,
        "best_bitstring": best_bit,
        "best_energy_qubo": best_energy,
        "probs": probs,
        "energies_qubo": energies,
        "hamming_weights": weights,   # weights in terms of x (after bitorder mapping)
        "counts_raw": counts,
        "bitorder": args.bitorder,
        "strip_measure": bool(args.strip_measure),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== qBraid IonQ inference summary ===")
    print(f"device_id      : {args.device_id}")
    print(f"shots          : {args.shots}")
    print(f"n_qubits       : {n}")
    print(f"k (constraint) : {k}")
    print(f"p_layers       : {p_layers}")
    print(f"params layout  : {params_layout} (len={len(params)})")
    print(f"bitorder       : {args.bitorder}")
    print(f"strip_measure  : {args.strip_measure}")
    print(f"mass_kept      : {mass_kept:.6f}")
    print(f"best_energy    : {best_energy}")
    print(f"best_bitstring : {best_bit}")
    print(f"output         : {out_path}")


if __name__ == "__main__":
    main()
