#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import numpy as np
import pennylane as qml

from qbraid.runtime import QbraidProvider
from qbraid.transpiler.conversions.pennylane import pennylane_to_qasm2

def hamming_weight(bitstring: str) -> int:
    return bitstring.count("1")

def strip_measurements(qasm: str) -> str:
    return "\n".join([ln for ln in qasm.splitlines() if "measure" not in ln])

def get_tape(qnode):
    for attr in ("tape", "qtape", "_tape"):
        if hasattr(qnode, attr):
            return getattr(qnode, attr)
    raise RuntimeError("Cannot access qnode tape")

# --- ZZ primitive ---
def exp_iZZ(theta, wires):
    a, b = wires
    qml.CNOT(wires=[a, b])
    qml.RZ(2 * theta, wires=b)
    qml.CNOT(wires=[a, b])

# --- XX via H⊗H ---
def exp_iXX(theta, wires):
    a, b = wires
    qml.Hadamard(a); qml.Hadamard(b)
    exp_iZZ(theta, wires=[a, b])
    qml.Hadamard(a); qml.Hadamard(b)

# --- YY via (S†H)⊗(S†H) ---
# IMPORTANT: apply H then S†, and inverse S then H
def exp_iYY(theta, wires):
    a, b = wires
    qml.Hadamard(a); qml.PhaseShift(-np.pi/2, wires=a)
    qml.Hadamard(b); qml.PhaseShift(-np.pi/2, wires=b)
    exp_iZZ(theta, wires=[a, b])
    qml.PhaseShift(np.pi/2, wires=a); qml.Hadamard(a)
    qml.PhaseShift(np.pi/2, wires=b); qml.Hadamard(b)

def apply_xy_ring_mixer(n, beta_x, beta_y):
    for i in range(n):
        j = (i + 1) % n
        exp_iXX(beta_x, wires=[i, j])
        exp_iYY(beta_y, wires=[i, j])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device-id", default="ionq_simulator")
    p.add_argument("--n", type=int, default=7)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--beta-x", type=float, default=0.7)
    p.add_argument("--beta-y", type=float, default=0.7)
    p.add_argument("--shots", type=int, default=10000)
    p.add_argument("--strip-measure", action="store_true")
    p.add_argument("--out", default="runs/mixer_test_qbraid.json")
    return p.parse_args()

def main():
    args = parse_args()
    n, k = args.n, args.k

    # init state |1...10...0> with k ones
    init_state = np.array([1]*k + [0]*(n-k), dtype=int)

    # Build export circuit (no shots)
    dev = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(dev)
    def export_circuit():
        for i, b in enumerate(init_state):
            if b == 1:
                qml.PauliX(i)
        apply_xy_ring_mixer(n, args.beta_x, args.beta_y)
        return qml.probs(wires=list(range(n)))

    _ = export_circuit()
    tape = get_tape(export_circuit)

    qasm2 = pennylane_to_qasm2(tape)
    if args.strip_measure:
        qasm2 = strip_measurements(qasm2)

    # Run on qBraid device
    provider = QbraidProvider()
    device = provider.get_device(args.device_id)

    t0 = time.time()
    run_ret = device.run([qasm2], shots=args.shots)
    runtime_sec = time.time() - t0

    obj = run_ret[0] if isinstance(run_ret, list) else run_ret
    result = obj.result() if hasattr(obj, "result") and callable(obj.result) else obj
    counts_obj = result.data.get_counts()

    # Normalize counts
    if isinstance(counts_obj, list):
        counts = counts_obj[0]
    elif isinstance(counts_obj, dict):
        # either direct dict or indexed dict
        if all(isinstance(v, (int, np.integer)) for v in counts_obj.values()):
            counts = counts_obj
        elif 0 in counts_obj and isinstance(counts_obj[0], dict):
            counts = counts_obj[0]
        else:
            counts = next(iter(counts_obj.values()))
    else:
        raise RuntimeError(f"Unknown counts type: {type(counts_obj)}")

    total = sum(counts.values())
    probs = {b: c/total for b, c in counts.items()}

    # Hamming mass at k
    mass_k = 0.0
    hist = {w: 0.0 for w in range(n+1)}
    for b, p in probs.items():
        w = hamming_weight(b)
        hist[w] += p
        if w == k:
            mass_k += p

    out = {
        "device_id": args.device_id,
        "n": n, "k": k,
        "beta_x": args.beta_x, "beta_y": args.beta_y,
        "shots": args.shots,
        "strip_measure": bool(args.strip_measure),
        "runtime_sec": runtime_sec,
        "mass_k": mass_k,
        "hist": hist,
        "counts": counts,
        "qasm_head": qasm2.splitlines()[:25],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("=== MIXER WEIGHT TEST ===")
    print("device_id :", args.device_id)
    print("n,k       :", n, k)
    print("beta_x,y  :", args.beta_x, args.beta_y)
    print("shots     :", args.shots)
    print("mass_k    :", mass_k)
    print("hist      :", hist)
    print("saved to  :", args.out)

if __name__ == "__main__":
    main()
