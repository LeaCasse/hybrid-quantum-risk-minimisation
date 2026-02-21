"""QAOA utilities (ansatz + exact-k vs <=k).

We build QAOA circuits for Ising cost Hamiltonians derived from QUBO.

Two constraint modes:
  1) exact-k selection: enforce Hamming weight = k using a Dicke initial state
     and an XY mixer (ring). This matches the notebook's constraint-preserving
     approach.
  2) <=k selection: use standard X mixer and encode budget via a quadratic penalty
     in the QUBO (leq-k in qubo.py). This is easier but less strict.

Note: This module is intentionally lightweight and artifact-friendly.
For large instances, you should move to dedicated QAOA toolchains.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import pennylane as qml
except Exception as e:  # pragma: no cover
    raise ImportError("PennyLane is required for risk_pipeline.qaoa") from e


@dataclass(frozen=True)
class QAOAConfig:
    p: int = 1
    seed: int = 0
    steps: int = 200
    lr: float = 0.1
    device_name: str = "default.qubit"
    shots: Optional[int] = None


def ising_to_cost_hamiltonian(h: np.ndarray, J: np.ndarray) -> qml.Hamiltonian:
    """Create PennyLane Hamiltonian H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    n = h.shape[0]

    coeffs = []
    ops = []

    for i in range(n):
        if abs(h[i]) > 0:
            coeffs.append(h[i])
            ops.append(qml.PauliZ(i))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 0:
                coeffs.append(J[i, j])
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    if len(coeffs) == 0:
        coeffs = [0.0]
        ops = [qml.Identity(0)]

    return qml.Hamiltonian(coeffs, ops)


def _xy_ring_mixer(wires):
    # PennyLane provides xy_mixer; we wrap for explicitness.
    return qml.qaoa.xy_mixer(wires)


def build_qaoa_qnode(
    cost_h: qml.Hamiltonian,
    cfg: QAOAConfig,
    constraint: str = "exact",
    k: Optional[int] = None,
) -> Tuple[Callable, int]:
    """Build a QAOA circuit returning expectation of the cost Hamiltonian.

    constraint:
      - 'exact': requires k (Hamming weight). Uses DickeState + XY mixer.
      - 'leq'  : ignores k, uses |+>^n + X mixer.

    Returns (qnode, n_qubits).
    """
    # infer n from max wire index in the Hamiltonian
    wires = set()
    for op in cost_h.ops:
        wires |= set(op.wires)
    n = max(wires) + 1 if wires else 1

    dev = qml.device(cfg.device_name, wires=n, shots=cfg.shots)

    constraint = constraint.lower().strip()
    if constraint == "exact" and k is None:
        raise ValueError("constraint='exact' requires k")

    def _prepare_initial_state():
        if constraint == "exact":
            qml.templates.state_preparations.DickeState(k, wires=range(n))
        else:
            for i in range(n):
                qml.Hadamard(wires=i)

    def _mixer(beta):
        if constraint == "exact":
            qml.qaoa.mixer_layer(_xy_ring_mixer(range(n)), beta)
        else:
            qml.qaoa.mixer_layer(qml.qaoa.x_mixer(range(n)), beta)

    def _cost(gamma):
        qml.qaoa.cost_layer(cost_h, gamma)

    @qml.qnode(dev)
    def circuit(params):
        gammas = params[: cfg.p]
        betas = params[cfg.p :]

        _prepare_initial_state()
        for layer in range(cfg.p):
            _cost(gammas[layer])
            _mixer(betas[layer])
        return qml.expval(cost_h)

    return circuit, n


def solve_qaoa(
    cost_h: qml.Hamiltonian,
    cfg: QAOAConfig,
    constraint: str = "exact",
    k: Optional[int] = None,
) -> Tuple[np.ndarray, float, Callable]:
    """Optimize QAOA parameters to minimize <cost_h>.

    Returns (best_params, best_energy, qnode).
    """
    circuit, _ = build_qaoa_qnode(cost_h, cfg, constraint=constraint, k=k)

    rng = np.random.default_rng(cfg.seed)
    params = rng.normal(scale=0.1, size=(2 * cfg.p,)).astype(float)

    opt = qml.GradientDescentOptimizer(stepsize=cfg.lr)

    best_params = params.copy()
    best_energy = float(circuit(best_params))

    for _ in range(cfg.steps):
        params = opt.step(circuit, params)
        e = float(circuit(params))
        if e < best_energy:
            best_energy = e
            best_params = np.array(params, dtype=float)

    return best_params, best_energy, circuit


def sample_bitstrings(
    qnode: Callable,
    params: np.ndarray,
    n_qubits: int,
    shots: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample bitstrings from the trained QAOA circuit.

    For this we rebuild a sampling circuit with the same ansatz, returning samples.
    """
    # This helper assumes qnode was built with default.qubit; for other devices
    # you may want a more explicit interface.
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    # Try to access cost hamiltonian from closure is not robust; we require
    # the caller to rebuild a sampling circuit if needed. Here we provide a generic
    # template by inspecting qnode.qtape, which is available after first call.
    _ = qnode(params)  # build tape
    tape = qnode.qtape

    @qml.qnode(dev)
    def sampler(p):
        qml.apply(tape.operations[0])
        for op in tape.operations[1:]:
            qml.apply(op)
        return qml.sample(wires=range(n_qubits))

    samples = sampler(params)
    # samples shape (shots, n_qubits) in {0,1}
    # compress to unique bitstrings
    arr = np.asarray(samples, dtype=int)
    uniq, counts = np.unique(arr, axis=0, return_counts=True)
    return uniq, counts


# --- Convenience: inference-only on default.qubit ---
def run_qaoa_inference_default_qubit(h, J, gammas, betas, shots: int = 10_000, k: int | None = None):
    """Run inference-only QAOA on PennyLane's default.qubit and return raw counts.

    This helper is used by `experiments/end_to_end_sim_demo.py`. For paper figures
    and backend-specific routes (qBraid/IBM), see scripts in `experiments/`.
    """
    import pennylane as qml

    n = len(h)
    dev = qml.device("default.qubit", wires=n, shots=shots)
    H = ising_to_cost_hamiltonian(h, J)

    @qml.qnode(dev)
    def qnode():
        if k is not None:
            qml.templates.state_preparations.DickeState(k, wires=range(n))
        else:
            for w in range(n):
                qml.Hadamard(wires=w)

        for gamma, beta in zip(gammas, betas):
            qml.ApproxTimeEvolution(H, gamma, 1)
            if k is not None:
                _xy_ring_mixer(beta, wires=list(range(n)))
            else:
                for w in range(n):
                    qml.RX(2 * beta, wires=w)

        return qml.counts()

    return qnode()
