"""QRU forecaster implemented with PennyLane + JAX.

This module implements a 1-qubit Quantum Re-Uploading Unit (QRU) forecaster,
aligned with the paper's design:

- History window H (e.g., 6 days)
- Re-uploading layers L (e.g., 4)
- For each layer and each day j in the window, apply an encoding block on wire 0
  using rainfall r_j and river level l_j (both scaled), plus trainable parameters.
- Then apply a small variational block.
- Output is <Z>, which lives in [-1,1]. Train on scaled targets.

Implementation notes (artifact-quality):
- deterministic init
- optax Adam
- early stopping on validation loss
- vectorized prediction

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# JAX/Optax are optional at import time for tooling; we fail with a clear message.
try:
    import jax
    import jax.numpy as jnp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "JAX is required for risk_pipeline.qru_jax. Install via requirements.txt"
    ) from e

try:
    import optax
except Exception as e:  # pragma: no cover
    raise ImportError(
        "optax is required for risk_pipeline.qru_jax. Install via requirements.txt"
    ) from e

try:
    import pennylane as qml
except Exception as e:  # pragma: no cover
    raise ImportError(
        "PennyLane is required for risk_pipeline.qru_jax. Install via requirements.txt"
    ) from e


@dataclass(frozen=True)
class QRUConfig:
    history: int = 6
    layers: int = 4
    seed: int = 0
    lr: float = 5e-3
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 25
    weight_decay: float = 0.0
    device_name: str = "default.qubit"
    shots: Optional[int] = None  # None => analytic expectation


def _init_params(key: "jax.Array", cfg: QRUConfig) -> Dict[str, jnp.ndarray]:
    """Initialize trainable parameters.

    Params are shaped to match the paper-like structure:
      - enc_r: (L, H)
      - enc_l: (L, H)
      - enc_z: (L, H)
      - var_x: (L,)
      - var_z: (L,)

    All are initialized small to avoid barren-ish regions.
    """
    L, H = cfg.layers, cfg.history
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    scale = 0.1
    return {
        "enc_r": scale * jax.random.normal(k1, (L, H)),
        "enc_l": scale * jax.random.normal(k2, (L, H)),
        "enc_z": scale * jax.random.normal(k3, (L, H)),
        "var_x": scale * jax.random.normal(k4, (L,)),
        "var_z": scale * jax.random.normal(k5, (L,)),
    }


def build_qru_qnode(cfg: QRUConfig):
    """Create a PennyLane QNode (JAX interface) returning <Z> on a single wire."""
    dev = qml.device(cfg.device_name, wires=1, shots=cfg.shots)

    @qml.qnode(dev, interface="jax")
    def qru_circuit(params: Dict[str, jnp.ndarray], x: jnp.ndarray):
        # x shape: (H, 2) -> columns: [rainfall, level] in SCALED space
        for ell in range(cfg.layers):
            for j in range(cfg.history):
                r = x[j, 0]
                l = x[j, 1]
                qml.RX(params["enc_r"][ell, j] * r, wires=0)
                qml.RZ(params["enc_z"][ell, j], wires=0)
                qml.RY(params["enc_l"][ell, j] * l, wires=0)
            qml.RX(params["var_x"][ell], wires=0)
            qml.RZ(params["var_z"][ell], wires=0)
        return qml.expval(qml.PauliZ(0))

    return qru_circuit


def predict(
    qnode,
    params: Dict[str, jnp.ndarray],
    X: np.ndarray,
) -> np.ndarray:
    """Vectorized prediction.

    X shape: (N, H, 2)
    Returns: (N,)
    """
    Xj = jnp.asarray(X)
    vmapped = jax.vmap(lambda xi: qnode(params, xi))
    y = vmapped(Xj)
    return np.asarray(y)


def train(
    cfg: QRUConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Train a QRU on scaled data.

    Returns
    -------
    best_params : dict[str, np.ndarray]
    metrics : dict[str, float]
    """
    qnode = build_qru_qnode(cfg)

    key = jax.random.PRNGKey(cfg.seed)
    params = _init_params(key, cfg)

    def loss_fn(p, xb, yb):
        preds = jax.vmap(lambda xi: qnode(p, xi))(xb)
        mse = jnp.mean((preds - yb) ** 2)
        if cfg.weight_decay > 0:
            l2 = sum([jnp.sum(v * v) for v in p.values()])
            mse = mse + cfg.weight_decay * l2
        return mse

    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s, xb, yb):
        l, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
        updates, s2 = opt.update(grads, s, p)
        p2 = optax.apply_updates(p, updates)
        return p2, s2, l

    Xtr = jnp.asarray(X_train)
    ytr = jnp.asarray(y_train)
    Xva = jnp.asarray(X_val)
    yva = jnp.asarray(y_val)

    n = X_train.shape[0]
    bs = int(cfg.batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be positive")

    best_params = params
    best_val = float("inf")
    best_epoch = -1
    patience_left = cfg.patience

    for epoch in range(cfg.max_epochs):
        # deterministic permutation per epoch
        key, sub = jax.random.split(key)
        idx = jax.random.permutation(sub, n)
        Xtr_s = Xtr[idx]
        ytr_s = ytr[idx]

        # minibatches
        losses = []
        for start in range(0, n, bs):
            end = min(start + bs, n)
            xb = Xtr_s[start:end]
            yb = ytr_s[start:end]
            params, opt_state, l = step(params, opt_state, xb, yb)
            losses.append(l)

        train_loss = float(jnp.mean(jnp.stack(losses)))
        val_loss = float(loss_fn(params, Xva, yva)) if X_val.shape[0] > 0 else train_loss

        if val_loss + 1e-10 < best_val:
            best_val = val_loss
            best_params = params
            best_epoch = epoch
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    metrics = {
        "best_val_mse": float(best_val),
        "best_epoch": float(best_epoch),
        "final_epoch": float(epoch),
    }

    # Convert to numpy for serialization
    best_np = {k: np.asarray(v) for k, v in best_params.items()}
    return best_np, metrics
