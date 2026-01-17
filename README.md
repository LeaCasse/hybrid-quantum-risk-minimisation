# Hybrid Quantum Risk Minimisation — Artifact-Ready Code

This folder contains a **modular, testable** implementation of the pipeline described in
**"Hybrid Quantum Risk Minimisation: A QRU–QAOA Pipeline for Spatial Flood Tail Risk Allocation"**
and its accompanying notebook.

The design goals are:
- **No unit/scale ambiguity**: explicit fit/transform/inverse_transform contract.
- **No time leakage**: temporal splits; risk scoring uses **test-only** predictions.
- **Reproducibility**: fixed seeds, deterministic preprocessing, and unit tests.
- **Transparent optimization objective**:
  - The paper discusses a **CVaR-to-QUBO** formulation.
  - The shipped executable prototype uses a **quadratic proxy QUBO** (risk scores + spatial dependence),
    matching the notebook's NISQ constraints.

## Package layout

- `scaling.py`  
  Standardization or min-max-to-[-1,1] scalers + JSON serialization.

- `dataset.py`  
  Leak-free windowing and **temporal splits** (no random shuffles).

- `qru_jax.py`  
  1-qubit QRU forecaster using **PennyLane + JAX**, training on **scaled targets**.

- `risk.py`  
  Quantile-band risk scoring (p5/p95) and aggregation, **enforcing test-only usage**.

- `qubo.py`  
  Proxy QUBO builder + **verified QUBO↔Ising conversion**.

- `qaoa.py`  
  QAOA utilities:
  - **exact-k** selection via DickeState + XY mixer
  - **≤k** selection via penalty-encoded QUBO + X mixer

- `tests/`  
  Unit tests (`pytest`).

## Installation

### Option A — pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,quantum]
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate qru-qaoa-risk
```

## Running tests

```bash
pytest -q
```

## Minimal end-to-end usage (single site)

Below is a **minimal** sketch. You will adapt `feature_cols` / `target_col` to your dataset.

```python
import pandas as pd
from risk_pipeline.dataset import load_time_series_csv, make_supervised_windows, temporal_split
from risk_pipeline.scaling import make_scaler
from risk_pipeline.qru_jax import QRUConfig, train, predict, build_qru_qnode

# 1) Load
site_df = load_time_series_csv("data/site_A.csv", date_col="date")
feature_cols = ["rainfall_mm", "river_level"]   # rainfall + previous level
target_col = "river_level"                      # predict next-day level

# 2) Windowing (H=6 -> predict t+1)
ds = make_supervised_windows(site_df, feature_cols, target_col, history=6, horizon=1)
train_ds, val_ds, test_ds = temporal_split(ds, train_end="2021-12-31", val_end="2022-06-30")

# 3) Fit scalers on TRAIN ONLY
x_scaler = make_scaler("standard").fit(train_ds.X.reshape(len(train_ds.X), -1))
y_scaler = make_scaler("standard").fit(train_ds.y.reshape(-1, 1))

Xtr = x_scaler.transform(train_ds.X.reshape(len(train_ds.X), -1)).reshape(train_ds.X.shape)
Xva = x_scaler.transform(val_ds.X.reshape(len(val_ds.X), -1)).reshape(val_ds.X.shape)
Xte = x_scaler.transform(test_ds.X.reshape(len(test_ds.X), -1)).reshape(test_ds.X.shape)

ytr = y_scaler.transform(train_ds.y)
yva = y_scaler.transform(val_ds.y)

# 4) Train QRU in scaled space
cfg = QRUConfig(history=6, layers=4, seed=0, max_epochs=200)
best_params, metrics = train(cfg, Xtr, ytr, Xva, yva)

# 5) Predict (scaled) -> invert to physical units
pred_scaled = predict(build_qru_qnode(cfg), best_params, Xte)
pred_physical = y_scaler.inverse_transform(pred_scaled)
```

## Notes on correctness & scope

- If you change scaling (standard vs minmax), you must keep it consistent across:
  **training targets**, **inverse transform**, and **risk thresholds**.
- The shipped QUBO objective is the **proxy** used in the notebook (risk + correlation).
  If you want the full **empirical CVaR-QUBO**, implement it as a separate objective builder
  with auxiliary variables and add tests similar to `test_qubo.py`.

## Citation

If you reuse this artifact in a paper, cite the associated manuscript and provide:
- commit hash / release tag
- exact environment (requirements.txt or environment.yml)
- seeds used in experiments

