# Reproducibility map (paper ↔ code)

This repository is split into:

- `risk_pipeline/`: **core library** (unit-tested, stable APIs)
- `experiments/`: **paper reproduction scripts** (I/O, backends, plotting)
- `configs/`: YAML configs for the paper runs
- `artifacts/`: generated outputs (gitignored by default)

## Paper components

### QRU forecaster (sim)
- Core: `risk_pipeline/qru_jax.py`
- Train: `experiments/qru_train.py`
- Config: `configs/paper_qru_72params.yaml`

### QRU on QPU (IBM)
- Script: `experiments/qru_qpu_ibm_batch.py` (batch runtime / estimator)
- Script: `experiments/qru_qpu_ibm_predict.py` (simple baseline)
- Config: `configs/paper_qpu_ibm.yaml`

### Risk scoring
- Core: `risk_pipeline/risk.py`
- The band mapping is `band_risk(h, p5, p95)` with clipping into [0,1].

### Coupling matrix C
- Script: `experiments/build_c_matrix.py`
- The paper's components are stored alongside the final `C`.

### QUBO / Ising conversion
- Core: `risk_pipeline/qubo.py`
- Tests: `tests/test_qubo.py`

### QAOA (sim)
- Core: `risk_pipeline/qaoa.py`
- End-to-end demo: `experiments/end_to_end_sim_demo.py`

### QAOA via qBraid
- Inference: `experiments/run_qaoa_inference_qbraid_ionq.py`
- Mixer leakage diagnostic: `experiments/qaoa_mixer_only_leakage_qbraid.py`
- Config: `configs/paper_qbraid_ionq.yaml`

## Notes on CVaR
The repository implements the **quadratic proxy objective** used in the prototype runs.
The full CVaR-to-QUBO discretization (η grid + one-hot selection) is included in the paper
as a general construction; a tiny demo can be added if required by reviewers.
