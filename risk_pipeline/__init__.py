"""Risk Pipeline package.

Modules:
- scaling: deterministic reversible scalers
- dataset: windowing and temporal splits
- qru_jax: PennyLane+JAX QRU training/inference
- risk: quantile-band risk scoring (test-only)
- qubo: proxy QUBO + verified QUBO<->Ising conversion
- qaoa: QAOA ansatz utilities for exact-k and <=k budget constraints

"""
