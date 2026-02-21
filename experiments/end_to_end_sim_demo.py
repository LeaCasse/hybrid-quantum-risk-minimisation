#!/usr/bin/env python3
"""End-to-end simulator demo: QRU -> risk -> QUBO -> QAOA.

This is a *small* runnable example intended for reviewers/readers to reproduce
the full chain without QPU access.

Outputs:
- artifacts/demo_qubo.npz
- artifacts/demo_qaoa_artifacts.npz
- artifacts/demo_qaoa_counts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from risk_pipeline.qubo import build_proxy_qubo, qubo_to_ising
from risk_pipeline.qaoa import run_qaoa_inference_default_qubit
from risk_pipeline.risk import band_risk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts", help="Output directory")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--shots", type=int, default=5000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Synthetic per-site risk scores (as in the paper: aggregated daily risk in [0,1])
    rng = np.random.default_rng(123)
    raw = rng.normal(size=args.n)
    # make pseudo "extreme" distribution and map to [0,1]
    r = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    # Simple coupling matrix C in [0,1] (geo-like kernel)
    d = np.abs(np.subtract.outer(np.arange(args.n), np.arange(args.n)))
    C = 1.0 / (1.0 + d)
    np.fill_diagonal(C, 0.0)

    Q, const = build_proxy_qubo(r, C, k=args.k, lambda_corr=1.0, lambda_k=10.0)

    np.savez_compressed(out / "demo_qubo.npz", Q=Q, const=const, r=r, C=C, n=args.n, k=args.k)

    h, J, offset = qubo_to_ising(Q, const)

    # Inference-only QAOA (parameters here are toy; for paper you store optimized params)
    gammas = np.array([0.8] * args.p)
    betas = np.array([0.7] * args.p)

    counts = run_qaoa_inference_default_qubit(h, J, gammas, betas, shots=args.shots, k=args.k)

    (out / "demo_qaoa_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    np.savez_compressed(out / "demo_qaoa_artifacts.npz", Q=Q, const=const, h=h, J=J, offset=offset,
                        gammas=gammas, betas=betas, n=args.n, k=args.k, p=args.p)

    print("Wrote demo artifacts to", out)


if __name__ == "__main__":
    main()
