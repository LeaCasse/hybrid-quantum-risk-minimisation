#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_*.json from run_qaoa_inference.py")
    p.add_argument("--outdir", required=True, help="output directory for figures")
    p.add_argument("--topn", type=int, default=10)
    return p.parse_args()


def main():
    args = parse()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = json.load(open(args.run, "r", encoding="utf-8"))
    probs = d["probs"]
    weights = d["hamming_weights"]
    k = d["k_constraint"]
    n = d["n_qubits"]

    # ---------------------------
    # Figure 12: Hamming-weight histogram + mass kept
    # ---------------------------
    ws = np.array(list(weights.values()), dtype=int)
    ps = np.array([probs[b] for b in probs.keys()], dtype=float)

    # aggregate probability mass by weight
    mass_by_w = np.zeros(n + 1, dtype=float)
    for b, p in probs.items():
        mass_by_w[weights[b]] += p

    plt.figure()
    x = np.arange(n + 1)
    plt.bar(x, mass_by_w)
    plt.xlabel("Hamming weight |x|")
    plt.ylabel("Probability mass")
    plt.title(f"Hamming-weight distribution (k={k}), mass_kept={d['mass_kept']:.4f}")
    plt.tight_layout()
    plt.savefig(outdir / "fig12_hamming_mass.png", dpi=300)
    plt.close()

    # ---------------------------
    # Figure 13: Top-N feasible bitstrings (postselected)
    # ---------------------------
    feasible = {b: p for b, p in probs.items() if weights[b] == k}
    mass_kept = sum(feasible.values())

    if mass_kept > 0:
        feasible_post = {b: p / mass_kept for b, p in feasible.items()}
        top = sorted(feasible_post.items(), key=lambda kv: kv[1], reverse=True)[:args.topn]

        labels = [b for b, _ in top]
        vals = [v for _, v in top]

        plt.figure()
        plt.bar(np.arange(len(vals)), vals)
        plt.xticks(np.arange(len(vals)), labels, rotation=90)
        plt.xlabel("Feasible bitstrings (|x|=k)")
        plt.ylabel("Postselected probability")
        plt.title(f"Top-{args.topn} feasible bitstrings (postselected)")
        plt.tight_layout()
        plt.savefig(outdir / "fig13_top_feasible.png", dpi=300)
        plt.close()
    else:
        print("[WARN] No feasible mass, cannot plot Fig 13.")

    print("Wrote figures to:", outdir)


if __name__ == "__main__":
    main()
