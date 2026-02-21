#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="runs/run_sim_paper.json (or QPU run)")
    p.add_argument("--ref", required=True, help="runs/classical_ref.json")
    p.add_argument("--outdir", required=True, help="output directory")
    p.add_argument("--topm", type=int, default=10, help="Top-M solutions to display")
    return p.parse_args()

def main():
    args = parse()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = json.load(open(args.run, "r", encoding="utf-8"))
    c = json.load(open(args.ref, "r", encoding="utf-8"))

    probs = {b: float(p) for b, p in r["probs"].items()}
    weights = {b: int(w) for b, w in r["hamming_weights"].items()}
    energies = {b: float(e) for b, e in r["energies_qubo"].items()}
    k = int(r["k_constraint"])
    n = int(r["n_qubits"])

    # Postselect exact-k
    feasible = {b: p for b, p in probs.items() if weights[b] == k}
    mass_kept = sum(feasible.values())
    if mass_kept <= 0:
        raise RuntimeError("No feasible mass to analyze.")

    post = {b: p / mass_kept for b, p in feasible.items()}

    # Classical top solutions
    top_solutions = c["top_solutions"]
    topm = min(args.topm, len(top_solutions))
    top_bits = [top_solutions[i]["bitstring"] for i in range(topm)]
    top_E = [float(top_solutions[i]["energy_qubo"]) for i in range(topm)]
    E_opt = float(c["E_opt_qubo"])
    x_opt = c["x_opt_bitstring"]

    # Probabilities of those solutions under QAOA (postselected)
    top_P = [post.get(b, 0.0) for b in top_bits]

    # Metrics: P(opt), cumulative top-mass, expected energy gap
    P_opt = post.get(x_opt, 0.0)
    cum_top = float(sum(top_P))
    E_exp = float(sum(post[b] * energies[b] for b in post.keys()))
    gap = E_exp - E_opt

    # -------- Figure 14: top-M by energy with QAOA postselected probs --------
    plt.figure()
    x = np.arange(topm)
    plt.bar(x, top_P)
    plt.xticks(x, [f"{i+1}" for i in range(topm)])
    plt.xlabel("Rank by classical energy (1 = optimum)")
    plt.ylabel("Postselected probability")
    plt.title(f"QAOA mass on classical top-{topm} (k={k}, mass_kept={mass_kept:.4f})")
    # annotate rank-1
    if topm >= 1:
        plt.annotate(f"opt bit={x_opt}\nE*={E_opt:.4f}\nP*={P_opt:.3f}",
                     xy=(0, top_P[0]), xytext=(0.2, max(top_P)*0.8),
                     arrowprops=dict(arrowstyle="->"))
    plt.tight_layout()
    plt.savefig(outdir / "fig14_topM_mass_vs_optimum.png", dpi=300)
    plt.close()

    # Also save a small text summary for paper/table
    summary = {
        "n_qubits": n,
        "k": k,
        "mass_kept": mass_kept,
        "x_opt": x_opt,
        "E_opt_qubo": E_opt,
        "P_opt_postselected": P_opt,
        "cum_prob_topM_postselected": cum_top,
        "E_expected_postselected": E_exp,
        "E_gap_expected_minus_opt": gap,
        "topM_bits": top_bits,
        "topM_E": top_E,
        "topM_P_postselected": top_P,
    }
    with open(outdir / "fig14_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(" -", outdir / "fig14_topM_mass_vs_optimum.png")
    print(" -", outdir / "fig14_summary.json")
    print(f"Metrics: P_opt={P_opt:.6f}, cum_topM={cum_top:.6f}, E_exp={E_exp:.6f}, gap={gap:.6f}")

if __name__ == "__main__":
    main()
