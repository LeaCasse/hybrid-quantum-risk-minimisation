import json, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.load(open(args.inp))
    hist = d["hist"]
    n = d["n"]; k = d["k"]; mass_k = d["mass_k"]

    xs = list(range(n+1))
    ys = [hist.get(str(x), hist.get(x, 0.0)) for x in xs]  # robust keys

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Hamming weight |x|")
    plt.ylabel("Probability mass")
    plt.title(f"Mixer-only Hamming-weight distribution (k={k}), mass_k={mass_k:.4f}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
