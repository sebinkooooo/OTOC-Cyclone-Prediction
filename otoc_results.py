# plot_otoc_results.py
# ---------------------------------------------------------
# Plot OTOC results saved by quantum_pipeline.py into:
#   - plots/otoc/otoc_vs_time.png
#   - plots/otoc/variance_vs_otoc.png
#   - plots/otoc/entropy_vs_time.png
#   - plots/otoc/hamming_weight_snapshot_<idx>.png
#   - plots/otoc/bitstring_heatmap_top32.png
# ---------------------------------------------------------

import json
import os
from pathlib import Path
from math import log

import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots") / "otoc"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_otoc_results(path: str = "otoc_results.json"):
    data = json.loads(Path(path).read_text())
    # sort by index just to be safe
    data = sorted(data, key=lambda r: r["index"])
    return data


def compute_entropy(counts: dict) -> float:
    """Shannon entropy (base 2) of a bitstring distribution."""
    if not counts:
        return float("nan")
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * log(p, 2)
    return H


def hamming_weight(bitstring: str) -> int:
    return bitstring.count("1")


def plot_otoc_vs_time(records):
    idxs = [r["index"] for r in records]
    F = [r.get("otoc", float("nan")) for r in records]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(idxs, F, marker="o", linestyle="-", linewidth=1.8)
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel(r"$F(t)$ (OTOC)")
    ax.set_title("OTOC vs time snapshot")
    ax.grid(True, alpha=0.3)

    # optional: annotate with first/last timestamps
    t0 = records[0]["timestamp"]
    t1 = records[-1]["timestamp"]
    ax.text(
        0.99,
        0.01,
        f"{t0} → {t1}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
    )

    out = PLOTS_DIR / "otoc_vs_time.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_variance_vs_otoc(records):
    vars_ = [r["variance_proxy"] for r in records]
    F = [r.get("otoc", float("nan")) for r in records]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(vars_, F, s=40)

    # label each point by snapshot index to see evolution
    for v, f, r in zip(vars_, F, records):
        ax.text(v, f, str(r["index"]), fontsize=7, alpha=0.7)

    ax.set_xlabel(r"Variance proxy  $\mathrm{Var}(\alpha)$")
    ax.set_ylabel(r"$F(t)$ (OTOC)")
    ax.set_title("Correlation between initial structure and scrambling")
    ax.grid(True, alpha=0.3)

    out = PLOTS_DIR / "variance_vs_otoc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_entropy_vs_time(records):
    idxs = [r["index"] for r in records]
    entropies = [compute_entropy(r.get("counts", {})) for r in records]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(idxs, entropies, marker="s", linestyle="-", linewidth=1.8)
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel(r"Shannon entropy $S$ (bits)")
    ax.set_title("Measurement entropy vs time snapshot")
    ax.grid(True, alpha=0.3)

    out = PLOTS_DIR / "entropy_vs_time.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def plot_hamming_weight_histograms(records):
    """One histogram per snapshot: distribution of Hamming weights."""
    for r in records:
        counts = r.get("counts", {})
        if not counts:
            continue

        weights = []
        freq = []
        for bit, c in counts.items():
            weights.append(hamming_weight(bit))
            freq.append(c)

        weights = np.array(weights)
        freq = np.array(freq)
        if freq.sum() == 0:
            continue

        # Convert to probability per weight
        total = freq.sum()
        max_w = max(weights)
        probs = np.zeros(max_w + 1)
        for w, f in zip(weights, freq):
            probs[w] += f / total

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(np.arange(max_w + 1), probs, width=0.8)
        ax.set_xlabel("Hamming weight (number of 1s)")
        ax.set_ylabel("Probability")
        ax.set_title(f"Hamming weight distribution (snapshot {r['index']})")
        ax.set_xticks(np.arange(max_w + 1))
        ax.grid(True, axis="y", alpha=0.3)

        out = PLOTS_DIR / f"hamming_weight_snapshot_{r['index']}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved {out}")


def plot_bitstring_heatmap(records, top_k: int = 32):
    """
    Build a matrix P[time, bitstring] for the top_k most frequent bitstrings
    across all snapshots, and show as a heatmap.
    """
    # Aggregate counts across all snapshots
    agg = {}
    for r in records:
        for bit, c in r.get("counts", {}).items():
            agg[bit] = agg.get(bit, 0) + c

    if not agg:
        print("[WARN] No counts in records; skipping bitstring heatmap.")
        return

    # pick top_k bitstrings
    sorted_bits = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    top_bits = [b for b, _ in sorted_bits[:top_k]]

    # Build probability matrix: rows = snapshots, cols = bitstrings
    mat = np.zeros((len(records), len(top_bits)))
    for ti, r in enumerate(records):
        counts = r.get("counts", {})
        total = sum(counts.values()) or 1
        for bi, bit in enumerate(top_bits):
            mat[ti, bi] = counts.get(bit, 0) / total

    fig, ax = plt.subplots(figsize=(min(10, 0.3 * len(top_bits) + 3), 4))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")

    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Snapshot index")
    ax.set_yticks(np.arange(len(records)))
    ax.set_yticklabels([r["index"] for r in records])

    # Show only some x-ticks to avoid clutter
    step = max(1, len(top_bits) // 16)
    xticks = np.arange(0, len(top_bits), step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([top_bits[i] for i in xticks], rotation=90, fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(bitstring)")

    ax.set_title("Top bitstring statistics across snapshots")

    out = PLOTS_DIR / "bitstring_heatmap_top32.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


def main():
    if not Path("otoc_results.json").exists():
        raise FileNotFoundError("otoc_results.json not found in current directory.")

    records = load_otoc_results("otoc_results.json")

    plot_otoc_vs_time(records)
    plot_variance_vs_otoc(records)
    plot_entropy_vs_time(records)
    plot_hamming_weight_histograms(records)
    plot_bitstring_heatmap(records, top_k=32)


if __name__ == "__main__":
    main()