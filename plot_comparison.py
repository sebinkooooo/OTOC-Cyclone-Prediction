import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")


def load_json(path):
    return json.loads(Path(path).read_text())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_correlations(corr_dict, outdir):
    pearson = {k: v["pearson"] for k, v in corr_dict.items()}
    spearman = {k: v["spearman"] for k, v in corr_dict.items()}

    labels = list(pearson.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, pearson.values(), width, label="Pearson", color="tab:blue")
    ax.bar(x + width/2, spearman.values(), width, label="Spearman", color="tab:orange")

    ax.set_ylabel("Correlation")
    ax.set_title("Correlation Comparison (Pearson vs Spearman)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{outdir}/correlations.png", dpi=300)
    plt.close()


def plot_null_models(nulls, outdir):
    for key, values in nulls.items():
        plt.figure(figsize=(8, 4))
        plt.plot(values, marker="o")
        plt.title(f"Null Model Distribution: {key}")
        plt.xlabel("Index")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(f"{outdir}/{key}.png", dpi=300)
        plt.close()


def plot_pca_metrics(pca, outdir):
    mean = pca["pca_energy_mean"]
    std = pca["pca_energy_std"]

    plt.figure(figsize=(6, 4))
    plt.bar(["Energy Mean", "Energy Std"], [mean, std], color=["tab:green", "tab:red"])
    plt.title("PCA Energy Stability")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(f"{outdir}/pca_energy.png", dpi=300)
    plt.close()

    # correlation with variance proxy
    corr = pca["pca_energy_corr_variance"]
    plt.figure(figsize=(6, 4))
    plt.bar(["Pearson", "Spearman"],
            [corr["pearson"], corr["spearman"]],
            color=["tab:blue", "tab:orange"])
    plt.title("PCA Energy Correlation With Variance Proxy")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(f"{outdir}/pca_vs_variance.png", dpi=300)
    plt.close()


def plot_physical_gradients(phys, outdir):
    plt.figure(figsize=(10, 4))
    plt.plot(phys, marker="o", linewidth=2)
    plt.title("Physical Gradient Magnitude Over Time")
    plt.xlabel("Timestep index")
    plt.ylabel("|grad|")
    plt.tight_layout()
    plt.savefig(f"{outdir}/physical_gradients.png", dpi=300)
    plt.close()


def main():
    data = load_json("comparison.json")  # change if needed

    outdir = "plots/comp"
    ensure_dir(outdir)

    # 1. Correlations
    plot_correlations(data["correlations"], outdir)

    # 2. Null models
    null_out = f"{outdir}/nulls"
    ensure_dir(null_out)
    plot_null_models(data["null_models"], null_out)

    # 3. PCA metrics
    pca_out = f"{outdir}/pca"
    ensure_dir(pca_out)
    plot_pca_metrics(data["pca_metrics"], pca_out)

    # 4. Physical gradients
    plot_physical_gradients(data["physical_gradients"], outdir)

    print("✔ All plots saved under:", outdir)


if __name__ == "__main__":
    main()