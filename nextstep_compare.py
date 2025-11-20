import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

def load_json(path):
    return json.loads(Path(path).read_text())

def safe_corr(x, y):
    if len(x) < 3:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x, y)[0]),
        "spearman": float(spearmanr(x, y)[0]),
    }

def compute_deltas(arr):
    """Return |arr[t+1] - arr[t]| for t=0..n-2."""
    return [abs(arr[i+1] - arr[i]) for i in range(len(arr)-1)]

def main():
    era = load_json("era5_processed.json")
    otoc = load_json("otoc_results.json")

    # match order
    era_sorted = sorted(era, key=lambda r: r["index"])
    otoc_sorted = sorted(otoc, key=lambda r: r["index"])
    otoc_indices = [r["index"] for r in otoc_sorted]

    mu_all = [e["mu_grad"] for e in era_sorted]
    sigma_all = [e["sigma_grad"] for e in era_sorted]
    phys_grad_all = [np.sqrt(e["mu_grad"]**2 + e["sigma_grad"]**2) for e in era_sorted]

    mu = [mu_all[idx] for idx in otoc_indices]
    sigma = [sigma_all[idx] for idx in otoc_indices]
    phys_grad = [phys_grad_all[idx] for idx in otoc_indices]
    variance = [r["variance_proxy"] for r in otoc_sorted]
    otoc_vals = [r["otoc"] for r in otoc_sorted]

    # Δ over time
    d_mu = compute_deltas(mu)
    d_sigma = compute_deltas(sigma)
    d_phys = compute_deltas(phys_grad)
    d_variance = compute_deltas(variance)
    d_otoc = compute_deltas(otoc_vals)

    # correlations OTOC(t) vs Δ at next timestep
    corr_otoc_phys_next = safe_corr(otoc_vals[:-1], d_phys)
    corr_otoc_mu_next = safe_corr(otoc_vals[:-1], d_mu)
    corr_otoc_sigma_next = safe_corr(otoc_vals[:-1], d_sigma)
    corr_otoc_var_next = safe_corr(otoc_vals[:-1], d_variance)

    # correlations ΔOTOC(t) vs ΔPhys(t)
    corr_delta_otoc_delta_phys = safe_corr(d_otoc, d_phys)

    # correlations ΔOTOC vs Δμ, Δσ
    corr_delta_otoc_delta_mu = safe_corr(d_otoc, d_mu)
    corr_delta_otoc_delta_sigma = safe_corr(d_otoc, d_sigma)

    out = {
        "next_step_correlations": {
            "OTOC_vs_next_phys_grad": corr_otoc_phys_next,
            "OTOC_vs_next_mu_grad": corr_otoc_mu_next,
            "OTOC_vs_next_sigma_grad": corr_otoc_sigma_next,
            "OTOC_vs_next_variance": corr_otoc_var_next,
        },
        "delta_correlations": {
            "delta_OTOC_vs_delta_phys": corr_delta_otoc_delta_phys,
            "delta_OTOC_vs_delta_mu": corr_delta_otoc_delta_mu,
            "delta_OTOC_vs_delta_sigma": corr_delta_otoc_delta_sigma,
        },
        "raw_arrays": {
            "mu": mu,
            "sigma": sigma,
            "phys_grad": phys_grad,
            "variance": variance,
            "otoc": otoc_vals,
            "d_mu": d_mu,
            "d_sigma": d_sigma,
            "d_phys": d_phys,
            "d_variance": d_variance,
            "d_otoc": d_otoc,
        }
    }

    Path("nextstep_comparison.json").write_text(json.dumps(out, indent=2))
    print("✔ Saved nextstep_comparison.json")

if __name__ == "__main__":
    main()
