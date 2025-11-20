import json
import numpy as np
import xarray as xr
from scipy.stats import pearsonr, spearmanr
import random
import os

###############################################
# LOAD DATA
###############################################

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

quantum = load_json("otoc_results.json")            # quantum OTOC + variance
pca_data = load_json("era5_processed.json")    # alpha, mu_grad, sigma_grad
era5 = xr.open_dataset("data/era5_cyclone_dikeledi_700hPa.nc")  # native fields
quantum_indices = np.array([entry["index"] for entry in quantum], dtype=int)


###############################################
# CONVERT TO NUMPY ARRAYS
###############################################

def extract_series(key):
    """Extract a list of values from quantum results"""
    return np.array([entry[key] for entry in quantum])

variance = extract_series("variance_proxy")
otoc = extract_series("otoc")
mu_all = np.array([p["mu_grad"] for p in pca_data])
sigma_all = np.array([p["sigma_grad"] for p in pca_data])
mu_grad = mu_all[quantum_indices]
sigma_grad = sigma_all[quantum_indices]


###############################################
# BASELINE PHYSICAL METRICS
###############################################

def compute_gradient_magnitude(ds):
    """
    Computes spatial mean gradient magnitude for each snapshot.
    This is a classical physical diagnostic.
    """
    temps = ds["t"][:, 0]  # shape: [time, lat, lon]
    grads = []
    for i in range(temps.shape[0]):
        t = temps[i].values
        gx, gy = np.gradient(t)
        mag = np.sqrt(gx**2 + gy**2)
        grads.append(mag.mean())
    return np.array(grads)

phys_grad = compute_gradient_magnitude(era5)
phys_grad_sel = phys_grad[quantum_indices]


###############################################
# NULL MODELS — RANDOM / SHUFFLED / NOISE
###############################################

def shuffle_time(series):
    return np.array(random.sample(list(series), len(series)))

def white_noise_like(series):
    return np.random.normal(0, np.std(series), size=len(series))

null_models = {
    "variance_time_shuffled": shuffle_time(variance),
    "otoc_time_shuffled": shuffle_time(otoc),
    "variance_noise": white_noise_like(variance),
    "otoc_noise": white_noise_like(otoc)
}


###############################################
# CORRELATIONS
###############################################

def corr(a, b):
    if np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(a, b)[0]),
        "spearman": float(spearmanr(a, b)[0])
    }

correlations = {
    "variance_vs_mu_grad": corr(variance, mu_grad),
    "variance_vs_sigma_grad": corr(variance, sigma_grad),
    "variance_vs_phys_grad": corr(variance, phys_grad_sel),
    "otoc_vs_mu_grad": corr(otoc, mu_grad),
    "otoc_vs_sigma_grad": corr(otoc, sigma_grad),
    "otoc_vs_phys_grad": corr(otoc, phys_grad_sel),

    # Against null models
    "variance_vs_shuffled": corr(variance, null_models["variance_time_shuffled"]),
    "variance_vs_noise": corr(variance, null_models["variance_noise"]),
    "otoc_vs_shuffled": corr(otoc, null_models["otoc_time_shuffled"]),
    "otoc_vs_noise": corr(otoc, null_models["otoc_noise"])
}


###############################################
# PCA ENERGY CAPTURE & SPREAD
###############################################

def pca_energy(alpha_vecs):
    """Total PCA energy (sum of coefficients)."""
    return np.array([sum(a) for a in alpha_vecs])

pca_energy_series = pca_energy([p["alpha"] for p in pca_data])
pca_energy_selected = pca_energy_series[quantum_indices]

pca_metrics = {
    "pca_energy_mean": float(pca_energy_series.mean()),
    "pca_energy_std": float(pca_energy_series.std()),
    "pca_energy_corr_variance": corr(pca_energy_selected, variance)
}


###############################################
# INTERPRETIVE SCORE
###############################################

def score_usefulness():
    """Determine if quantum metric beats baselines."""
    useful = {}

    # Variance "signal" must beat random noise
    if abs(correlations["variance_vs_phys_grad"]["pearson"]) > abs(correlations["variance_vs_noise"]["pearson"]):
        useful["variance_has_physical_signal"] = True
    else:
        useful["variance_has_physical_signal"] = False

    # OTOC significance check
    if abs(correlations["otoc_vs_phys_grad"]["pearson"]) > abs(correlations["otoc_vs_noise"]["pearson"]):
        useful["otoc_has_physical_signal"] = True
    else:
        useful["otoc_has_physical_signal"] = False

    return useful

usefulness = score_usefulness()


###############################################
# SAVE OUTPUT
###############################################

comparison = {
    "correlations": correlations,
    "null_models": {
        name: arr.tolist() for name, arr in null_models.items()
    },
    "pca_metrics": pca_metrics,
    "physical_gradients": phys_grad.tolist(),
    "usefulness_scores": usefulness,
}

with open("comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("✔ comparison.json generated!")
