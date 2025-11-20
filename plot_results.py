import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.ticker as ticker

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
otoc_file = "otoc_results.json"
processed_file = "era5_processed.json"
era5_nc = "data/era5_cyclone_dikeledi_700hPa.nc"

otoc_data = json.loads(Path(otoc_file).read_text())
processed = json.loads(Path(processed_file).read_text())
ds = xr.open_dataset(era5_nc)

# Ensure plots/ folder exists
os.makedirs("plots", exist_ok=True)


# ----------------------------------------------------
# Helper: time arrays
# ----------------------------------------------------
otoc_times = [x["timestamp"] for x in otoc_data]
otoc_idx = list(range(len(otoc_times)))
proc_idx = list(range(len(processed)))

# ----------------------------------------------------
# Extract arrays
# ----------------------------------------------------
otoc_vals = np.array([x["otoc"] for x in otoc_data])
var_vals = np.array([x["variance_proxy"] for x in otoc_data])
mu_vals  = np.array([x["mu_grad"] for x in otoc_data])
sg_vals  = np.array([x["sigma_grad"] for x in otoc_data])

alphas = np.array([x["alpha"] for x in processed])  # [T, 8]


# ----------------------------------------------------
# Plot 1 — OTOC vs Time
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(otoc_idx, otoc_vals, marker="o", linewidth=2)
plt.title("OTOC vs Time (Quantum Chaos Indicator)")
plt.xlabel("Snapshot Index")
plt.ylabel("OTOC F(t)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/otoc_vs_time.png", dpi=300)
plt.close()


# ----------------------------------------------------
# Plot 2 — Classical VarianceProxy vs Time
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(otoc_idx, var_vals, marker="o", color="orange", linewidth=2)
plt.title("Classical Variance Proxy vs Time")
plt.xlabel("Snapshot Index")
plt.ylabel("Var(α)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/variance_vs_time.png", dpi=300)
plt.close()


# ----------------------------------------------------
# Plot 3 — OTOC vs Variance (scatter)
# ----------------------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(var_vals, otoc_vals, s=80, c=otoc_idx, cmap="viridis")
plt.colorbar(label="Time Index")
plt.xlabel("Variance Proxy")
plt.ylabel("OTOC F(t)")
plt.title("OTOC vs Classical Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/otoc_vs_variance.png", dpi=300)
plt.close()


# ----------------------------------------------------
# Plot 4 — Alpha time series
# ----------------------------------------------------
plt.figure(figsize=(10,6))
for i in range(alphas.shape[1]):
    plt.plot(proc_idx, alphas[:, i], marker="o", label=f"α{i+1}")

plt.title("EOF Mode Amplitudes α₁…α₈ Over Time")
plt.xlabel("Snapshot Index")
plt.ylabel("α Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/alpha_timeseries.png", dpi=300)
plt.close()


# ----------------------------------------------------
# Plot 5 — μ_grad and σ_grad evolution
# ----------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(otoc_idx, mu_vals, marker="o", linewidth=2, label="μ_grad (coupling)")
plt.plot(otoc_idx, sg_vals, marker="o", linewidth=2, label="σ_grad (turbulence)")
plt.title("Hamiltonian Parameters Over Time")
plt.xlabel("Snapshot Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/grads_timeseries.png", dpi=300)
plt.close()


# ----------------------------------------------------
# Plot 6 — Temperature heatmap per snapshot
# ----------------------------------------------------
temps = ds["t"]  # shape: (time, 1, lat, lon)

for i in range(len(temps.valid_time)):
    field = temps[i, 0, :, :]  # select pressure level 700hPa

    plt.figure(figsize=(6,5))
    plt.imshow(field, origin="lower", cmap="coolwarm")
    plt.colorbar(label="Temp (K)")
    plt.title(f"ERA5 Cyclone Temperature — Snapshot {i}")
    plt.tight_layout()
    plt.savefig(f"plots/temp_map_{i}.png", dpi=300)
    plt.close()


# ----------------------------------------------------
# Plot 7 — Alpha barplot per snapshot
# ----------------------------------------------------
for i, snap in enumerate(processed):
    plt.figure(figsize=(6,5))
    plt.bar(range(1,9), snap["alpha"])
    plt.xticks(range(1,9))
    plt.ylabel("α Value")
    plt.xlabel("EOF Mode")
    plt.title(f"EOF Amplitudes (Snapshot {i})")
    plt.tight_layout()
    plt.savefig(f"plots/alpha_bars_{i}.png", dpi=300)
    plt.close()

print("🎉 All plots saved into: plots/")
