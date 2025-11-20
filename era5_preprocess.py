# era5_preprocess.py
# ---------------------------------------------------------
# ERA5 → anomalies → PCA → α(t) → gradients → μ∇(t), σ∇(t)
# ---------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# CONFIG — modify only if your file is in a different place
# ---------------------------------------------------------
ERA5_FILE = "data/era5_cyclone_dikeledi_700hPa.nc"

# Mozambique Channel BOX used in papers:
LAT_MIN = -20.0
LAT_MAX = -10.0
LON_MIN = 40.0
LON_MAX = 50.0

N_MODES = 8   # PCA / EOF modes => number of qubits


# ---------------------------------------------------------
# LOAD ERA5 TEMPERATURE FIELD
# ---------------------------------------------------------
def load_era5_temperature(path: str, var_name: str = "t") -> xr.DataArray:
    """
    Load ERA5 temperature 700hPa from NetCDF and extract spatial box.
    Expected dims: (time, level, latitude, longitude)
    """
    ds = xr.open_dataset(path)

    # Find temperature variable name if "t" doesn't exist
    if var_name not in ds:
        # search for any variable containing temperature
        for k in ds.data_vars:
            if "temp" in k.lower():
                var_name = k
                break

    da = ds[var_name]

    # Subset 700 hPa level (if present)
    level_dim = next((dim for dim in da.dims if "level" in dim), None)
    if level_dim is not None:
        da = da.sel({level_dim: 700})

    # Subset Mozambique Channel
    da = da.sel(
        latitude=slice(LAT_MAX, LAT_MIN),  # ERA5 lat DESCENDING
        longitude=slice(LON_MIN, LON_MAX)
    )

    # Remove length-1 dims (e.g., pressure_level) and normalize time naming
    da = da.squeeze(drop=True)
    if "time" not in da.dims:
        time_dim = next((dim for dim in da.dims if "time" in dim), None)
        if time_dim and time_dim != "time":
            da = da.rename({time_dim: "time"})

    print("[ERA5] Loaded:", da.shape, "dims:", da.dims)
    return da


# ---------------------------------------------------------
# DETREND → anomalies
# ---------------------------------------------------------
def detrend_spatial(da: xr.DataArray) -> xr.DataArray:
    """
    Remove spatial mean at each time → anomalies.
    """
    spatial_mean = da.mean(dim=("latitude", "longitude"))
    return da - spatial_mean


# ---------------------------------------------------------
# FLATTEN 2D → 1D
# ---------------------------------------------------------
def flatten_to_matrix(anom: xr.DataArray) -> np.ndarray:
    """
    Convert (time, lat, lon) to (time, Npoints)
    """
    arr = anom.values  # shape: (time, lat, lon)
    t = arr.shape[0]
    return arr.reshape(t, -1)


# ---------------------------------------------------------
# PCA BASIS
# ---------------------------------------------------------
def compute_pca_basis(X: np.ndarray, n_components: int):
    pca = PCA(n_components=n_components)
    pca.fit(X)

    V = pca.components_.T  # shape: (D, n_modes)
    print(f"[PCA] Explained variance: {pca.explained_variance_ratio_}")
    print(f"[PCA] Total retained variance = {pca.explained_variance_ratio_.sum():.4f}")

    return pca, V


# ---------------------------------------------------------
# PROJECTION → PCA coefficients
# ---------------------------------------------------------
def project_onto_pca(X: np.ndarray, V: np.ndarray):
    return X @ V   # (time, n_modes)


# ---------------------------------------------------------
# α(t)_k from c(t)_k
# ---------------------------------------------------------
def compute_alphas(C: np.ndarray) -> np.ndarray:
    """
    α(t)_k = (c_k)^2 / Σ_j (c_j)^2
    """
    A = C**2
    denom = A.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1e-12, denom)
    return A / denom


# ---------------------------------------------------------
# GRADIENTS → μ∇(t), σ∇(t)
# ---------------------------------------------------------
def compute_gradients(anom: xr.DataArray):
    T = anom.values  # shape (time, lat, lon)
    time_len, ny, nx = T.shape

    dTdx = np.zeros_like(T)
    dTdy = np.zeros_like(T)

    # x-gradients
    dTdx[:, :, 1:-1] = T[:, :, 2:] - T[:, :, :-2]
    dTdx[:, :, 0] = T[:, :, 1] - T[:, :, 0]
    dTdx[:, :, -1] = T[:, :, -1] - T[:, :, -2]

    # y-gradients
    dTdy[:, 1:-1, :] = T[:, 2:, :] - T[:, :-2, :]
    dTdy[:, 0, :] = T[:, 1, :] - T[:, 0, :]
    dTdy[:, -1, :] = T[:, -1, :] - T[:, -2, :]

    grad_mag = np.sqrt(dTdx**2 + dTdy**2)

    mu_grad = grad_mag.mean(axis=(1, 2))
    sigma_grad = grad_mag.std(axis=(1, 2))
    return mu_grad, sigma_grad


# ---------------------------------------------------------
# BUILD JSON
# ---------------------------------------------------------
def build_summary(alpha, mu_grad, sigma_grad, times):
    out = []
    for i in range(len(times)):
        out.append({
            "index": int(i),
            "time": str(times[i]),
            "alpha": alpha[i].tolist(),
            "mu_grad": float(mu_grad[i]),
            "sigma_grad": float(sigma_grad[i]),
        })
    return out


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print(f"[LOAD] Using {ERA5_FILE}")
    da = load_era5_temperature(ERA5_FILE)

    print("[STEP] Detrending...")
    anom = detrend_spatial(da)

    print("[STEP] Flattening...")
    X = flatten_to_matrix(anom)

    print("[STEP] PCA...")
    _, V = compute_pca_basis(X, N_MODES)

    C = project_onto_pca(X, V)

    print("[STEP] Computing α(t)...")
    alpha = compute_alphas(C)

    print("[STEP] Gradients...")
    mu_grad, sigma_grad = compute_gradients(anom)

    times = da["time"].values
    summary = build_summary(alpha, mu_grad, sigma_grad, times)

    out_path = Path("era5_processed.json")
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[SAVED] {len(summary)} timesteps → era5_processed.json")
    print("[DONE]")


if __name__ == "__main__":
    main()
