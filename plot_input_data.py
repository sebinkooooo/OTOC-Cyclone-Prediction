# plot_era5_fields.py
# ---------------------------------------------------------
# Generate physical plots from ERA5 input:
#  - Temperature field at 700 hPa for each valid_time
#  - Temperature anomaly field (deviation from spatial mean)
#  - Cyclone track from min temperature anomaly
#  - Time series of min and mean temperature
# ---------------------------------------------------------

from pathlib import Path
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def ensure_dirs():
    out_dir = Path("plots/era5")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_era5(path="data/era5_cyclone_dikeledi_700hPa.nc"):
    ds = xr.open_dataset(path)
    # Temperature variable: t(valid_time, pressure_level, latitude, longitude)
    # Fix pressure dimension to 700 hPa index 0
    t = ds["t"].isel(pressure_level=0)  # shape: (time, lat, lon)
    times = ds["valid_time"].values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    return ds, t, times, lats, lons


def plot_temperature_snapshots(t, times, lats, lons, out_dir):
    """Save 2D T maps for each time."""
    for ti, time_val in enumerate(times):
        field = t.isel(valid_time=ti).values  # (lat, lon)

        fig, ax = plt.subplots(figsize=(6, 5))
        # note: imshow could be used, but contourf keeps lat/lon coordinates explicit
        cs = ax.contourf(lons, lats, field, levels=25, cmap="RdYlBu_r")
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("T (K)")

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"700 hPa Temperature\n{np.datetime_as_string(time_val, unit='h')}")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fname = out_dir / f"temperature_snapshot_{ti:02d}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved {fname}")


def plot_anomaly_snapshots_and_track(t, times, lats, lons, out_dir):
    """
    Compute anomaly = T - spatial_mean(T) per time.
    Track cyclone as the grid point with minimum anomaly.
    Save anomaly maps and return track + summary series.
    """
    center_lats = []
    center_lons = []
    min_t = []
    mean_t = []

    for ti, time_val in enumerate(times):
        field = t.isel(valid_time=ti).values  # (lat, lon)
        mean_val = field.mean()
        anomaly = field - mean_val

        # Cyclone "center": min anomaly (coldest relative region) or max if warm-core
        # Here we use min (adjust to max if that makes more physical sense).
        flat_idx = anomaly.argmin()
        lat_idx, lon_idx = np.unravel_index(flat_idx, anomaly.shape)
        clat = float(lats[lat_idx])
        clon = float(lons[lon_idx])

        center_lats.append(clat)
        center_lons.append(clon)
        min_t.append(field.min())
        mean_t.append(mean_val)

        # Plot anomaly map
        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(lons, lats, anomaly, levels=25, cmap="RdYlBu_r")
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("T anomaly (K)")

        # Mark center
        ax.plot(clon, clat, marker="x", markersize=8, linewidth=0, label="Cyclone center")

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(
            "700 hPa Temperature Anomaly\n"
            f"{np.datetime_as_string(time_val, unit='h')}"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fname = out_dir / f"anomaly_snapshot_{ti:02d}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"[PLOT] Saved {fname}")

    return np.array(center_lats), np.array(center_lons), np.array(min_t), np.array(mean_t)


def plot_cyclone_track(center_lats, center_lons, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(center_lons, center_lats, marker="o", linestyle="-", linewidth=2)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Cyclone Track (700 hPa temperature center)")
    ax.grid(True, alpha=0.3)

    # Mark start / end
    ax.scatter(center_lons[0], center_lats[0], marker="s", s=70, label="Start")
    ax.scatter(center_lons[-1], center_lats[-1], marker="*", s=100, label="End")
    ax.legend()

    fig.tight_layout()
    fname = out_dir / "cyclone_track.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {fname}")


def plot_temp_timeseries(times, min_t, mean_t, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, min_t, marker="o", linestyle="-", linewidth=2, label="Min T")
    ax.plot(times, mean_t, marker="s", linestyle="--", linewidth=2, label="Mean T")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("700 hPa Temperature Extremes vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fname = out_dir / "temperature_timeseries.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved {fname}")


def main():
    if not Path("data/era5_cyclone_dikeledi_700hPa.nc").exists():
        raise FileNotFoundError("data/era5_cyclone_dikeledi_700hPa.nc not found.")

    out_dir = ensure_dirs()
    ds, t, times, lats, lons = load_era5()

    # 1) Raw temperature snapshots
    plot_temperature_snapshots(t, times, lats, lons, out_dir)

    # 2) Anomaly snapshots + track statistics
    center_lats, center_lons, min_t, mean_t = plot_anomaly_snapshots_and_track(
        t, times, lats, lons, out_dir
    )

    # 3) Cyclone track
    plot_cyclone_track(center_lats, center_lons, out_dir)

    # 4) Time series of min/mean temperature
    plot_temp_timeseries(times, min_t, mean_t, out_dir)


if __name__ == "__main__":
    main()
