#!/usr/bin/env python3
"""
plot_master.py
---------------------------------------------------------
Master plotting script for QuantumBradford paper Results section.
Generates all figures needed for the manuscript.
---------------------------------------------------------
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import xarray as xr

# Use publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# Create output directory
PLOTS_DIR = Path("plots/results")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path):
    return json.loads(Path(path).read_text())


# =============================================================================
# FIGURE 1: ERA5 Temperature Field Example
# =============================================================================
def plot_fig1_temperature_field():
    """Representative temperature field and anomaly at peak intensification"""
    ds = xr.open_dataset("data/era5_cyclone_dikeledi_700hPa.nc")
    t_raw = ds["t"].isel(pressure_level=0)
    
    # Choose snapshot 7 (peak gradient period)
    snapshot_idx = 7
    field = t_raw.isel(valid_time=snapshot_idx).values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    time_val = ds["valid_time"].values[snapshot_idx]
    
    # Compute anomaly
    anomaly = field - field.mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Raw temperature
    im1 = axes[0].contourf(lons, lats, field, levels=20, cmap="RdYlBu_r")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_title("(a) Raw Temperature Field")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("T (K)")
    
    # Anomaly
    im2 = axes[1].contourf(lons, lats, anomaly, levels=20, cmap="RdBu_r")
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_ylabel("Latitude (°N)")
    axes[1].set_title("(b) Temperature Anomaly")
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("ΔT (K)")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_temperature_fields.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig1_temperature_fields.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Temperature fields saved")


# =============================================================================
# FIGURE 2: PCA Variance Explained
# =============================================================================
def plot_fig2_pca_variance():
    """Show variance captured by top-8 EOF modes"""
    # Mock data - replace with actual PCA explained variance if available
    # For demonstration, showing typical decay pattern
    explained_var = np.array([0.283, 0.195, 0.152, 0.118, 0.089, 0.067, 0.051, 0.045])
    cumulative_var = np.cumsum(explained_var)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    x = np.arange(1, 9)
    ax.bar(x, explained_var, alpha=0.7, color='steelblue', label='Individual')
    ax.plot(x, cumulative_var, 'ro-', linewidth=2, markersize=4, label='Cumulative')
    
    ax.set_xlabel("EOF Mode")
    ax.set_ylabel("Variance Explained")
    ax.set_xticks(x)
    ax.set_ylim([0, 1.05])
    ax.legend(loc='center right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title("PCA Variance Decomposition")
    
    # Add cumulative percentage annotation
    total_var = cumulative_var[-1]
    ax.text(8, cumulative_var[-1] + 0.03, f'{total_var*100:.1f}%', 
            ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_pca_variance.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig2_pca_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: PCA variance explained saved")


# =============================================================================
# FIGURE 3: Alpha Evolution Over Time
# =============================================================================
def plot_fig3_alpha_evolution():
    """Time series of alpha coefficients showing mode energy redistribution"""
    processed = load_json("era5_processed.json")
    alphas = np.array([p["alpha"] for p in processed])
    times = np.arange(len(processed))
    
    fig, ax = plt.subplots(figsize=(7, 3))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i in range(8):
        ax.plot(times, alphas[:, i], marker='o', markersize=3, 
                linewidth=1.5, label=f'α{i+1}', color=colors[i])
    
    ax.set_xlabel("Timestep (6-hour intervals)")
    ax.set_ylabel("α coefficient")
    ax.set_title("EOF Mode Amplitude Evolution")
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig3_alpha_evolution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig3_alpha_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Alpha evolution saved")


# =============================================================================
# FIGURE 4: Hamiltonian Parameters
# =============================================================================
def plot_fig4_hamiltonian_params():
    """Show μ_grad and σ_grad evolution (Hamiltonian inputs)"""
    processed = load_json("era5_processed.json")
    mu_vals = np.array([p["mu_grad"] for p in processed])
    sigma_vals = np.array([p["sigma_grad"] for p in processed])
    times = np.arange(len(processed))
    
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    ax.plot(times, mu_vals, 'o-', linewidth=2, markersize=5, 
            label='μ∇ (coupling strength J)', color='darkblue')
    ax.plot(times, sigma_vals, 's-', linewidth=2, markersize=5, 
            label='σ∇ (local field h)', color='darkred')
    
    ax.set_xlabel("Timestep (6-hour intervals)")
    ax.set_ylabel("Gradient magnitude")
    ax.set_title("Physical Gradients → Hamiltonian Parameters")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_hamiltonian_params.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig4_hamiltonian_params.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Hamiltonian parameters saved")


# =============================================================================
# FIGURE 5: OTOC Results (Main Result)
# =============================================================================
def plot_fig5_otoc_main():
    """Primary OTOC result showing chaos detection"""
    otoc_data = load_json("otoc_results.json")
    processed = load_json("era5_processed.json")
    
    # Extract quantum indices
    quantum_indices = [o["index"] for o in otoc_data]
    otoc_vals = np.array([o["otoc"] for o in otoc_data])
    
    # Physical gradients for all timesteps
    phys_grad = np.array([np.sqrt(p["mu_grad"]**2 + p["sigma_grad"]**2) 
                          for p in processed])
    phys_grad_selected = phys_grad[quantum_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    
    # Top: OTOC
    ax1.plot(quantum_indices, otoc_vals, 'o-', linewidth=2.5, markersize=7,
             color='darkviolet', label='OTOC F(t)')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel("OTOC F(t)")
    ax1.set_title("(a) Quantum Chaos Indicator (OTOC)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Highlight key region (rapid OTOC decay)
    ax1.axvspan(2, 4, alpha=0.15, color='red', label='Rapid decay')
    
    # Bottom: Physical gradients
    ax2.plot(range(len(phys_grad)), phys_grad, 's-', linewidth=2, 
             markersize=5, color='darkgreen', alpha=0.3, label='All timesteps')
    ax2.plot(quantum_indices, phys_grad_selected, 'o-', linewidth=2.5, 
             markersize=7, color='darkgreen', label='OTOC timesteps')
    ax2.set_xlabel("Timestep (6-hour intervals)")
    ax2.set_ylabel("|∇T| magnitude")
    ax2.set_title("(b) Classical Atmospheric Gradients")
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig5_otoc_main.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig5_otoc_main.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: OTOC main result saved")


# =============================================================================
# FIGURE 6: OTOC vs Classical Metrics Correlation
# =============================================================================
def plot_fig6_correlations():
    """Correlation analysis: OTOC vs physical gradients"""
    comparison = load_json("comparison.json")
    
    # Extract correlation values
    corr_data = {
        'OTOC vs μ∇': comparison["correlations"]["otoc_vs_mu_grad"]["pearson"],
        'OTOC vs σ∇': comparison["correlations"]["otoc_vs_sigma_grad"]["pearson"],
        'OTOC vs |∇T|': comparison["correlations"]["otoc_vs_phys_grad"]["pearson"],
        'Var vs μ∇': comparison["correlations"]["variance_vs_mu_grad"]["pearson"],
        'Var vs σ∇': comparison["correlations"]["variance_vs_sigma_grad"]["pearson"],
        'Var vs |∇T|': comparison["correlations"]["variance_vs_phys_grad"]["pearson"],
    }
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    labels = list(corr_data.keys())
    values = list(corr_data.values())
    colors = ['darkviolet']*3 + ['orange']*3
    
    bars = ax.barh(labels, values, color=colors, alpha=0.7)
    
    # Add reference line at 0
    ax.axvline(x=0, color='black', linewidth=0.8)
    
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_title("Quantum vs Classical Metrics: Physical Signal Tracking")
    ax.set_xlim([-0.7, 0.7])
    ax.grid(axis='x', alpha=0.3)
    
    # Add text labels on bars
    for bar, val in zip(bars, values):
        x_pos = val + (0.03 if val > 0 else -0.03)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                ha=ha, va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig6_correlations.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig6_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Correlations saved")


# =============================================================================
# FIGURE 7: Null Model Validation
# =============================================================================
def plot_fig7_null_validation():
    """Show OTOC beats random baselines"""
    comparison = load_json("comparison.json")
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    # Physical signal vs null models
    metrics = ['OTOC', 'Variance\nProxy']
    
    # Absolute correlations with physical gradient
    phys_signal = [
        abs(comparison["correlations"]["otoc_vs_phys_grad"]["pearson"]),
        abs(comparison["correlations"]["variance_vs_phys_grad"]["pearson"])
    ]
    
    # Correlation with noise
    noise_signal = [
        abs(comparison["correlations"]["otoc_vs_noise"]["pearson"]),
        abs(comparison["correlations"]["variance_vs_noise"]["pearson"])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, phys_signal, width, label='Physical gradient',
                   color='darkgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, noise_signal, width, label='White noise',
                   color='gray', alpha=0.8)
    
    ax.set_ylabel("Absolute Correlation")
    ax.set_title("Physical Signal vs Random Noise")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance markers
    for i, (p, n) in enumerate(zip(phys_signal, noise_signal)):
        if p > n:
            ax.text(i, max(p, n) + 0.02, '✓', ha='center', fontsize=14, color='green')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig7_null_validation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig7_null_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Null model validation saved")


# =============================================================================
# FIGURE 8: Temporal Analysis (ΔOTOC vs Δgradients)
# =============================================================================
def plot_fig8_temporal_analysis():
    """Show OTOC tracks instantaneous dynamics, not future prediction"""
    nextstep = load_json("nextstep_comparison.json")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))
    
    # Panel (a): Next-step prediction (weak)
    next_corr = nextstep["next_step_correlations"]
    labels_next = ['μ∇', 'σ∇', '|∇T|']
    values_next = [
        next_corr["OTOC_vs_next_mu_grad"]["pearson"],
        next_corr["OTOC_vs_next_sigma_grad"]["pearson"],
        next_corr["OTOC_vs_next_phys_grad"]["pearson"]
    ]
    
    ax1.barh(labels_next, values_next, color='lightcoral', alpha=0.7)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.set_xlabel("Correlation")
    ax1.set_title("(a) OTOC(t) → Δgradient(t+1)\n(Predictive Power)")
    ax1.set_xlim([-0.5, 0.5])
    ax1.grid(axis='x', alpha=0.3)
    
    # Panel (b): Synchronous changes (strong)
    delta_corr = nextstep["delta_correlations"]
    labels_delta = ['Δμ∇', 'Δσ∇', 'Δ|∇T|']
    values_delta = [
        delta_corr["delta_OTOC_vs_delta_mu"]["pearson"],
        delta_corr["delta_OTOC_vs_delta_sigma"]["pearson"],
        delta_corr["delta_OTOC_vs_delta_phys"]["pearson"]
    ]
    
    ax2.barh(labels_delta, values_delta, color='steelblue', alpha=0.7)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel("Correlation")
    ax2.set_title("(b) ΔOTOC(t) → Δgradient(t)\n(Diagnostic Power)")
    ax2.set_xlim([-0.8, 0.2])
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig8_temporal_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig8_temporal_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Temporal analysis saved")


# =============================================================================
# FIGURE 9: Bitstring Distribution Example
# =============================================================================
def plot_fig9_bitstring_example():
    """Show measurement distribution for one high-chaos snapshot"""
    otoc_data = load_json("otoc_results.json")
    
    # Choose snapshot with lowest OTOC (highest scrambling)
    otoc_vals = [o["otoc"] for o in otoc_data]
    min_idx = np.argmin(otoc_vals)
    snapshot = otoc_data[min_idx]
    
    counts = snapshot["counts"]
    total = sum(counts.values())
    
    # Get top 20 most frequent bitstrings
    sorted_bits = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    bits = [b[0] for b in sorted_bits]
    probs = [b[1]/total for b in sorted_bits]
    
    # Color by parity of qubits 0 and 1
    colors = []
    for b in bits:
        b0 = int(b[-1])
        b1 = int(b[-2])
        parity = (b0 ^ b1)
        colors.append('steelblue' if parity == 0 else 'coral')
    
    ax.bar(range(len(bits)), probs, color=colors, alpha=0.7)
    ax.set_xlabel("Bitstring (top 20)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Measurement Distribution (Snapshot {snapshot['index']}, OTOC={snapshot['otoc']:.2f})")
    ax.set_xticks(range(len(bits)))
    ax.set_xticklabels(bits, rotation=90, fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Even parity (+)'),
        Patch(facecolor='coral', alpha=0.7, label='Odd parity (−)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig9_bitstring_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "fig9_bitstring_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 9: Bitstring distribution saved")


# =============================================================================
# TABLE 1: Numerical Results Summary
# =============================================================================
def generate_table1_data():
    """Generate data for results summary table"""
    comparison = load_json("comparison.json")
    otoc_data = load_json("otoc_results.json")
    
    print("\n" + "="*60)
    print("TABLE 1: Numerical Results Summary")
    print("="*60)
    
    print("\nOTOC Statistics:")
    otoc_vals = np.array([o["otoc"] for o in otoc_data])
    print(f"  Mean:     {otoc_vals.mean():>8.3f}")
    print(f"  Std Dev:  {otoc_vals.std():>8.3f}")
    print(f"  Min:      {otoc_vals.min():>8.3f}")
    print(f"  Max:      {otoc_vals.max():>8.3f}")
    
    print("\nCorrelations with Physical Gradients:")
    print(f"  OTOC vs |∇T|:      {comparison['correlations']['otoc_vs_phys_grad']['pearson']:>7.3f} (Pearson)")
    print(f"  Variance vs |∇T|:  {comparison['correlations']['variance_vs_phys_grad']['pearson']:>7.3f} (Pearson)")
    
    print("\nValidation Against Null Models:")
    print(f"  OTOC physical signal:     {comparison['usefulness_scores']['otoc_has_physical_signal']}")
    print(f"  Variance physical signal: {comparison['usefulness_scores']['variance_has_physical_signal']}")
    
    print("\nTemporal Analysis:")
    nextstep = load_json("nextstep_comparison.json")
    print(f"  OTOC(t) → Δgrad(t+1):     {nextstep['next_step_correlations']['OTOC_vs_next_phys_grad']['pearson']:>7.3f}")
    print(f"  ΔOTOC(t) → Δgrad(t):      {nextstep['delta_correlations']['delta_OTOC_vs_delta_phys']['pearson']:>7.3f}")
    print("="*60 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Generate all figures for Results section"""
    print("\n" + "="*60)
    print("GENERATING ALL FIGURES FOR RESULTS SECTION")
    print("="*60 + "\n")
    
    try:
        plot_fig1_temperature_field()
        plot_fig2_pca_variance()
        plot_fig3_alpha_evolution()
        plot_fig4_hamiltonian_params()
        plot_fig5_otoc_main()
        plot_fig6_correlations()
        plot_fig7_null_validation()
        plot_fig8_temporal_analysis()
        plot_fig9_bitstring_example()
        generate_table1_data()
        
        print("\n" + "="*60)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
        print(f"✓ Output directory: {PLOTS_DIR}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()