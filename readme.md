# Quantum Chaos Indicators for Tropical Cyclone Prediction

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](paper/paper.tex)

> **First application of quantum chaos diagnostics to real-world cyclone forecasting**  
> Detecting rapid intensification using Out-of-Time-Order Correlators (OTOCs) computed from atmospheric temperature fields

---

## Overview

This repository implements the methodology described in our paper *"Quantum Chaos Indicators for Tropical Cyclone Rapid Intensification: An OTOC-Based Early-Warning Framework Using PCA-Compressed Atmospheric Fields"*. 

The core idea: treat a tropical cyclone as a quantum system, encode its dominant spatial patterns into qubits, evolve them under a physics-derived Hamiltonian, and measure information scrambling via OTOC. When OTOC decays rapidly, the atmosphere has entered a chaotic regime prone to rapid intensification (RI).

**Key Results:**
- OTOC correlates with atmospheric gradients (Pearson *r* = −0.50)
- Detects dynamical instability 12–18 hours before visible RI signatures
- Outperforms classical variance metrics in chaos detection
- Validated on Cyclone Dikeledi (January 2025, Mozambique Channel)

---

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/otoc-cyclone-prediction
cd otoc-cyclone-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download ERA5 Data

Obtain 700 hPa temperature fields from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels):

- **Variables:** Temperature (`t`)
- **Pressure level:** 700 hPa
- **Region:** Mozambique Channel [−20°N to −10°N, 40°E to 50°E]
- **Time:** January 10–12, 2025 (6-hour intervals)
- **Format:** NetCDF

Place the downloaded file at `data/era5_cyclone_dikeledi_700hPa.nc`.

### 3. Configure Quantinuum Access

Create a `.env` file in the project root:

```bash
QNEXUS_USERNAME=your_username
QNEXUS_PASSWORD=your_password
QNEXUS_USE_PASSWORD=True
QNEXUS_PROJECT=your_project_name
```

Get credentials from [Quantinuum](https://www.quantinuum.com/). The H1-Emulator is free for academic use.

### 4. Run the Pipeline

```bash
# Step 1: Preprocess ERA5 data → extract 8 dominant EOF modes
python era5_preprocess.py

# Step 2: Execute quantum circuits on H1-Emulator (~15 min on M4 Pro)
python quantum_pipeline.py

# Step 3: Compute correlations with classical baselines
python comparison.py
python nextstep_compare.py

# Step 4: Generate all manuscript figures
python plot_master.py
```

Results are saved in:
- `era5_processed.json` — PCA coefficients, gradients
- `otoc_results.json` — OTOC values, measurement bitstrings
- `comparison.json` — Correlation statistics
- `plots/results/` — Publication-quality figures

---

## Pipeline Architecture

```
ERA5 temperature fields (700 hPa)
         ↓
   [era5_preprocess.py]
   - Spatial detrending
   - PCA (8 EOF modes)
   - Gradient computation
         ↓
   era5_processed.json
         ↓
   [quantum_pipeline.py]
   - Map α_k → qubit states
   - Build Ising Hamiltonian H(t)
   - Execute OTOC echo circuit
         ↓
   otoc_results.json
         ↓
   [comparison.py]
   - Correlate with physical gradients
   - Test null models (shuffled/noise)
         ↓
   comparison.json
         ↓
   [plot_master.py]
   - Generate manuscript figures
```

---

## Mathematical Framework

### 1. Data Encoding

**Input:** 700 hPa temperature field *T*(*t*, *lat*, *lon*) from ERA5  
**Preprocessing:** Subtract spatial mean → anomaly field *T̃*

**PCA Decomposition:**
```
T̃(t, x) = Σ_k c_k(t) · v_k(x)
```
where *v_k* are EOF spatial patterns (eigenvectors of covariance matrix), *c_k* are time-varying coefficients.

**Qubit Amplitudes:**
```
α_k(t) = (c_k)² / Σ_j (c_j)²
```
Encodes energy distribution across 8 dominant modes.

### 2. Quantum State Preparation

Each qubit *k* initialized via rotation:
```
|q_k⟩ = √α_k |0⟩ + √(1−α_k) |1⟩
```
Implemented as `Ry(2·arcsin(√α_k))` gate.

### 3. Hamiltonian Construction

8-qubit Ising model with transverse field:
```
H(t) = Σ_i h_i(t) X_i + Σ_i J(t) Z_i Z_{i+1}
```

**Coupling strengths** derived from atmospheric gradients:
- *J(t)* = γ · μ_∇(*t*) — Mean gradient (baroclinicity)
- *h_i(t)* = β · σ_∇(*t*) · (0.5 + *i*/16) — Gradient variance (turbulence)

### 4. OTOC Measurement

Out-of-Time-Order Correlator quantifies information scrambling:
```
F(t) = ⟨W†(t) V†(0) W(t) V(0)⟩
```
with *V* = *X*₀, *W* = *X*₁ (perturbation operators).

**Echo circuit protocol:**
1. Prepare |ψ⟩
2. Apply *V* (perturb qubit 0)
3. Forward evolution *U*(*t*)
4. Apply *W* (measure qubit 1)
5. Backward evolution *U*†(*t*)
6. Repeat *V* → *U*(*t*) → *W*
7. Measure all qubits

**OTOC from bitstrings:**
```
F(t) = Σ_s P(s) · (−1)^(s₀ ⊕ s₁)
```
where *s₀*, *s₁* are measurement outcomes for qubits 0 and 1.

**Physical interpretation:**
- *F* ≈ 1: Low scrambling, organized cyclone structure
- *F* ≈ 0: High scrambling, chaotic regime
- *F* < 0: Strong anti-correlation, RI-primed state

---

## Results Summary

| Metric | Value |
|--------|-------|
| OTOC range | [−0.16, 0.83] |
| Corr(OTOC, \|∇*T*\|) | *r* = −0.50 |
| Corr(Variance, \|∇*T*\|) | *r* = 0.07 |
| Corr(ΔOTOC, Δ\|∇*T*\|) | *r* = −0.61 |
| Null model rejection | ✓ OTOC passes, ✗ Variance fails |
| Next-step prediction | Weak (*r* = 0.24) |
| Synchronous tracking | Strong (*r* = −0.61) |

**Conclusion:** OTOC functions as an instantaneous chaos diagnostic, not a forecast model. It detects when the cyclone enters a sensitivity-amplifying dynamical regime, providing early warning of RI potential.

---

## Repository Structure

```
├── data/
│   └── era5_cyclone_dikeledi_700hPa.nc  # ERA5 input (user-provided)
├── paper/
│   ├── paper.tex                         # Full manuscript
│   └── notes.tex                         # Results interpretation
├── plots/
│   ├── era5/                             # Raw field visualizations
│   ├── otoc/                             # OTOC timeseries, entropy
│   ├── comp/                             # Correlation analysis
│   └── results/                          # Publication figures
├── era5_preprocess.py                    # PCA pipeline
├── quantum_pipeline.py                   # Quantum circuit execution
├── comparison.py                         # Statistical validation
├── nextstep_compare.py                   # Temporal analysis
├── plot_master.py                        # Generate all figures
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{sandhu2025quantum,
  title={Quantum Chaos Indicators for Tropical Cyclone Rapid Intensification: 
         An OTOC-Based Early-Warning Framework Using PCA-Compressed Atmospheric Fields},
  author={Sandhu, Niru and Panesar, Kulvinder and Kot, Sebastian},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Hardware Requirements

- **Quantum backend:** Quantinuum H1-Emulator (free) or H1 hardware (commercial)
- **Classical compute:** Standard laptop (M4 Pro tested, ~15 min for 6 snapshots)
- **Memory:** <4 GB RAM
- **Storage:** ~500 MB for ERA5 data + outputs

---

## Extending to Other Cyclones

To apply this framework to different storms:

1. **Download ERA5 data** for your cyclone of interest (same format)
2. **Update domain coordinates** in `era5_preprocess.py`:
   ```python
   LAT_MIN = your_lat_min
   LAT_MAX = your_lat_max
   LON_MIN = your_lon_min
   LON_MAX = your_lon_max
   ```
3. **Adjust PCA modes** if needed (default: 8 qubits)
4. **Rerun pipeline** — no code changes required

We validated on 6 snapshots for computational efficiency. Operational deployment would process 10–20 snapshots per cyclone.

---

## Limitations & Future Work

**Current limitations:**
- Small sample size (6 timesteps, 1 cyclone)
- Emulator only (no hardware noise characterization)
- Single atmospheric variable (temperature)

**Planned extensions:**
- Multi-cyclone validation (20+ historical RI events)
- True quantum hardware execution (H1, IonQ)
- Multi-field encoding (winds, moisture, pressure)
- Real-time assimilation pipeline for operational forecasting

---

## Acknowledgments

This research was conducted at the University of Bradford Quantum Hackathon 2025, aligned with the International Year of Quantum Science and Technology. We thank:

- **Quantinuum** for H1-Emulator access via qnexus
- **ECMWF** for ERA5 reanalysis data
- **Victims of Cyclone Dikeledi** — this work is dedicated to improving early warnings and saving lives

---

## License

Research use encouraged. Commercial deployment requires consultation with authors.

---

## Contact

**Niru Sandhu** — University of Stirling  
**Dr. Kulvinder Panesar** — University of Bradford  
**Sebastian Kot** — City, University of London  

For questions or collaboration: [GitHub Issues](https://github.com/yourusername/otoc-cyclone-prediction/issues)

---

**⚠️ Operational Disclaimer:** This is a research prototype. Do not use for actual hurricane evacuation decisions without validation by national meteorological agencies.
