# quantum_pipeline.py
# ---------------------------------------------------------
# Uses era5_processed.json to build:
#  - state preparation circuits from alpha(t)
#  - Ising Hamiltonian time evolution circuits U(t)
#  - OTOC echo circuit (V = X0, W = X1)
#  - variance proxy as debugging chaos indicator
# Updated for qnexus SDK 0.31+ API
# ---------------------------------------------------------

import json
from math import asin, sqrt
from pathlib import Path

from dotenv import load_dotenv
import os
load_dotenv()

import qnexus as qnx
from qnexus.client.auth import login_with_credentials

import numpy as np
from pytket import Circuit


N_MODES = 8  # 8 EOFs → 8 qubits
_QNEXUS_LOGGED_IN = False


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def load_processed(path: str = "era5_processed.json"):
    data = json.loads(Path(path).read_text())
    return data


def variance_proxy(alpha: list[float]) -> float:
    """
    Variance(α) = (1/8) * Σ_k (α_k - ᾱ)^2
    """
    arr = np.array(alpha, dtype=float)
    mean = arr.mean()
    var = ((arr - mean) ** 2).mean()
    return float(var)


# ---------------------------------------------------------
# STATE PREPARATION
# ---------------------------------------------------------
def build_state_prep(alpha: list[float]) -> Circuit:
    """
    Prepare 8-qubit product state:

      |q_k> = sqrt(alpha_k) |0> + sqrt(1 - alpha_k) |1>

    implemented as Ry(θ_k) on |0>, with:
      sin(θ_k / 2) = sqrt(alpha_k)
    """
    n = len(alpha)
    circ = Circuit(n)

    for k, ak in enumerate(alpha):
        ak_clamped = min(max(ak, 0.0), 1.0)
        theta = 2.0 * asin(sqrt(ak_clamped))
        circ.Ry(theta, k)

    return circ


# ---------------------------------------------------------
# HAMILTONIAN: Ising with transverse X field
# ---------------------------------------------------------
def add_ising_trotter_layer(
    circ: Circuit,
    mu_grad: float,
    sigma_grad: float,
    delta_t: float,
    beta: float = 1.0,
    gamma: float = 1.0,
):
    """
    One first-order Trotter layer for:

      H(t) = Σ_i h_i X_i + Σ_i J Z_i Z_{i+1}

    with:
      J_i,i+1(t) = gamma * mu_grad(t)
      h_i(t)     = beta  * sigma_grad(t) * (0.5 + i/16)

    Implemented as:
      - ZZ part by CX-Rz-CX (up to overall phase)
      - X part by Rx rotations
    """
    n = N_MODES

    # ZZ couplings
    J = gamma * mu_grad
    theta_zz = 2.0 * J * delta_t  # from e^{-i J Z Z δt}

    for i in range(n - 1):
        # e^{-i J Z_i Z_{i+1} δt} ≈ CX-Rz-CX (global phase ignored)
        circ.CX(i, i + 1)
        circ.Rz(theta_zz, i + 1)
        circ.CX(i, i + 1)

    # X local fields
    for i in range(n):
        h_i = beta * sigma_grad * (0.5 + i / 16.0)
        phi_x = 2.0 * h_i * delta_t  # e^{-i h_i X δt} = Rx(2 h_i δt)
        circ.Rx(phi_x, i)


def build_time_evolution_circuit(
    mu_grad: float,
    sigma_grad: float,
    total_time: float = 1.0,
    n_trotter: int = 4,
) -> Circuit:
    """
    Build U(t) ≈ [U_ZZ U_X]^{n_trotter}
    """
    delta_t = total_time / n_trotter
    circ = Circuit(N_MODES)

    for _ in range(n_trotter):
        add_ising_trotter_layer(circ, mu_grad, sigma_grad, delta_t)

    return circ


# ---------------------------------------------------------
# OTOC ECHO CIRCUIT
# ---------------------------------------------------------
def build_otoc_echo_circuit(
    alpha: list[float],
    mu_grad: float,
    sigma_grad: float,
    total_time: float = 1.0,
    n_trotter: int = 4,
) -> Circuit:
    """
    Build the OTOC echo circuit:

      1. Prepare |ψ>
      2. Apply V = X on qubit 0
      3. Apply U(t)
      4. Apply W = X on qubit 1
      5. Apply U†(t)
      6. Apply V again (X on qubit 0)
      7. Apply U(t)
      8. Apply W again (X on qubit 1)
      9. Measure all qubits

    We then post-process bitstring counts to compute:

      F(t) = Σ_b P(b) (-1)^{b0 ⊕ b1}
    """
    # State preparation
    prep = build_state_prep(alpha)

    # Time evolution U and its inverse
    U = build_time_evolution_circuit(mu_grad, sigma_grad, total_time, n_trotter)
    U_dag = U.dagger()

    circ = Circuit(N_MODES)
    circ.append(prep)

    # V at t=0: X on qubit 0
    circ.X(0)

    # Forward evolution
    circ.append(U)

    # W at time t: X on qubit 1
    circ.X(1)

    # Backward evolution
    circ.append(U_dag)

    # V again
    circ.X(0)

    # Forward again
    circ.append(U)

    # W again
    circ.X(1)

    # Measure all qubits into fresh classical bits
    circ.measure_all()

    return circ


def ensure_qnexus_login():
    """
    Log into Nexus once per run, preferring username/password env vars.
    Updated for qnexus 0.31+ API.
    """
    global _QNEXUS_LOGGED_IN
    if _QNEXUS_LOGGED_IN:
        return

    username = os.getenv("QNEXUS_USERNAME")
    password = os.getenv("QNEXUS_PASSWORD")

    if username and password:
        print(f"[INFO] Attempting non-interactive qnexus login for {username}...")
        try:
            # Use newer login method with credentials
            login_with_credentials()
            _QNEXUS_LOGGED_IN = True
            print("[INFO] Login successful!")
            return
        except Exception as err:
            print(f"[WARN] Credential login failed: {err}")
            print("[INFO] Falling back to browser login...")

    # Fallback to browser-based login
    print("[INFO] Launching browser login for qnexus…")
    qnx.login()
    _QNEXUS_LOGGED_IN = True


# ---------------------------------------------------------
# QUANTINUUM BACKEND
# ---------------------------------------------------------
def run_on_quantinuum(
    circ: Circuit,
    shots: int = 1000,
    backend_name: str = "H2-Emulator",
    machine_debug: bool = False,
):
    """
    Run circuit on Quantinuum backend via qnexus.
    Updated for qnexus 0.31+ API - no QASM conversion needed.
    """
    ensure_qnexus_login()

    # Get or create project
    proj = os.getenv("QNEXUS_PROJECT", "default")
    print(f"[INFO] Using qnexus project: {proj}")
    project_obj = qnx.projects.get_or_create(name=proj)
    
    # Set as active project for this session
    qnx.context.set_active_project(project_obj)

    try:
        # --- Upload circuit directly (no QASM conversion) ---
        print("[INFO] Uploading circuit to qnexus...")
        circ_ref = qnx.circuits.upload(
            circuit=circ,  # Direct Circuit upload
            name="otoc_circuit",
            project=project_obj
        )
        print(f"[INFO] Circuit uploaded: {circ_ref.id}")

        # --- Backend configuration ---
        config = qnx.QuantinuumConfig(device_name=backend_name, shots=shots)

        # --- Compile ---
        print("[INFO] Starting compilation job...")
        compile_job = qnx.start_compile_job(
            circuits=[circ_ref],
            backend_config=config,
            optimisation_level=2,
            name="compile_otoc"
        )
        
        print("[INFO] Waiting for compilation to complete...")
        qnx.jobs.wait_for(compile_job)
        
        # Check compilation status
        compile_status = qnx.jobs.status(compile_job)
        print(f"[INFO] Compilation status: {compile_status}")
        
        compiled_ref = qnx.jobs.results(compile_job)[0].get_output()

        # --- Execute ---
        print("[INFO] Starting execution job...")
        exec_job = qnx.start_execute_job(
            circuits=[compiled_ref],
            n_shots=[shots],
            backend_config=config,
            name="exec_otoc"
        )
        
        print("[INFO] Waiting for execution to complete...")
        qnx.jobs.wait_for(exec_job)
        
        # Check execution status
        exec_status = qnx.jobs.status(exec_job)
        print(f"[INFO] Execution status: {exec_status}")

        # --- Result retrieval ---
        print("[INFO] Retrieving results...")
        exec_res = qnx.jobs.results(exec_job)[0]
        result_obj = exec_res.download_result()

        # Process results - try pytket method first, fallback to raw
        counts = {}
        try:
            if hasattr(result_obj, 'get_counts'):
                counts = result_obj.get_counts()
                print(f"[INFO] Retrieved {len(counts)} unique bitstrings via get_counts()")
            else:
                # Fallback to raw format parsing
                shots_list = result_obj.get("shots", [])
                for shot in shots_list:
                    bit = shot.get("bitstring")
                    if bit is not None:
                        counts[bit] = counts.get(bit, 0) + 1
                print(f"[INFO] Retrieved {len(counts)} unique bitstrings via raw format")
        except Exception as e:
            print(f"[ERROR] Failed to extract counts: {e}")
            print(f"[DEBUG] Result object type: {type(result_obj)}")
            print(f"[DEBUG] Result object: {result_obj}")

        if not counts:
            print("[WARN] No measurement counts returned; downstream OTOC will report NaN.")

        return counts

    except Exception as e:
        print(f"[ERROR] Quantinuum execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ---------------------------------------------------------
# OTOC FROM COUNTS
# ---------------------------------------------------------
def compute_otoc_from_counts(counts: dict) -> float:
    """
    Given bitstring counts, compute:

      F(t) = Σ_b P(b) (-1)^{b0 ⊕ b1}

    NOTE: tket uses bitstrings with qubit 0 as the *rightmost* bit
    in the string (little-endian). If results look weird, try swapping
    indices.
    """
    if not counts:
        print("[WARN] No measurement counts returned; cannot compute OTOC.")
        return float("nan")

    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    F = 0.0
    for bitstring, cnt in counts.items():
        p = cnt / total_shots

        # Take last 2 bits as qubits 0 and 1 (little-endian)
        # Example: "00000011" -> q1q0 = '1','1'
        b0 = int(bitstring[-1])
        b1 = int(bitstring[-2])
        parity = (b0 ^ b1)  # XOR
        F += p * ((-1) ** parity)

    return float(F)


# ---------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------
def main(
    processed_json: str = "era5_processed.json",
    n_snapshots: int = 3,
    do_otoc: bool = False,
):
    """
    High-level pipeline:

      - Load processed ERA5 data
      - For first n_snapshots:
          * print variance proxy
          * build U(t) and echo circuit
          * (optionally) run OTOC on Quantinuum
    """
    data = load_processed(processed_json)
    print(f"[Q] Loaded {len(data)} timesteps from {processed_json}")

    selected = data[:n_snapshots]

    for rec in selected:
        idx = rec["index"]
        t_str = rec["time"]
        alpha = rec["alpha"]
        mu = rec["mu_grad"]
        sigma = rec["sigma_grad"]

        var = variance_proxy(alpha)
        print(f"\n[Q] t={idx} ({t_str})")
        print(f"    variance proxy: {var:.6e}")
        print(f"    mu_grad:        {mu:.6e}")
        print(f"    sigma_grad:     {sigma:.6e}")

        # Build circuits
        state_circ = build_state_prep(alpha)
        U_circ = build_time_evolution_circuit(mu, sigma)
        echo_circ = build_otoc_echo_circuit(alpha, mu, sigma)

        print(f"    state depth: {state_circ.depth()}, U depth: {U_circ.depth()}, echo depth: {echo_circ.depth()}")

        if do_otoc:
            print("    [Q] Running OTOC echo on Quantinuum backend...")
            # For real OTOC, use H2-1 (hardware) or H2-Emulator (emulator)
            counts = run_on_quantinuum(echo_circ, shots=1000, backend_name="H2-Emulator", machine_debug=False)
            F_t = compute_otoc_from_counts(counts)
            print(f"    OTOC F(t) ≈ {F_t:.6f}")

            save_otoc_record({
                "index": idx,
                "timestamp": t_str,
                "alpha": alpha,
                "mu_grad": mu,
                "sigma_grad": sigma,
                "variance_proxy": var,
                "otoc": F_t,
                "counts": counts,
            }, path="otoc_results.json")
def save_otoc_record(record, path="otoc_results.json"):
    counts = record.get("counts")
    if counts:
        serializable_counts = {}
        for key, value in counts.items():
            if isinstance(key, tuple):
                key_str = "".join(str(bit) for bit in key)
            else:
                key_str = str(key)
            serializable_counts[key_str] = value
        record = {**record, "counts": serializable_counts}

    if os.path.exists(path):
        prev = json.loads(Path(path).read_text())
    else:
        prev = []

    prev.append(record)
    Path(path).write_text(json.dumps(prev, indent=2))
    print(f"[INFO] Saved OTOC record → {path}")


if __name__ == "__main__":
    main(n_snapshots=6, do_otoc=True)
