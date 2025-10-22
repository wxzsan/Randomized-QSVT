import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import matplotlib.pyplot as plt
import time
import pickle
import os
import numba as nb
from scipy.special import zeta
# Step 1: Precompute the diagonal part of the Hamiltonian once
@nb.njit(parallel=True, fastmath=True)
def precompute_diagonal(n, J, alpha):
    dim = 2**n
    H_diag = np.zeros(dim, dtype=np.float64)
    
    # For each basis state s_idx, compute its corresponding diagonal energy
    for s_idx in nb.prange(dim):
        interaction_energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                s_i = (s_idx >> i) & 1
                s_j = (s_idx >> j) & 1
                
                # Z|0> = +1|0>, Z|1> = -1|1>
                # <s|Z_i|s> = (-1)^s_i
                sign = (-1.0)**s_i * (-1.0)**s_j
                
                distance = float(j - i)
                interaction_energy -= (J / (distance**alpha)) * sign
        
        H_diag[s_idx] = interaction_energy
    return H_diag

# Step 2: Highly efficient matvec with complexity O(n * 2^n)
@nb.njit(parallel=True, fastmath=True)
def hamiltonian_matvec_final(v_flat, n, h, H_diag):
    """
    Final optimized matvec: uses the precomputed diagonal.
    """
    dim = v_flat.shape[0]
    
    # Diagonal contribution: efficient element-wise multiplication
    Hv = H_diag * v_flat
    
    # Off-diagonal contribution (from transverse field)
    for s_idx in nb.prange(dim):
        transverse_val = 0.0
        for i in range(n):
            s_flipped_idx = s_idx ^ (1 << i)
            transverse_val += v_flat[s_flipped_idx]
        Hv[s_idx] -= h * transverse_val
            
    return Hv

# --- Main ---
if __name__ == "__main__":
    # --- Parameters ---
    J_val = 1.0
    h_val = 3.0
    alpha_val = 3.0
    n_values = range(4, 28, 2)
    gaps = []
    checkpoint_file = f"spectral_gap_final_opt_J={J_val}_h={h_val}_alpha={alpha_val}.pkl"
    
    # Checkpoint loading logic
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            gaps = checkpoint_data['gaps']
            completed_n_values = checkpoint_data['completed_n_values']
        print(f"Loaded {len(gaps)} completed calculations")
    else:
        completed_n_values = []
        print("No checkpoint found, starting from scratch")

    print("Starting simulation (Final Optimization: Pre-computation + Numba)...")
    print(f"Parameters: J={J_val}, h={h_val}, alpha={alpha_val}")

    print("theoretical sepctral gap: ", 2 * abs(h_val - J_val * zeta(alpha_val)))
    for n in n_values:
        if n in completed_n_values:
            print(f"Skipping N = {n} (already computed)")
            continue
        
        start_time = time.time()
        print(f"\nComputing system size N = {n}...")
        dim = 2**n
        
        # *** Final optimized workflow ***
        # 1. Precompute diagonal once
        print(f"  Pre-computing {dim}-element diagonal part...")
        H_diag_precomputed = precompute_diagonal(n, J_val, alpha_val)
        
        # 2. Create LinearOperator with precomputed diagonal
        H_op = LinearOperator(
            (dim, dim), 
            matvec=lambda v: hamiltonian_matvec_final(v, n, h_val, H_diag_precomputed),
            dtype=np.float64
        )
        
        print(f"  Diagonalizing with O(n*2^n) matvec...")
        
        # =======================================================================
        # *** Key adjustments ***
        # 1) Create a more physical initial vector v0: equal superposition (x+)
        v0 = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
        
        # 2) Increase ncv in eigsh: for k=2, raise from ~20 to 40
        eigenvalues = eigsh(
            H_op, 
            k=2, 
            which='SA', 
            return_eigenvectors=False,
            ncv=40,  
            v0=v0    
        )
        
        gap = np.abs(eigenvalues[1] - eigenvalues[0])
        print(f"  Ground state E0: {eigenvalues[0]:.6f}, First excited E1: {eigenvalues[1]:.6f}")
        gaps.append(gap)
        
        end_time = time.time()
        print(f"  N={n} computation completed. Spectral gap = {gap:.6f}. Time taken: {end_time - start_time:.2f} seconds.")
        
        # Checkpoint saving logic (unchanged)
        completed_n_values.append(n)
        checkpoint_data = {'gaps': gaps, 'completed_n_values': completed_n_values, 'parameters': {'J': J_val, 'h': h_val, 'alpha': alpha_val}}
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"  Checkpoint saved.")

    # --- Plotting and data organization ---
    plt.figure(figsize=(10, 6))
    plot_n_values = sorted(completed_n_values)
    sorted_gaps = [g for n, g in sorted(zip(completed_n_values, gaps))]
    plt.plot(plot_n_values, sorted_gaps, 'o-', label='Numerically computed spectral gap')
    
    # Add theoretical spectral gap as horizontal dotted line
    theoretical_gap = 2 * abs(h_val - J_val * zeta(alpha_val))
    plt.axhline(y=theoretical_gap, color='red', linestyle=':', linewidth=2, 
                label=f'Theoretical bound')
    
    plt.xlabel('System Size (n)', fontsize=20)
    plt.ylabel('Spectral Gap', fontsize=20)
    # plt.title(f'Spectral Gap vs. System Size (J={J_val}, h={h_val}, α={alpha_val})', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.xticks(plot_n_values, fontsize=18) 
    plt.yticks(fontsize=18)
    plt.ylim(bottom=0) 
    plot_filename = f'spectral_gap_vs_size_J={J_val}_h={h_val}_alpha={alpha_val}.png'
    plt.savefig(plot_filename, dpi=200, bbox_inches='tight')
    print(f"\nSimulation completed. Results plotted in '{plot_filename}'")
    print("\n--- Results Summary ---")
    for n, gap in zip(plot_n_values, sorted_gaps):
        print(f"N = {n}, Gap = {gap:.6f}")