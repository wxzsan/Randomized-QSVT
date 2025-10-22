# combined_depth_vs_n.py
# Combined depth-vs-n plots for long-range and hybrid models (shared legend and y-axis)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta

# Import required functions from our module
from depth_functions import (
    our_method_depth,
    qsvt_trotter_depth,
    qetu_trotter_suzuki_depth,
    qetu_qdrift_depth,
)

def lambda_comm_scaling(n, k, g_H, t, eps):
    return max(k * n ** (1 / (2 * k + 1)), np.log(n * g_H * t / eps)) * g_H

def lambda_comm_scaling_prime(n, k, g_H):
    return k * n ** (1 / (2 * k + 1)) * g_H

# ============================================================================== 
# Main - plot circuit depth vs number of qubits (n) - combined version
# ============================================================================== 
if __name__ == "__main__":
    # --- Fixed algorithm parameters ---
    EPSILON_FIXED = 1e-2  # fixed error tolerance
    GAMMA_VAL = 1e-1      # fixed initial overlap

    # --- Define two models ---
    models = [
        {
            'name': 'Long-Range Model',
            'title': 'Long-Range Model',
            'H_CONST': 3.0,
            'J_CONST': 1.0,
            'ALPHA_CONST': 3.0,
            'G_CONST': None,  # no g parameter for long-range model
            'n_range': np.logspace(3, 7, 50),  # from 1e3 to 1e7 qubits
            'L_scaling': lambda n: (n**2 + n) / 2,
            'lambda_scaling': lambda n, H, J, G, ZETA_ALPHA: (H + J * ZETA_ALPHA) * n,
            'g_H': lambda n, H, J, G, ZETA_ALPHA: H + 2 * J * ZETA_ALPHA,
            'DELTA_VAL': 3,  # fixed spectral gap
            'condition_check': None  # no constraint check for long-range model
        },
        {
            'name': 'Hybrid Model',
            'title': 'Hybrid Model',
            'H_CONST': 3.0,
            'J_CONST': 1.0,
            'ALPHA_CONST': 3.0,
            'G_CONST': 0.1,
            'n_range': np.logspace(3, 6, 50),  # from 1e3 to 1e6 qubits
            'L_scaling': lambda n: (n**2 + 3*n) / 2,
            'lambda_scaling': lambda n, H, J, G, ZETA_ALPHA: (H + J) * n + G * ZETA_ALPHA,
            'g_H': lambda n, H, J, G, ZETA_ALPHA: H + 2 * J + 2 * G * ZETA_ALPHA / n,
            'DELTA_VAL': None,  # computed
            'condition_check': lambda H, J, G, ZETA_ALPHA: G < abs(H - J) / (2 * ZETA_ALPHA)
        }
    ]

    print("--- Parameter settings ---")
    print(f"ε (epsilon) = {EPSILON_FIXED}")
    print(f"γ (Overlap) = {GAMMA_VAL}\n")

    # --- Create subplots ---
    fig, axs = plt.subplots(1, 2, figsize=(24, 9), sharey=True)
    
    # --- Define styles ---
    styles = {
        "QETU w/ qDRIFT":           {'color': 'black', 'linestyle': '-', 'label': 'QETU with qDRIFT'},
        "QETU w/ Trotter-Suzuki (k=1)": {'color': 'red', 'linestyle': '--', 'label': 'QETU with 2nd-order Trotter'},
        "QETU w/ Trotter-Suzuki (k=2)": {'color': 'orange', 'linestyle': '--', 'label': 'QETU with 4th-order Trotter'},
        "QSVT w/ Trotter (k=1)":    {'color': 'green', 'linestyle': '-.', 'label': 'QSVT with 2nd-order Trotter'},
        "QSVT w/ Trotter (k=2)":    {'color': 'cyan', 'linestyle': '-.', 'label': 'QSVT with 4th-order Trotter'},
        "Our Method":               {'color': 'purple', 'linestyle': '-', 'label': 'Our Method'}
    }

    method_names = list(styles.keys())

    for i, model in enumerate(models):
        ax = axs[i]
        print(f"--- Processing model: {model['name']} ---")
        
        # Compute model-specific parameters
        H_CONST = model['H_CONST']
        J_CONST = model['J_CONST']
        ALPHA_CONST = model['ALPHA_CONST']
        G_CONST = model['G_CONST']
        ZETA_ALPHA = zeta(ALPHA_CONST)
        
        # Compute spectral gap
        if model['DELTA_VAL'] is None:
            DELTA_VAL = abs(H_CONST - J_CONST)
        else:
            DELTA_VAL = model['DELTA_VAL']

        # Compute the total Hamiltonian evolution time T_VAL
        T_VAL = np.log(1 / EPSILON_FIXED) * np.log(1 / (EPSILON_FIXED * GAMMA_VAL)) / (GAMMA_VAL * DELTA_VAL)
        
        # Constraint check (hybrid only)
        if model['condition_check'] is not None:
            condition_val = abs(H_CONST - J_CONST) / (2 * ZETA_ALPHA)
            assert G_CONST < condition_val, f"Error: g={G_CONST} must be < {condition_val:.4f}"
        
        print(f"h = {H_CONST}, J = {J_CONST}, α = {ALPHA_CONST}")
        if G_CONST is not None:
            print(f"g = {G_CONST}")
        print(f"Δ (Gap) = {DELTA_VAL}")
        
        # Get n range
        n_range = model['n_range']
        
        # Initialize containers for results
        depths = {name: [] for name in method_names}
        
        print("--- Computing circuit depths over n... ---")
        for n in n_range:
            # Compute parameters for current n
            current_L = model['L_scaling'](n)
            current_lambda = model['lambda_scaling'](n, H_CONST, J_CONST, G_CONST, ZETA_ALPHA)
            current_g_H = model['g_H'](n, H_CONST, J_CONST, G_CONST, ZETA_ALPHA)
            current_lambda_comm_k1 = lambda_comm_scaling(n, 1, current_g_H, T_VAL, EPSILON_FIXED)
            current_lambda_comm_k2 = lambda_comm_scaling(n, 2, current_g_H, T_VAL, EPSILON_FIXED)
            current_lambda_comm_k1_prime = lambda_comm_scaling_prime(n, 1, current_g_H)
            current_lambda_comm_k2_prime = lambda_comm_scaling_prime(n, 2, current_g_H)

            
            # --- Compute depths ---
            depths["Our Method"].append(our_method_depth(EPSILON_FIXED, current_lambda, DELTA_VAL, GAMMA_VAL))
            depths["QSVT w/ Trotter (k=1)"].append(qsvt_trotter_depth(EPSILON_FIXED, current_lambda_comm_k1_prime, DELTA_VAL, GAMMA_VAL, 1, current_L))
            depths["QSVT w/ Trotter (k=2)"].append(qsvt_trotter_depth(EPSILON_FIXED, current_lambda_comm_k2_prime, DELTA_VAL, GAMMA_VAL, 2, current_L))
            depths["QETU w/ Trotter-Suzuki (k=1)"].append(qetu_trotter_suzuki_depth(EPSILON_FIXED, current_lambda_comm_k1, DELTA_VAL, GAMMA_VAL, 1, current_L))
            depths["QETU w/ Trotter-Suzuki (k=2)"].append(qetu_trotter_suzuki_depth(EPSILON_FIXED, current_lambda_comm_k2, DELTA_VAL, GAMMA_VAL, 2, current_L))
            depths["QETU w/ qDRIFT"].append(qetu_qdrift_depth(EPSILON_FIXED, current_lambda, DELTA_VAL, GAMMA_VAL))
        
        print("Computation finished.\n")

        # --- Plot ---
        for name, data in depths.items():
            style = styles[name]
            ax.plot(n_range, data, color=style['color'], linestyle=style['linestyle'], label=style['label'], linewidth=2.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of Qubits (n)", fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, which="major", axis="y", ls="--", linewidth=0.5)
        
        # Set title
        ax.set_title(model['title'], fontsize=24, pad=10)

    # Shared y-axis label
    axs[0].set_ylabel("Circuit Depth", fontsize=24)
    
    # Get legend handles and labels
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Create shared legend
    legend = fig.legend(handles, labels, 
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.06), 
                       ncol=3,
                       fontsize=18,
                       frameon=True,
                       shadow=True,
                       title="Methods", 
                       title_fontsize=20)
    
    legend._legend_box.sep = 10  # reduce spacing between legend items
    
    # Adjust layout to leave space for legend
    fig.subplots_adjust(left=0.06, right=0.98, top=0.80, bottom=0.18, wspace=0.1)
    
    # Save figure
    fig.savefig(f"combined_depth_vs_n_error={EPSILON_FIXED}_gamma={GAMMA_VAL}.png", 
                dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print("Plot complete. Figure saved.")
    plt.show()
