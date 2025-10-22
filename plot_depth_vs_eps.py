import pickle
import openfermion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# From our created module, import all necessary functions
from depth_functions import (
    our_method_depth,
    qsvt_trotter_depth,
    qetu_first_order_trotter_depth,
    qetu_trotter_suzuki_depth,
    qetu_qdrift_depth,
    calculate_pruned_params,
    get_ordinal
)

def calculate_identity_removed_params(hamiltonian):
    """Calculates lambda and L values after removing only the identity term without truncating others"""
    current_terms = hamiltonian.terms.copy()
    # Remove only the identity term ()
    if () in current_terms: 
        del current_terms[()]
    
    if not current_terms: 
        return 0, 0.0, 0.0
    
    # Calculate L (number of terms)
    L = len(current_terms)
    
    # Calculate lambda (sum of absolute values of all coefficients)
    lambda_val = sum(abs(coeff) for coeff in current_terms.values())
    
    # Calculate Lambda (absolute value of the largest coefficient)
    Lambda = max(abs(coeff) for coeff in current_terms.values()) if current_terms else 0.0
    
    return L, lambda_val, Lambda

# ==============================================================================
# Main Program
# ==============================================================================
if __name__ == "__main__":
    molecules = [
        {'name': 'Propane', 'file': 'propane_hamiltonian.pkl', 'qubits': 46},
        {'name': 'Carbon Dioxide', 'file': 'co2_hamiltonian.pkl', 'qubits': 54},
        {'name': 'Ethane', 'file': 'ethane_hamiltonian.pkl', 'qubits': 60}
    ]
    
    DELTA_VAL = 0.25
    GAMMA_VAL = 0.1
    K_VAL = 2 

    print("--- Parameter Settings ---")
    print(f"Δ (Delta) = {DELTA_VAL}")
    print(f"γ (Overlap) = {GAMMA_VAL}")
    print(f"k = {K_VAL}\n")
    
    # Output lambda and L values after only removing the identity term
    print("=== Hamiltonian Parameters (Identity Term Removed Only) ===")
    for molecule in molecules:
        try:
            with open(molecule['file'], 'rb') as f:
                hamiltonian = pickle.load(f)
            L, lambda_val, Lambda = calculate_identity_removed_params(hamiltonian)
            print(f"{molecule['name']}: L = {L}, λ = {lambda_val:.6f}, Λ = {Lambda:.6f}")
        except FileNotFoundError:
            print(f"{molecule['name']}: File not found")
        except Exception as e:
            print(f"{molecule['name']}: Error - {e}")
    print()

    # Keep a slightly taller canvas to provide ample space for layout
    fig, axs = plt.subplots(1, len(molecules), figsize=(24, 7.5), sharey=True)
    
    styles = {
        "QETU w/ qDRIFT":           {'color': 'black', 'linestyle': '-', 'label': 'QETU with qDRIFT'},
        "QETU w/ 1st-order Trotter":{'color': 'blue', 'linestyle': '--', 'label': 'QETU with 1st-order Trotter'},
        "QETU w/ Trotter-Suzuki":   {'color': 'red', 'linestyle': '--', 'label': f'QETU with {get_ordinal(2*K_VAL)}-order Trotter'},
        "QSVT w/ Trotter":          {'color': 'green', 'linestyle': '-.', 'label': f'QSVT with {get_ordinal(2*K_VAL)}-order Trotter'},
        "Our Method":               {'color': 'purple', 'linestyle': '-', 'label': 'Our Method'}
    }
    
    method_names = list(styles.keys())
    
    for i, molecule in enumerate(molecules):
        ax = axs[i]
        print(f"--- Processing: {molecule['name']} ---")
        
        try:
            with open(molecule['file'], 'rb') as f:
                original_hamiltonian = pickle.load(f)
            total_terms = len(original_hamiltonian.terms)
            print(f"Load successful. Original total number of terms: {total_terms}")
        except FileNotFoundError:
            print(f"❌ Error: File '{molecule['file']}' not found.")
            ax.text(0.5, 0.5, f"File not found:\n{molecule['file']}", ha='center', va='center', fontsize=16, color='red')
            ax.set_title(molecule['name'], fontsize=20, pad=10)
            continue
        except Exception as e:
            print(f"❌ An unknown error occurred: {e}")
            continue

        eps_range = np.logspace(-5, -1, 50)
        depths = {name: [] for name in method_names}
        
        for eps in eps_range:
            pruning_threshold = eps * GAMMA_VAL * DELTA_VAL
            L, lambda_val, _ = calculate_pruned_params(original_hamiltonian, pruning_threshold)
            
            if L == 0:
                for key in depths:
                    depths[key].append(np.nan)
                continue
            
            depths["Our Method"].append(our_method_depth(eps, lambda_val, DELTA_VAL, GAMMA_VAL))
            depths["QSVT w/ Trotter"].append(qsvt_trotter_depth(eps, lambda_val, DELTA_VAL, GAMMA_VAL, K_VAL, L))
            depths["QETU w/ 1st-order Trotter"].append(qetu_first_order_trotter_depth(eps, lambda_val, DELTA_VAL, GAMMA_VAL, L))
            depths["QETU w/ Trotter-Suzuki"].append(qetu_trotter_suzuki_depth(eps, lambda_val, DELTA_VAL, GAMMA_VAL, K_VAL, L))
            depths["QETU w/ qDRIFT"].append(qetu_qdrift_depth(eps, lambda_val, DELTA_VAL, GAMMA_VAL))

        for name, data in depths.items():
            style = styles[name]
            ax.plot(eps_range, data, color=style['color'], linestyle=style['linestyle'], label=style['label'])

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        title_text = f"{molecule['name']} - {molecule['qubits']} qubits"
        ax.set_title(title_text, fontsize=20, pad=10)
        ax.set_xlabel("Error Tolerance (ε)", fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=14)

    # *** CHANGE 1: Modified the Y-axis label as requested ***
    axs[0].set_ylabel("Circuit Depth", fontsize=16)
    
    handles, labels = axs[0].get_legend_handles_labels()

    # Legend position remains the same
    # ... other code ...

    # Legend position remains the same
    legend = fig.legend(handles, labels, 
            loc='upper center',
            bbox_to_anchor=(0.5, 1.04), 
            ncol=3,
            fontsize=16,
            frameon=True,
            shadow=True,
            title="Methods", 
            title_fontsize=18)
    
    legend._legend_box.sep = 15  # Add 5 points of separation
        
    # Keep the adjustment to make space for the legend
    fig.subplots_adjust(left=0.06, right=0.98, top=0.75, bottom=0.18, wspace=0.2)
    
    plt.show()

    fig.savefig(f"chem-eps-depth_gamma={GAMMA_VAL}.png", dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print("Plotting complete.")