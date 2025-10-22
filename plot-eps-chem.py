import pickle
import openfermion
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from complexity_functions import (
    our_method_complexity,
    qsvt_trotter_complexity,
    qetu_first_order_trotter_complexity,
    qetu_trotter_suzuki_complexity,
    qetu_qdrift_complexity,
    other_randomized_methods_complexity,
    calculate_pruned_params
)

def get_ordinal(n):
    """Converts a number to its ordinal string representation (e.g., 1 -> 1st, 4 -> 4th)."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th'][n % 10]
    return str(n) + suffix

# ==============================================================================
# Step 3: Main
# ==============================================================================
if __name__ == "__main__":
    molecules = [
        {'name': 'Propane', 'file': 'propane_hamiltonian.pkl', 'qubits': 46},
        {'name': 'Carbon Dioxide', 'file': 'co2_hamiltonian.pkl', 'qubits': 54},
        {'name': 'Ethane', 'file': 'ethane_hamiltonian.pkl', 'qubits': 60}
    ]
    
    DELTA_VAL = 0.25
    GAMMA_VAL = 0.01
    K_VAL = 2

    print("--- Parameter settings ---")
    print(f"Δ (Delta) = {DELTA_VAL}")
    print(f"γ (Overlap) = {GAMMA_VAL}")
    print(f"k = {K_VAL}\n")

    # *** 修改处：将 figsize 的高度从 8 改为 7 ***
    fig, axs = plt.subplots(1, len(molecules), figsize=(24, 5), sharey=True)
    
    styles = {
        "QETU w/ qDRIFT":           {'color': 'black', 'linestyle': '-', 'label': 'QETU withqDRIFT'},
        "QETU w/ 1st-order Trotter":{'color': 'blue', 'linestyle': '--', 'label': 'QETU with 1st-order Trotter'},
        "QETU w/ Trotter-Suzuki":   {'color': 'red', 'linestyle': '--', 'label': f'QETU with {get_ordinal(2*K_VAL)}-order Trotter'},
        "Randomized LCU":           {'color': 'red', 'linestyle': '-', 'label': 'Randomized LCU'},
        "QSVT w/ Trotter":          {'color': 'green', 'linestyle': '-.', 'label': f'QSVT with {get_ordinal(2*K_VAL)}-order Trotter'},
        "Our Method":               {'color': 'purple', 'linestyle': '-', 'label': 'Our Method'}
    }
    
    for i, molecule in enumerate(molecules):
        ax = axs[i]
        print(f"--- Processing: {molecule['name']} ---")
        
        try:
            with open(molecule['file'], 'rb') as f:
                original_hamiltonian = pickle.load(f)
            total_terms = len(original_hamiltonian.terms)
            print(f"Loaded successfully. Original total terms: {total_terms}")
        except FileNotFoundError:
            print(f"❌ Error: File '{molecule['file']}' not found.")
            continue
        except Exception as e:
            print(f"❌ Unknown error occurred: {e}")
            continue

        eps_range = np.logspace(-5, -1, 50)
        complexities = {
            "Our Method": [],
            "QSVT w/ Trotter": [],
            "QETU w/ 1st-order Trotter": [],
            "QETU w/ Trotter-Suzuki": [],
            "QETU w/ qDRIFT": [],
            "Randomized LCU": []
        }
        
        for eps in eps_range:
            pruning_threshold = eps * GAMMA_VAL * DELTA_VAL
            L, lambda_val, _ = calculate_pruned_params(original_hamiltonian, pruning_threshold)
            
            if L == 0:
                for key in complexities:
                    complexities[key].append(np.nan)
                continue
            
            complexities["Our Method"].append(our_method_complexity(eps, lambda_val, DELTA_VAL, GAMMA_VAL))
            complexities["QSVT w/ Trotter"].append(qsvt_trotter_complexity(eps, lambda_val, DELTA_VAL, GAMMA_VAL, K_VAL, L))
            complexities["QETU w/ 1st-order Trotter"].append(qetu_first_order_trotter_complexity(eps, lambda_val, DELTA_VAL, GAMMA_VAL, L))
            complexities["QETU w/ Trotter-Suzuki"].append(qetu_trotter_suzuki_complexity(eps, lambda_val, DELTA_VAL, GAMMA_VAL, K_VAL, L))
            complexities["QETU w/ qDRIFT"].append(qetu_qdrift_complexity(eps, lambda_val, DELTA_VAL, GAMMA_VAL))
            complexities["Randomized LCU"].append(other_randomized_methods_complexity(eps, lambda_val, DELTA_VAL, GAMMA_VAL))

        for name, data in complexities.items():
            style = styles[name]
            ax.plot(eps_range, data, color=style['color'], linestyle=style['linestyle'], label=style['label'])

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        title_text = f"{molecule['name']} - {molecule['qubits']} qubits"
        ax.set_title(title_text, fontsize=20, pad=10)
        ax.set_xlabel("Error Tolerance (ε)", fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.grid(True, which="both", ls="--", linewidth=0.5)  # 去掉网格线

    axs[0].set_ylabel("Asymptotic Gate Complexity", fontsize=16)
    
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc='center left', fontsize=16, 
                   bbox_to_anchor=(1.05, 0.5),
                   frameon=True, shadow=True, title="Methods", title_fontsize=18)
    
    fig.subplots_adjust(left=0.06, right=0.75, top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)
    
    plt.show()

    # --- Save (bbox_inches='tight' keeps all elements) ---
    fig.savefig(f"chem-eps-complexity_gamma={GAMMA_VAL}.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print("Plot complete.")