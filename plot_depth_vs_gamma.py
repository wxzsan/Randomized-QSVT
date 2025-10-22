import pickle
import openfermion
import numpy as np
import matplotlib.pyplot as plt

# Import required functions from our module
from depth_functions import (
    our_method_depth,
    qsvt_trotter_depth,
    qetu_first_order_trotter_depth,
    qetu_trotter_suzuki_depth,
    qetu_qdrift_depth,
    calculate_pruned_params,
    get_ordinal
)

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    # --- Define molecules ---
    molecules = [
        {'name': 'Propane', 'file': 'propane_hamiltonian.pkl', 'qubits': 46},
        {'name': 'Carbon Dioxide', 'file': 'co2_hamiltonian.pkl', 'qubits': 54},
        {'name': 'Ethane', 'file': 'ethane_hamiltonian.pkl', 'qubits': 60}
    ]
    
    # --- Fixed parameters ---
    EPSILON_FIXED = 0.0001
    DELTA_VAL = 0.25
    K_VAL = 2

    print("--- Parameter settings ---")
    print(f"ε (eps) = {EPSILON_FIXED} (fixed accuracy)")
    print(f"Δ (Delta) = {DELTA_VAL}")
    print(f"k = {K_VAL} (Trotter Order)\n")

    # --- Create a 1x3 subplot layout ---
    fig, axs = plt.subplots(1, len(molecules), figsize=(24, 5), sharey=True)
    
    # --- Plot styles ---
    styles = {
        "QETU w/ qDRIFT":           {'color': 'black', 'linestyle': '-', 'label': 'QETU with qDRIFT'},
        "QETU w/ 1st-order Trotter":{'color': 'blue', 'linestyle': '--', 'label': 'QETU with 1st-order Trotter'},
        "QETU w/ Trotter-Suzuki":   {'color': 'red', 'linestyle': '--', 'label': f'QETU with {get_ordinal(2*K_VAL)}-order Trotter'},
        "QSVT w/ Trotter":          {'color': 'green', 'linestyle': '-.', 'label': f'QSVT with {get_ordinal(2*K_VAL)}-order Trotter'},
        "Our Method":               {'color': 'purple', 'linestyle': '-', 'label': 'Our Method'}
    }
    
    method_names = list(styles.keys())

    # --- Iterate molecules ---
    for i, molecule in enumerate(molecules):
        ax = axs[i]
        print(f"--- Processing: {molecule['name']} ---")
        
        try:
            with open(molecule['file'], 'rb') as f:
                hamiltonian = pickle.load(f)
            print(f"Loaded successfully. Original total terms: {len(hamiltonian.terms)}")
        except FileNotFoundError:
            print(f"❌ Error: File '{molecule['file']}' not found.")
            ax.text(0.5, 0.5, f"File not found:\n{molecule['file']}", ha='center', va='center', fontsize=16, color='red')
            ax.set_title(molecule['name'], fontsize=20, pad=10)
            continue
        except Exception as e:
            print(f"❌ Unknown error occurred: {e}")
            continue

        # --- Use overlap (gamma) as x-axis ---
        overlap_range = np.logspace(-3, -1, 50)
        
        depths = {name: [] for name in method_names}
        
        for overlap in overlap_range:
            # Dynamic pruning threshold
            pruning_threshold = EPSILON_FIXED * overlap * DELTA_VAL
            L, lambda_val, _ = calculate_pruned_params(hamiltonian, pruning_threshold)
            
            if L == 0:
                for key in depths:
                    depths[key].append(np.nan)
                continue

            # Compute new depth functions
            depths["Our Method"].append(our_method_depth(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma=overlap))
            depths["QSVT w/ Trotter"].append(qsvt_trotter_depth(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma=overlap, k=K_VAL, L=L))
            depths["QETU w/ 1st-order Trotter"].append(qetu_first_order_trotter_depth(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma=overlap, L=L))
            depths["QETU w/ Trotter-Suzuki"].append(qetu_trotter_suzuki_depth(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma=overlap, k=K_VAL, L=L))
            depths["QETU w/ qDRIFT"].append(qetu_qdrift_depth(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma=overlap))

        for name, data in depths.items():
            style = styles[name]
            ax.plot(overlap_range, data, color=style['color'], linestyle=style['linestyle'], label=style['label'])

        ax.set_xscale('log')
        ax.set_yscale('log')
        
        title_text = f"{molecule['name']} - {molecule['qubits']} qubits"
        ax.set_title(title_text, fontsize=20, pad=10)
        ax.set_xlabel("Initial State Overlap (γ)", fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.grid(True, which="both", ls="--", linewidth=0.5)  # 去掉网格线

    # --- Shared Y label ---
    axs[0].set_ylabel("Circuit Depth", fontsize=16)
    
    # --- Shared legend ---
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc='center left', fontsize=16, 
                   bbox_to_anchor=(1.05, 0.5),
                   frameon=True, shadow=True, title="Methods", title_fontsize=18)
    
    # --- Layout ---
    fig.subplots_adjust(left=0.06, right=0.75, top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)
    
    # --- Save file ---
    # Create a descriptive filename
    filename = f"chem-gamma-depth_error={EPSILON_FIXED}.png"
    # Use bbox_inches='tight' to keep all elements
    fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nFigure saved as: {filename}")
    
    plt.show()
    
    print("Plot complete.")