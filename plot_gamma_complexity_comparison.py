import pickle
import openfermion
import numpy as np
import matplotlib.pyplot as plt
from complexity_functions import (
    our_method_complexity,
    other_randomized_methods_complexity
)

def calculate_identity_removed_params(hamiltonian):
    """Compute L and lambda after removing only the identity term (no truncation)."""
    current_terms = hamiltonian.terms.copy()
    # Only remove the identity term
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
# Main - Plot complexity vs gamma (comparison)
# ==============================================================================
if __name__ == "__main__":
    # --- Define molecules ---
    molecules = [
        {'name': 'Propane', 'file': 'propane_hamiltonian.pkl', 'qubits': 46},
        {'name': 'Carbon Dioxide', 'file': 'co2_hamiltonian.pkl', 'qubits': 54},
        {'name': 'Ethane', 'file': 'ethane_hamiltonian.pkl', 'qubits': 60}
    ]
    
    # --- Fixed parameters ---
    EPSILON_FIXED = 0.01
    DELTA_VAL = 0.25
    
    print("--- Parameter settings ---")
    print(f"ε (eps) = {EPSILON_FIXED} (fixed accuracy)")
    print(f"Δ (Delta) = {DELTA_VAL}\n")
    
    # Output L and lambda after removing only the identity term
    print("=== Hamiltonian parameters with identity term removed only ===")
    molecule_params = {}
    for molecule in molecules:
        try:
            with open(molecule['file'], 'rb') as f:
                hamiltonian = pickle.load(f)
            L, lambda_val, Lambda = calculate_identity_removed_params(hamiltonian)
            molecule_params[molecule['name']] = {'L': L, 'lambda': lambda_val, 'Lambda': Lambda}
            print(f"{molecule['name']}: L = {L}, λ = {lambda_val:.6f}, Λ = {Lambda:.6f}")
        except FileNotFoundError:
            print(f"{molecule['name']}: File not found")
        except Exception as e:
            print(f"{molecule['name']}: Error - {e}")
    print()
    
    # Use a larger canvas similar to the first script
    fig, axs = plt.subplots(1, len(molecules), figsize=(24, 7.5), sharey=True)
    
    # --- Plot styles ---
    styles = {
        "Our Method": {'color': 'purple', 'linestyle': '-', 'label': 'Our Method', 'linewidth': 2.5},
        "Other Randomized Method": {'color': 'red', 'linestyle': '-', 'label': 'Other Randomized Methods', 'linewidth': 2.5}
    }
    
    # --- Iterate molecules ---
    for i, molecule in enumerate(molecules):
        ax = axs[i]
        print(f"--- Processing: {molecule['name']} ---")
        
        if molecule['name'] not in molecule_params:
            ax.text(0.5, 0.5, f"File not found:\n{molecule['file']}", ha='center', va='center', fontsize=16, color='red')
            ax.set_title(molecule['name'], fontsize=20, pad=10)
            continue
        
        # Retrieve parameters for this molecule
        L = molecule_params[molecule['name']]['L']
        lambda_val = molecule_params[molecule['name']]['lambda']
        
        # --- Use gamma as x-axis ---
        gamma_range = np.logspace(-4, -1, 50)  # from 0.0001 to 0.1
        
        complexities = {
            "Our Method": [],
            "Other Randomized Method": []
        }
        
        for gamma in gamma_range:
            # Calculate complexity
            our_complexity = our_method_complexity(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma)
            other_complexity = other_randomized_methods_complexity(EPSILON_FIXED, lambda_val, DELTA_VAL, gamma)
            
            complexities["Our Method"].append(our_complexity)
            complexities["Other Randomized Method"].append(other_complexity)
        
        # Plot curves
        for name, data in complexities.items():
            style = styles[name]
            ax.plot(gamma_range, data, color=style['color'], linestyle=style['linestyle'], 
                   label=style['label'], linewidth=style['linewidth'])
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        title_text = f"{molecule['name']} - {molecule['qubits']} qubits"
        ax.set_title(title_text, fontsize=24, pad=10)
        ax.set_xlabel("Initial State Overlap (γ)", fontsize=20)
        
        ax.tick_params(axis='both', which='major', labelsize=18)

    # --- Shared Y label ---
    axs[0].set_ylabel("Total Number of Gates", fontsize=20)
    
    # --- Shared legend ---
    handles, labels = axs[0].get_legend_handles_labels()
    
    # Use fig.legend with shared styles
    legend = fig.legend(handles, labels, 
               loc='upper center',
               bbox_to_anchor=(0.5, 1.0), 
               ncol=2,  
               fontsize=20,
               frameon=True, 
               shadow=True, 
               title="Methods", 
               title_fontsize=22)
    legend._legend_box.sep = 15
    # Layout configuration
    fig.subplots_adjust(left=0.06, right=0.98, top=0.75, bottom=0.18, wspace=0.2)
    
    # --- Save file ---
    filename = f"gamma-complexity-comparison_error={EPSILON_FIXED}.png"
    fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nFigure saved as: {filename}")
    
    plt.show()
    
    print("Plot complete.")