import openfermion
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner
import pickle

def calculate_and_save_hamiltonian(geometry, basis, multiplicity, charge, description, filename):
    print(f"\n--- Starting calculation for {description} ---")
    
    molecule = MolecularData(geometry, basis, multiplicity, charge, description=description)
    molecule = run_pyscf(molecule)
    
    fermionic_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    
    with open(filename, 'wb') as f:
        pickle.dump(qubit_hamiltonian, f)
    print(f"\n✅ Qubit Hamiltonian for {description} saved to '{filename}'")

    # --- Fixed coefficient extraction below ---
    coefficients = [abs(coeff) for coeff in qubit_hamiltonian.terms.values()]
    # --------------------------

    L = len(coefficients)
    lambda_val = sum(coefficients)
    Lambda_val = max(coefficients) if coefficients else 0
    Lambda_L = Lambda_val * L

    print("\n--- Results ---")
    print(f"Number of qubits: {molecule.n_qubits}")
    print(f"Number of terms (L): {L}")
    print(f"Sum of coefficients (λ): {lambda_val:.4f}")
    print(f"Largest coefficient (Λ): {Lambda_val:.4f}")
    print(f"Product (Λ * L): {Lambda_L:.1f}")
    print(f"--- Calculation for {description} complete ---")


# --- (Molecule definitions) ---
# 1. Propane (C3H8)
propane_geometry = [
    ('C', ( 0.0000,  0.5863,  0.0000)), ('C', (-1.2681, -0.2626,  0.0000)),
    ('C', ( 1.2681, -0.2626,  0.0000)), ('H', ( 0.0000,  1.2449,  0.8760)),
    ('H', (-0.0003,  1.2453, -0.8758)), ('H', (-2.1576,  0.3742,  0.0000)),
    ('H', ( 2.1576,  0.3743, -0.0000)), ('H', (-1.3271, -0.9014,  0.8800)),
    ('H', (-1.3271, -0.9014, -0.8800)), ('H', ( 1.3271, -0.9014, -0.8800)),
    ('H', ( 1.3272, -0.9014,  0.8800))
]
# 2. Carbon Dioxide (CO2)
co2_geometry = [
    ('C', (0.0, 0.0, 0.0)),
    ('O', (0.0, 0.0, 1.1621)),   
    ('O', (0.0, 0.0, -1.1621))
]
# 3. Ethane (C2H6)
ethane_geometry = [
    ('C', ( 0.0000,  0.0000,  0.7680)), ('C', ( 0.0000,  0.0000, -0.7680)),
    ('H', (-1.0192,  0.0000,  1.1573)), ('H', ( 0.5096,  0.8826,  1.1573)),
    ('H', ( 0.5096, -0.8826,  1.1573)), ('H', ( 1.0192,  0.0000, -1.1573)),
    ('H', (-0.5096, -0.8826, -1.1573)), ('H', (-0.5096,  0.8826, -1.1573))
]

if __name__ == "__main__":
    calculate_and_save_hamiltonian(co2_geometry, '6-31g', 1, 0, 
                                   "Carbon_Dioxide", "co2_hamiltonian.pkl")
    calculate_and_save_hamiltonian(ethane_geometry, '6-31g', 1, 0, 
                                   "Ethane", "ethane_hamiltonian.pkl")
    calculate_and_save_hamiltonian(propane_geometry, 'sto-3g', 1, 0, 
                                   "Propane", "propane_hamiltonian.pkl")