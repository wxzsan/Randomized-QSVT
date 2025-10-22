import numpy as np

# ==============================================================================
# New circuit depth calculation functions
# ==============================================================================

def our_method_depth(epsilon, lambda_val, Delta, gamma):
    """Our Method - Circuit Depth"""
    if epsilon <= 0: return np.nan
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    
    term1 = (lambda_val**2 * log_eps**4 * log_gamma_eps**2) / (Delta**2 * gamma**2)
    term2 = (log_eps**4 * log_gamma_eps) / (Delta * gamma)
    
    return np.maximum(term1, term2)

def qsvt_trotter_depth(epsilon, lambda_val, Delta, gamma, k, L):
    """QSVT with Trotterization - Circuit Depth"""
    if epsilon <= 0: return np.nan
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    
    exp1 = 1 + 1 / (2 * k)
    exp2 = 2 + 1 / (2 * k)
    
    term1_num = (25/3)**k * (lambda_val**exp1) * (log_eps**exp2) * (log_gamma_eps**exp1)
    term1_den = (Delta * gamma)**exp1
    term1 = term1_num / term1_den
    
    term2 = (log_eps**3 * log_gamma_eps) / (Delta * gamma)
    
    return L * np.maximum(term1, term2)

def qetu_first_order_trotter_depth(epsilon, lambda_val, Delta, gamma, L):
    """QET-U with the first-order Trotter formula - Circuit Depth"""
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    
    numerator = L * lambda_val**2 * log_eps**2 * log_gamma_eps**2
    denominator = epsilon * (Delta * gamma)**2
    
    return numerator / denominator

def qetu_trotter_suzuki_depth(epsilon, lambda_val, Delta, gamma, k, L):
    """QET-U with Trotter-Suzuki formula - Circuit Depth"""
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))

    exp_k = 1 + 1 / (2 * k)
    exp_eps = 1 / (2 * k)
    
    numerator = L * (25/3)**k * (lambda_val**exp_k) * (log_eps**exp_k) * (log_gamma_eps**exp_k)
    denominator = (epsilon**exp_eps) * ((Delta * gamma)**exp_k)

    return numerator / denominator

def qetu_qdrift_depth(epsilon, lambda_val, Delta, gamma):
    """QET-U with qDRIFT - Circuit Depth"""
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    
    numerator = lambda_val**2 * log_eps**2 * log_gamma_eps**2
    denominator = epsilon * (Delta * gamma)**2
    
    return numerator / denominator

# ==============================================================================
# Helper functions
# ==============================================================================

def get_ordinal(n):
    """Converts a number to its ordinal string representation (e.g., 1 -> 1st, 4 -> 4th)."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th'][n % 10]
    return str(n) + suffix

# Also need calculate_pruned_params
def calculate_pruned_params(hamiltonian, pruning_eps: float):
    current_terms = hamiltonian.terms.copy()
    if () in current_terms: del current_terms[()]
    if not current_terms: return 0, 0.0, 0.0
    sorted_terms_by_abs = sorted(current_terms.items(), key=lambda item: abs(item[1]))
    num_to_discard, cumulative_sum = 0, 0.0
    for _, coeff in sorted_terms_by_abs:
        if cumulative_sum + abs(coeff) <= pruning_eps:
            cumulative_sum += abs(coeff)
            num_to_discard += 1
        else: break
    pruned_terms_list = sorted_terms_by_abs[num_to_discard:]
    if not pruned_terms_list: return 0, 0.0, 0.0
    L_new = len(pruned_terms_list)
    lambda_new = sum(abs(coeff) for _, coeff in pruned_terms_list)
    Lambda_new = abs(pruned_terms_list[-1][1])
    return L_new, lambda_new, Lambda_new