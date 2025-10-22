import numpy as np

def our_method_complexity(epsilon, lambda_val, Delta, gamma):
    """Asymptotic gate complexity for Our Method."""
    if epsilon <= 0 or 1/epsilon <= 1: return np.nan
    log_eps = np.log(1 / epsilon)
    if log_eps <= 0: return np.nan
    log_log_eps = np.log(log_eps)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    term1 = (lambda_val**2 * log_eps**4 * log_gamma_eps**2) / (Delta**2 * gamma**2)
    term2 = (log_eps**4 * log_gamma_eps) / (Delta * gamma)
    return (log_log_eps**3 / epsilon**2) * np.maximum(term1, term2)

def qsvt_trotter_complexity(epsilon, lambda_val, Delta, gamma, k, L):
    """Asymptotic gate complexity for QSVT with Trotterization."""
    if epsilon <= 0 or 1/epsilon <= 1: return np.nan
    log_eps = np.log(1 / epsilon)
    if log_eps <= 0: return np.nan
    log_log_eps = np.log(log_eps)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    exp1 = 1 + 1 / (2 * k)
    exp2 = 2 + 1 / (2 * k)
    term1_num = (25/3)**k * (lambda_val**exp1) * (log_eps**exp2) * (log_gamma_eps**exp1)
    term1_den = (Delta * gamma)**exp1
    term1 = term1_num / term1_den
    term2 = (log_eps**3 * log_gamma_eps) / (Delta * gamma)
    return L * (log_log_eps**3 / epsilon**2) * np.maximum(term1, term2)

def qetu_first_order_trotter_complexity(epsilon, lambda_val, Delta, gamma, L):
    """Asymptotic gate complexity for QET-U with first-order Trotter."""
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    numerator = L * lambda_val**2 * log_eps**2 * log_gamma_eps**2
    denominator = epsilon**3 * (Delta * gamma)**2
    return numerator / denominator

def qetu_trotter_suzuki_complexity(epsilon, lambda_val, Delta, gamma, k, L):
    """Asymptotic gate complexity for QET-U with Trotter–Suzuki."""
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    exp_k = 1 + 1 / (2 * k)
    exp_eps = 2 + 1 / (2 * k)
    numerator = L * (25/3)**k * (lambda_val**exp_k) * (log_eps**exp_k) * (log_gamma_eps**exp_k)
    denominator = (epsilon**exp_eps) * ((Delta * gamma)**exp_k)
    return numerator / denominator

def qetu_qdrift_complexity(epsilon, lambda_val, Delta, gamma):
    """Asymptotic gate complexity for QET-U with qDRIFT."""
    log_eps = np.log(1 / epsilon)
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    numerator = lambda_val**2 * log_eps**2 * log_gamma_eps**2
    denominator = epsilon**3 * (Delta * gamma)**2
    return numerator / denominator

def other_randomized_methods_complexity(epsilon, lambda_val, Delta, gamma):
    """Asymptotic gate complexity for other randomized methods."""
    log_gamma_eps = np.log(1 / (gamma * epsilon))
    numerator = log_gamma_eps**2 * lambda_val**2
    denominator = epsilon**2 * Delta**2 * gamma**4
    return numerator / denominator

def calculate_pruned_params(hamiltonian, pruning_eps):
    """Efficiently compute pruned Hamiltonian parameters (L, λ, Λ) without building a new object.
    Args:
        hamiltonian: Original QubitOperator.
        pruning_eps: Pruning threshold (sum of absolute values of discarded terms <= threshold).
    Returns:
        Tuple (L, lambda, Lambda) of pruned parameters.
    """
    current_terms = hamiltonian.terms.copy()

    # 1) Remove identity term if present
    if () in current_terms:
        del current_terms[()]
    
    if not current_terms:
        return 0, 0.0, 0.0

    # 2) Sort all terms by absolute coefficient (ascending)
    sorted_terms_by_abs = sorted(
        current_terms.items(), 
        key=lambda item: abs(item[1])
    )
    
    # 3) Decide how many smallest terms to discard under the threshold
    num_to_discard = 0
    cumulative_sum = 0.0
    for _, coeff in sorted_terms_by_abs:
        if cumulative_sum + abs(coeff) <= pruning_eps:
            cumulative_sum += abs(coeff)
            num_to_discard += 1
        else:
            break
            
    # 4) Slice to obtain the kept terms
    pruned_terms_list = sorted_terms_by_abs[num_to_discard:]
    
    if not pruned_terms_list:
        return 0, 0.0, 0.0

    # L = number of kept terms
    L_new = len(pruned_terms_list)
    # lambda = sum of absolute coefficients over kept terms
    lambda_new = sum(abs(coeff) for _, coeff in pruned_terms_list)
    # Lambda = largest absolute coefficient among kept terms
    Lambda_new = abs(pruned_terms_list[-1][1])
    
    return L_new, lambda_new, Lambda_new

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
