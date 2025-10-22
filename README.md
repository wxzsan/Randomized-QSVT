### Randomized-QSVT: Depth and Complexity Studies

This repository contains the source code for the paper:

> **Randomized Quantum Singular Value Transformation**
>
> Xinzhao Wang*, Yuxin Zhang*, Soumyabrata Hazra, Tongyang Li, Changpeng Shao, Shantanav Chakraborty
>
> **[arXiv:2510.06851](https://arxiv.org/abs/2510.06851)**
>
> ---
> <small>\* Equal contribution</small>

## Overview
- Study circuit depth and asymptotic time complexity for ground state property estimation using:
  - Our Method
  - QSVT with Trotterization
  - QETU with Trotterization
  - QETU with qDRIFT
- Includes utilities to construct molecular Hamiltonians and numerically estimate spectral gaps.


## Repository Structure
- depth_functions.py: Asymptotic circuit depth formulas.
- complexity_functions.py: Asymptotic gate complexity formulas.
- combined_depth_vs_n.py: Depth vs $n$ for long-range and hybrid models
- plot_depth_vs_eps.py: Depth vs error tolerance $\epsilon$ for several molecules.
- plot_gamma_complexity_comparison.py: Time complexity vs overlap $\gamma$ (Our Method vs. other randomized methods).
- hamiltoinan_construction.py: Build and save qubit Hamiltonians (OpenFermion + PySCF).
- spectral-gap.py: Numerical spectral gap estimation for long-range spin models (SciPy/Numba).

## Dependencies
- Python 3.9+
- numpy, matplotlib, scipy, numba
- For Hamiltonian construction: openfermion, openfermionpyscf, pyscf


## How to Run
- Chemistry plots:
  Build and store the three molecular qubit Hamiltonians first:
  ```bash
  python hamiltoinan_construction.py
  ```
  This generates `co2_hamiltonian.pkl`, `ethane_hamiltonian.pkl`, and `propane_hamiltonian.pkl` in the repo root.

  Then run the chemistry plotting scripts:
  - Depth vs $\epsilon$ (per molecule, multiple methods):
  ```bash
  python plot_depth_vs_eps.py
  ```
  - Time complexity vs $\gamma$ (Our Method vs other randomized methods):
  ```bash
  python plot_gamma_complexity_comparison.py
  ```

- Depth vs $n$ (long-range and hybrid):
```bash
python combined_depth_vs_n.py
```

- Spectral gap (spin model):
```bash
python spectral_gap.py
```
