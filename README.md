## Variational Ground States and Quasiparticle Excitations in Isometric Tensor Network States

This repository provides implementations of all algorithms presented in the [Master's thesis](latex/thesis.pdf). <br>

Parts 1) and 2) reproduce well-established methods, while part 3) contains a newly developed bulk-weighted boundary compression and a quasiparticle excitation ansatz for isoPEPS:

### 1) Uniform matrix product states (uMPS) `src/uniform_mps/`
- Ansatz `a_umps.py`
- Bond Hamiltonian `b_model_infinite.py`
- Variational uniform matrix product states (VUMPS) algorithm [1] `c_vumps.py`
- Variational plane wave excitations [2] `d_uexcitations.py`

### 2) Matrix product states (MPS) `src/mps/`
- Ansatz `a_mps.py`
- MPO Hamiltonian `b_model_finite.py`
- Density matrix renormalization group (DMRG) algorithm [3] `c_dmrg.py`
- Variational quasiparticle excitations [4] `d_excitations.py`

### 3) Isometric projected entangled pair states (isoPEPS) `src/isometric_peps/`
- Ansatz [5, 6] `a_iso_peps/src/isoTPS/square/isoTPS.py`
- Column MPO Hamiltonian `b_model.py`
- Variational and **bulk-weighted boundary compression** `e_boundary_compression.py`
- $\text{DMRG}^2$ [7] `g_dmrg2.py`
- **Variational quasiparticle excitations** `h_excitations2_overlap.py` and `i_excitations2.py`

The `experiments` folder contains benchmarks on the Transverse field Ising (TFI) model. 
Results are stored in the `data` directory.
The `latex` folder contains the complete LaTeX source code used to generate the [Master's thesis PDF](latex/thesis.pdf).

__References__ <br>
[1] Zauner-Stauber et al., [Variational optimization algorithms for uniform matrix product states](https://arxiv.org/abs/1701.07035), 2018. <br>
[2] Vanderstraeten et al., [Tangent-space methods for uniform matrix product states](https://arxiv.org/abs/1810.07006), 2019. <br>
[3] Schollwöck, [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477), 2011. <br>
[4] Van Damme et al., [Efficient matrix product state methods for extracting spectral information on rings and cylinders](https://arxiv.org/abs/2102.10982), 2021. <br>
[5] Zaletel and Pollmann, [Isometric tensor network states in two dimensions](https://arxiv.org/abs/1902.05100), 2020. <br>
[6] Sappler et al., [Diagonal Isometric Form for Tensor Product States in Two Dimensions](https://arxiv.org/abs/2507.08080), 2025. <br>
[7] Lin et al., [Efficient simulation of dynamics in two-dimensional quantum spin systems with isometric tensor networks](https://arxiv.org/abs/2112.08394), 2022. 
