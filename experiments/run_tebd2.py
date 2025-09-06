import sys
from pathlib import Path
import pickle
import cProfile

import numpy as np
import scipy.sparse as sparse

from ..src.isometric_peps.a_iso_peps.src.isoTPS.square.isoTPS import isoTPS_Square as DiagonalIsometricPEPS
from ..src.isometric_peps.b_model import TFIModelDiagonalSquare


def run_tebd2(Lx, Ly, g, D_max, chi_max_c, dt=0.05, N_sweeps=100, profile=False):
    """Initialize an iso_peps with all spins up and perform N_sweeps TEBD^2 sweeps with imaginary 
    time step dt to find the ground state of the TFI model with transverse field g. Redirect prints 
    to log file and safe iso_peps after each sweep in pkl file."""
    if chi_max_c == "6_D_max":
        chi_max_c = 6*D_max
    script_path = Path(__file__).resolve().parent
    file_base = f"tebd_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{dt}_{N_sweeps}"
    log_path = script_path.parent / "data" / "tebd2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "tebd2" / f"{file_base}.pkl"
    if profile:
        profile_path = script_path.parent / "data" / "tebd2" / f"{file_base}.pstat"
        profiler = cProfile.Profile()
        profiler.enable()
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}, D_max = {D_max}, chi_max_c = {chi_max_c}, dt = {dt}. \n")
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        # exact diagonalization
        if 2*Lx*Ly <= 20:
            H = tfi_model.get_H()
            E0, _ = sparse.linalg.eigsh(H, k=1, which="SA")
            print(f"E0_exact = {E0[0]}. \n")
        # tebd2
        iso_peps = DiagonalIsometricPEPS.from_qubit_product_state(Lx, Ly, D_max, chi_max_c, \
                                                                  spin_orientation="up")
        h_bonds = tfi_model.get_h_bonds()
        u_bonds = tfi_model.get_u_bonds(dt)
        print(f"E = {np.sum(iso_peps.copy().get_bond_expectation_values(h_bonds))}.")
        for i in range(N_sweeps):
            iso_peps.perform_TEBD2(u_bonds, 1)
            print(f"TEBD performed {i+1} sweeps " \
                  + f"-> E = {np.sum(iso_peps.copy().get_bond_expectation_values(h_bonds))}.")
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump(iso_peps, pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if profile:
        profiler.disable()
        profiler.dump_stats(profile_path)
    return iso_peps