import sys
from pathlib import Path
import pickle
import cProfile

import numpy as np
import scipy.sparse as sparse

from ..src.isometric_peps.a_iso_peps.src.isoTPS.square.isoTPS import isoTPS_Square as DiagonalIsometricPEPS
from ..src.isometric_peps.b_model import TFIModelDiagonalSquare


def run_tebd2(Lx, Ly, g, D_max, chi_max_c, dt=0.08, N_sweeps=100, profile=False):
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
        # exact ground state energies (from exact diagonalization or extrapolated 1d DMRG)
        E0 = None
        if 2*Lx*Ly <= 20:
            H = tfi_model.get_H()
            E0, _ = sparse.linalg.eigsh(H, k=1, which="SA")
        elif g == 3.5:
            if Lx == Ly == 4:
                E0 = -115.74035475
            elif Lx == Ly == 5:
                E0 = -181.21975679
            elif Lx == Ly == 6:
                E0 = -261.32772977
            elif Lx == Ly == 7:
                E0 = -356.06436454
            elif Lx == Ly == 8:
                E0 = -465.42959266
            elif Lx == Ly == 9:
                E0 = -589.42349792
            elif Lx == Ly == 10:
                E0 = -728.04628166
            elif Lx == Ly == 15:
                E0 = -1640.59928393
            elif Lx == Ly == 20:
                E0 = -2918.92808769
        # tebd2
        iso_peps = DiagonalIsometricPEPS.from_qubit_product_state(Lx, Ly, D_max, chi_max_c, \
                                                                  spin_orientation="up")
        h_bonds = tfi_model.get_h_bonds()
        u_bonds = tfi_model.get_u_bonds(dt)
        E = np.sum(iso_peps.copy().get_bond_expectation_values(h_bonds))
        if E0 is not None:
            print(f"deltaE = {E-E0}.")
        else: 
            print(f"E = {E}.")
        for i in range(N_sweeps):
            iso_peps.perform_TEBD2(u_bonds, 1)
            E = np.sum(iso_peps.copy().get_bond_expectation_values(h_bonds))
            if E0 is not None:
                print(f"TEBD performed {i+1} sweeps -> deltaE = {E-E0}.")
            else:
                print(f"TEBD performed {i+1} sweeps -> E = {E}.")
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump(iso_peps, pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if profile:
        profiler.disable()
        profiler.dump_stats(profile_path)
    return iso_peps