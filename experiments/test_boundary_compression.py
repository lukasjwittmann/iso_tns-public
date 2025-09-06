import sys
from pathlib import Path
import pickle
import cProfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..src.isometric_peps.a_iso_peps.src.isoTPS.square.isoTPS import isoTPS_Square as DiagonalIsometricPEPS
from ..src.isometric_peps.b_model import TFIModelDiagonalSquare
from ..src.isometric_peps.d_expectation_values import get_expectation_value_center, \
                                                      get_expectation_value_boundary, \
                                                      subtract_energy_offset_mpos, \
                                                      get_flipped_As, get_flipped_hs, get_flipped_Cs
from ..src.isometric_peps.e_boundary_compression import get_compressed_boundaries
from ..src.contraction_complexity import get_complexity


def get_compressed_boundary_expectation_value(iso_peps, h_mpos, chi_max_b, N_sweeps_b, combine_hs=True):
    """For an iso_peps compute the expectation value of h_mpos with boundary compression. The 
    maximal bond dimension of the boundary_mps are chi_max_b. N_sweeps_b not None gives the number 
    of sweeps for a variational BoundaryCompression, otherwise BoundaryColumnCompression is used.

    The expectation value has three contributions: 
        - mpos applied to the left of the orthogonality column (compressed)
        - mpo applied on the orthogonality column
        - mpos applied to the right of the orthogonality column (compressed)

    For combine_hs=True the boundary mps of already applied mpos are additively compressed with the 
    next column, whereas for combine_hs=False the boundary mps for each column is transferred 
    separately to the orthogonality column.

    -> E = EL + EC + ER = <Lh|CC*> + <ALhAL*|CC*|ARhAR*> + <CC*|Rh>,

    with <Lh| = sum_n <Lh_n| and |Rh> = sum_n |Rh_n> for combine_hs=False.

    Apart from E, return also all <Lh_n|s and |Rh_n>s and their truncation errors of the final sweep 
    in BoundaryCompression.
    """
    print(f"chi_max_b = {chi_max_b}, combine_hs = {combine_hs}.")
    nc = iso_peps.ortho_surface + 1
    N = 2 * iso_peps.Lx - 1
    NL = nc - 1
    NR = N - nc
    Lhs, Rhs, trunc_errors_Lhs, trunc_errors_Rhs = get_compressed_boundaries(iso_peps, h_mpos, \
                                                                             chi_max_b, N_sweeps_b, \
                                                                             combine_hs)
    E = 0.
    # left energy
    if Lhs is not None:
        print(f"Lh leg dimensions:") 
        for Lh in Lhs:
            print(f"- {[np.shape(M) for M in Lh.Ms]}")
        print(f"Lh truncation errors:") 
        for trunc_errors_Lh in trunc_errors_Lhs:
            print(f"- {np.array(trunc_errors_Lh)}")
        Cs = iso_peps.get_Cs(nc)
        if combine_hs:
            Lh = Lhs[NL-1]    
            EL = get_expectation_value_boundary(Cs, Lh, side="left")
            print(f"-> EL = <Lh|CC*> = {EL}.")
        elif not combine_hs:
            EL = 0.
            for Lh in Lhs:
                EL += get_expectation_value_boundary(Cs, Lh, side="left")
            print(f"-> EL = sum_n <Lh_n|CC*> = {EL}.")
        E += EL
    # center energy
    if nc != 0 and nc != N+1:
        print(f"Orthogonality column {nc}:")
        Cs = iso_peps.get_Cs(nc)
        ALs = iso_peps.get_ALs(nc)
        ARs = iso_peps.get_ARs(nc)
        hs = h_mpos[nc-1]
        if nc%2 == 1:
            EC = get_expectation_value_center(Cs, ALs, ARs, hs)
        elif nc%2 == 0:
            EC = get_expectation_value_center(get_flipped_Cs(Cs), \
                                              get_flipped_As(ALs), \
                                              get_flipped_As(ARs), \
                                              get_flipped_hs(hs))
        print(f"-> EC = <ALhAL*|CC*|ARhAR*> = {EC}.")
        E += EC
    # right energy
    if Rhs is not None:
        print(f"Rh leg dimensions:") 
        for Rh in Rhs:
            print(f"- {[np.shape(M) for M in Rh.Ms]}")
        print(f"Rh truncation errors:") 
        for trunc_errors_Rh in trunc_errors_Rhs:
            print(f"- {np.array(trunc_errors_Rh)}") 
        Cs = iso_peps.get_Cs(nc)
        if combine_hs:
            Rh = Rhs[NR-1] 
            ER = get_expectation_value_boundary(Cs, Rh, side="right")
            print(f"-> ER = <CC*|Rh> = {ER}.")
        elif not combine_hs:
            ER = 0.
            for Rh in Rhs:
                ER += get_expectation_value_boundary(Cs, Rh, side="right")
            print(f"-> ER = sum_n <CC*|Rh_n> = {ER}.")
        E += ER
    print(f"=> E = {E}.")
    return E, Lhs, Rhs, trunc_errors_Lhs, trunc_errors_Rhs


def test_boundary_compression_convergence(Lx, Ly, g, D_max, chi_max_c, dt, N_sweeps, \
                                          chi_max_bs, N_sweeps_b, \
                                          zeroE=True, separately_transferred=True, \
                                          save_bmps=False, profile=False, seed=0):
    """After having run tebd2, test the convergence of BoundaryCompression (with N_sweeps_b) or 
    BoundaryColumnCompression (N_sweeps_b=None) by comparing the resulting energy expectation value 
    to the one without compression (but yang-baxter moves). Redirect prints to log file and save 
    energies, truncation errors (and optionally boundary mps) in pkl file."""
    # unique and reproducible seed
    np.random.seed((seed + Lx + 100 * Ly + 1_000 * D_max) % (2**32))
    # maximal bond dimensions
    if chi_max_c == "6_D_max":
        chi_max_c = 6*D_max
    if chi_max_bs == "32_to_128":
        chi_max_bs = [32, 64, 96, 128]
    if N_sweeps_b == "None":
        N_sweeps_b = None
    # load iso_peps and initialize Hamiltonian mpos
    script_path = Path(__file__).resolve().parent
    file_base_tebd = f"tebd_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{dt}_{N_sweeps}"
    pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
    with open(pkl_path_tebd, "rb") as pkl_file:
        iso_peps = pickle.load(pkl_file)
    iso_peps_copy = DiagonalIsometricPEPS(iso_peps.Lx, iso_peps.Ly, D_max=iso_peps.D_max, \
                                          chi_factor=iso_peps.chi_factor, chi_max=iso_peps.chi_max, \
                                          d=iso_peps.d, shifting_options=iso_peps.shifting_options, \
                                          yb_options=iso_peps.yb_options, \
                                          tebd_options=iso_peps.tebd_options)
    iso_peps_copy._init_as_copy(iso_peps)
    iso_peps = iso_peps_copy
    iso_peps.move_orthogonality_column_to(1)
    tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
    h_mpos = tfi_model.get_h_mpos()
    if zeroE:
        es = iso_peps.copy().get_column_expectation_values(h_mpos)
        subtract_energy_offset_mpos(h_mpos, es)        
    # define log, pkl and profile path (and potentially subtract energy offset)
    file_base = f"bc_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{N_sweeps_b}"
    log_path = script_path.parent / "data" / "boundary_compression" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "boundary_compression" / f"{file_base}.pkl"
    if save_bmps:
        file_base_bmps = f"bmps_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{N_sweeps_b}"
        pkl_path_bmps = script_path.parent / "data" / "boundary_compression" / f"{file_base_bmps}.pkl"
    if profile:
        profile_path = script_path.parent / "data" / "boundary_compression" / f"{file_base}.pstat"
        profiler = cProfile.Profile()
        profiler.enable()
    # run get_compressed_boundary_expectation_value
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}, D_max = {D_max}, chi_max_c = {chi_max_c}, zeroE = {zeroE}. \n")
        # compute column expectation values
        es = iso_peps.copy().get_column_expectation_values(h_mpos)
        E = np.sum(es)
        print(f"E = sum_n <iso_peps|h_n|iso_peps> = sum_n {es} \n  = {E}. \n")
        # compute difference to boundary compression for various bond dimensions
        # with (and without) combine_hs
        chi_max_b_list = []
        if separately_transferred:
            E_bc_list = []  # list (chi_max_bs) of energy values 
            deltaE_bc_list = []  # list (chi_max_bs) of energy difference values
            trunc_errors_Lhs_bc_list = []  # list (chi_max_bs) of list (columns) of list (sites) of truncation errors
        E_bcc_list = []
        deltaE_bcc_list = []
        trunc_errors_Lhs_bcc_list = []
        if save_bmps:
            if separately_transferred:
                Lhs_bc_list = []  # list (chi_max_bs) of list (columns) of boundary mps
            Lhs_bcc_list = []
        for chi_max_b in chi_max_bs:
            chi_max_b_list.append(chi_max_b)
            # combine_hs = False
            if separately_transferred:
                iso_peps_bc = iso_peps.copy()
                iso_peps_bc.move_orthogonality_column_to(2*Lx-1)
                if save_bmps:
                    E_bc, Lhs_bc, _, trunc_errors_Lhs_bc, _ = get_compressed_boundary_expectation_value(iso_peps_bc, h_mpos, \
                                                                                                        chi_max_b, N_sweeps_b, \
                                                                                                        combine_hs=False)
                    Lhs_bc_list.append(Lhs_bc)
                else:
                    E_bc, _, _, trunc_errors_Lhs_bc, _ = get_compressed_boundary_expectation_value(iso_peps_bc, h_mpos, \
                                                                                                   chi_max_b, N_sweeps_b, \
                                                                                                   combine_hs=False)
                E_bc_list.append(E_bc)
                trunc_errors_Lhs_bc_list.append(trunc_errors_Lhs_bc)
                deltaE_bc = E_bc - E
                deltaE_bc_list.append(deltaE_bc)
                print(f"=> deltaE_bc = {deltaE_bc}.")
            # combine_hs = True
            iso_peps_bcc = iso_peps.copy()
            iso_peps_bcc.move_orthogonality_column_to(2*Lx-1)
            if save_bmps:
                E_bcc, Lhs_bcc, _, trunc_errors_Lhs_bcc, _ = get_compressed_boundary_expectation_value(iso_peps_bcc, h_mpos, \
                                                                                                       chi_max_b, N_sweeps_b, \
                                                                                                       combine_hs=True)
                Lhs_bcc_list.append(Lhs_bcc)
            else:
                E_bcc, _, _, trunc_errors_Lhs_bcc, _ = get_compressed_boundary_expectation_value(iso_peps_bcc, h_mpos, \
                                                                                                 chi_max_b, N_sweeps_b, \
                                                                                                 combine_hs=True)
            E_bcc_list.append(E_bcc)
            trunc_errors_Lhs_bcc_list.append(trunc_errors_Lhs_bcc)
            deltaE_bcc = E_bcc - E
            deltaE_bcc_list.append(deltaE_bcc)
            print(f"=> deltaE_bcc = {deltaE_bcc}. \n")
            # save energy, boundary mps (for all columns) and truncation errors (for all columns and all sites)
            if separately_transferred:
                with open(pkl_path, "wb") as pkl_file:
                    pickle.dump((E, chi_max_b_list, \
                                 E_bc_list, trunc_errors_Lhs_bc_list, \
                                 E_bcc_list, trunc_errors_Lhs_bcc_list), \
                                 pkl_file)
            else:
                with open(pkl_path, "wb") as pkl_file:
                    pickle.dump((E, chi_max_b_list, \
                                 E_bcc_list, trunc_errors_Lhs_bcc_list), \
                                 pkl_file)
            if save_bmps:
                if separately_transferred:
                    with open(pkl_path_bmps, "wb") as pkl_file_bmps:
                        pickle.dump((chi_max_b_list, Lhs_bc_list, Lhs_bcc_list), pkl_file_bmps)
                else:
                    with open(pkl_path_bmps, "wb") as pkl_file_bmps:
                        pickle.dump((chi_max_b_list, Lhs_bcc_list), pkl_file_bmps)
        # print the energies and maximal truncation errors for all chi_max_bs
        print(f"for chi_max_bs = {chi_max_bs}:")
        if separately_transferred:
            print(f"- without combining hs: \n" \
                  + f"deltaEs = {np.array(deltaE_bc_list)}, \n" \
                  + f"maximal truncation errors = {np.array([np.max(np.array(trunc_errors_Lhs_bc)) for trunc_errors_Lhs_bc in trunc_errors_Lhs_bc_list])}. ")
        print(f"- with combining hs: \n" \
              + f"deltaEs = {np.array(deltaE_bcc_list)}, \n" \
              + f"maximal truncation errors = {np.array([np.max(np.array(trunc_errors_Lhs_bcc)) for trunc_errors_Lhs_bcc in trunc_errors_Lhs_bcc_list])}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if profile:
        profiler.disable()
        profiler.dump_stats(profile_path)
    return


def plot_boundary_compression():
    """After having ran test_boundary_compression_convergence for iso_peps ground state of fixed Lx, 
    Ly, g and varying D_max = 2, ..., 6, plot the energy differences against chi_max_bs. Both for 
    N_sweeps_b = 3 (variational BoundaryCompression) and N_sweeps = None (BoundaryColumnCompression).
    """
    Lx = 6
    Ly = 6
    script_path = Path(__file__).resolve().parent
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    png_path = script_path.parent / "data" / "boundary_compression" / f"boundary_compression_{Lx}_{Ly}.png"
    # g = 3.5
    g = 3.5
    E0 = -261.32772977
    axes[0].set_title(rf"(a) $g = {g}$")
    axes[0].set_ylabel(r"$\frac{\vert E_{\text{bc}} - E_{\text{YB}} \vert}{\vert E_{\text{YB}} \vert}$", fontsize=13)
    axes[0].set_ylim(0.8*1e-5, 1.5*1e-1)
    # N_sweeps_b = 3 (variational BoundaryCompression)
    # load data
    script_path = Path(__file__).resolve().parent
    pkl_path_2 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{2}_{12}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{3}_{18}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{4}_{24}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{5}_{30}_{3}.pkl"
    pkl_path_6 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{6}_{36}_{3}.pkl" 
    chi_max_b_list_2 = None   
    try:
        with open(pkl_path_2, "rb") as pkl_file_2:
            E, chi_max_b_list_2, E_bc_list_2, trunc_errors_Lhs_bc_list_2, E_bcc_list_2, trunc_errors_Lhs_bcc_list_2 = pickle.load(pkl_file_2)
            deltaE_bc_list_2 = [(E_bc_2 - E)/np.abs(E0) for E_bc_2 in E_bc_list_2]
            deltaE_bcc_list_2 = [(E_bcc_2 - E)/np.abs(E0) for E_bcc_2 in E_bcc_list_2]
    except FileNotFoundError:
        print("No data available for D_max = 2 (N_sweeps_b = 3).")
    chi_max_b_list_3 = None   
    try:
        with open(pkl_path_3, "rb") as pkl_file_3:
            E, chi_max_b_list_3, E_bc_list_3, trunc_errors_Lhs_bc_list_3, E_bcc_list_3, trunc_errors_Lhs_bcc_list_3 = pickle.load(pkl_file_3)
            deltaE_bc_list_3 = [(E_bc_3 - E)/np.abs(E0) for E_bc_3 in E_bc_list_3]
            deltaE_bcc_list_3 = [(E_bcc_3 - E)/np.abs(E0) for E_bcc_3 in E_bcc_list_3]
    except FileNotFoundError:
        print("No data available for D_max = 3 (N_sweeps_b = 3).")
    chi_max_b_list_4 = None
    try:
        with open(pkl_path_4, "rb") as pkl_file_4:
            E, chi_max_b_list_4, E_bc_list_4, trunc_errors_Lhs_bc_list_4, E_bcc_list_4, trunc_errors_Lhs_bcc_list_4 = pickle.load(pkl_file_4)
            deltaE_bc_list_4 = [(E_bc_4 - E)/np.abs(E0) for E_bc_4 in E_bc_list_4]
            deltaE_bcc_list_4 = [(E_bcc_4 - E)/np.abs(E0) for E_bcc_4 in E_bcc_list_4]
    except FileNotFoundError:
        print("No data available for D_max = 4 (N_sweeps_b = 3).")
    chi_max_b_list_5 = None
    try:
        with open(pkl_path_5, "rb") as pkl_file_5:
            E, chi_max_b_list_5, E_bc_list_5, trunc_errors_Lhs_bc_list_5, E_bcc_list_5, trunc_errors_Lhs_bcc_list_5 = pickle.load(pkl_file_5)
            deltaE_bc_list_5 = [(E_bc_5 - E)/np.abs(E0) for E_bc_5 in E_bc_list_5]
            deltaE_bcc_list_5 = [(E_bcc_5 - E)/np.abs(E0) for E_bcc_5 in E_bcc_list_5]
    except FileNotFoundError:
        print("No data available for D_max = 5 (N_sweeps_b = 3).")
    chi_max_b_list_6 = None
    try:
        with open(pkl_path_6, "rb") as pkl_file_6:
            E, chi_max_b_list_6, E_bc_list_6, trunc_errors_Lhs_bc_list_6, E_bcc_list_6, trunc_errors_Lhs_bcc_list_6 = pickle.load(pkl_file_6)
            deltaE_bc_list_6 = [(E_bc_6 - E)/np.abs(E0) for E_bc_6 in E_bc_list_6]
            deltaE_bcc_list_6 = [(E_bcc_6 - E)/np.abs(E0) for E_bcc_6 in E_bcc_list_6]
    except FileNotFoundError:
        print(rf"No data available for D_max = 6 (N_sweeps_b = 3).")
    # plot energies
    if chi_max_b_list_2 is not None:
        chi_max_b_list_2_pos, deltE_bcc_list_2_pos, chi_max_b_list_2_neg, deltE_bcc_list_2_neg = separate_pos_neg_values(chi_max_b_list_2, deltaE_bcc_list_2)
        axes[0].semilogy(chi_max_b_list_2_pos, deltE_bcc_list_2_pos, ".", color="gold")
        axes[0].semilogy(chi_max_b_list_2_neg, np.abs(deltE_bcc_list_2_neg), "x", color="gold")
        energy_2_bcc_var, = axes[0].semilogy(chi_max_b_list_2, np.abs(deltaE_bcc_list_2), "--", color="gold", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_3 is not None:
        chi_max_b_list_3_pos, deltE_bcc_list_3_pos, chi_max_b_list_3_neg, deltE_bcc_list_3_neg = separate_pos_neg_values(chi_max_b_list_3, deltaE_bcc_list_3)
        axes[0].semilogy(chi_max_b_list_3_pos, deltE_bcc_list_3_pos, ".", color="yellowgreen")
        axes[0].semilogy(chi_max_b_list_3_neg, np.abs(deltE_bcc_list_3_neg), "x", color="yellowgreen")
        energy_3_bcc_var, = axes[0].semilogy(chi_max_b_list_3, np.abs(deltaE_bcc_list_3), "--", color="yellowgreen", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_4 is not None:
        chi_max_b_list_4_pos, deltE_bcc_list_4_pos, chi_max_b_list_4_neg, deltE_bcc_list_4_neg = separate_pos_neg_values(chi_max_b_list_4, deltaE_bcc_list_4)
        axes[0].semilogy(chi_max_b_list_4_pos, deltE_bcc_list_4_pos, ".", color="limegreen")
        axes[0].semilogy(chi_max_b_list_4_neg, np.abs(deltE_bcc_list_4_neg), "x", color="limegreen")
        energy_4_bcc_var, = axes[0].semilogy(chi_max_b_list_4, np.abs(deltaE_bcc_list_4), "--", color="limegreen", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_5 is not None:
        chi_max_b_list_5_pos, deltE_bcc_list_5_pos, chi_max_b_list_5_neg, deltE_bcc_list_5_neg = separate_pos_neg_values(chi_max_b_list_5, deltaE_bcc_list_5)
        axes[0].semilogy(chi_max_b_list_5_pos, deltE_bcc_list_5_pos, ".", color="green")
        axes[0].semilogy(chi_max_b_list_5_neg, np.abs(deltE_bcc_list_5_neg), "x", color="green")
        energy_5_bcc_var, = axes[0].semilogy(chi_max_b_list_5, np.abs(deltaE_bcc_list_5), "--", color="green", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_6 is not None:
        chi_max_b_list_6_pos, deltE_bcc_list_6_pos, chi_max_b_list_6_neg, deltE_bcc_list_6_neg = separate_pos_neg_values(chi_max_b_list_6, deltaE_bcc_list_6)
        axes[0].semilogy(chi_max_b_list_6_pos, deltE_bcc_list_6_pos, ".", color="darkslategray")
        axes[0].semilogy(chi_max_b_list_6_neg, np.abs(deltE_bcc_list_6_neg), "x", color="darkslategray")
        energy_6_bcc_var, = axes[0].semilogy(chi_max_b_list_6, np.abs(deltaE_bcc_list_6), "--", color="darkslategray", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    # N_sweeps_b = None (BoundaryColumnCompression)
    # load data
    script_path = Path(__file__).resolve().parent
    pkl_path_2 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{2}_{12}_{None}.pkl"
    pkl_path_3 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{3}_{18}_{None}.pkl"
    pkl_path_4 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{4}_{24}_{None}.pkl"
    pkl_path_5 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{5}_{30}_{None}.pkl"
    pkl_path_6 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{6}_{36}_{None}.pkl" 
    chi_max_b_list_2 = None   
    try:
        with open(pkl_path_2, "rb") as pkl_file_2:
            E, chi_max_b_list_2, E_bc_list_2, trunc_errors_Lhs_bc_list_2, E_bcc_list_2, trunc_errors_Lhs_bcc_list_2 = pickle.load(pkl_file_2)
            deltaE_bc_list_2 = [(E_bc_2 - E)/np.abs(E0) for E_bc_2 in E_bc_list_2]
            deltaE_bcc_list_2 = [(E_bcc_2 - E)/np.abs(E0) for E_bcc_2 in E_bcc_list_2]
    except FileNotFoundError:
        print(rf"No data available for D_max = 2 (N_sweeps_b = None).")
    chi_max_b_list_3 = None   
    try:
        with open(pkl_path_3, "rb") as pkl_file_3:
            E, chi_max_b_list_3, E_bc_list_3, trunc_errors_Lhs_bc_list_3, E_bcc_list_3, trunc_errors_Lhs_bcc_list_3 = pickle.load(pkl_file_3)
            deltaE_bc_list_3 = [(E_bc_3 - E)/np.abs(E0) for E_bc_3 in E_bc_list_3]
            deltaE_bcc_list_3 = [(E_bcc_3 - E)/np.abs(E0) for E_bcc_3 in E_bcc_list_3]
    except FileNotFoundError:
        print(rf"No data available for D_max = 3 (N_sweeps_b = None).")
    chi_max_b_list_4 = None
    try:
        with open(pkl_path_4, "rb") as pkl_file_4:
            E, chi_max_b_list_4, E_bc_list_4, trunc_errors_Lhs_bc_list_4, E_bcc_list_4, trunc_errors_Lhs_bcc_list_4 = pickle.load(pkl_file_4)
            deltaE_bc_list_4 = [(E_bc_4 - E)/np.abs(E0) for E_bc_4 in E_bc_list_4]
            deltaE_bcc_list_4 = [(E_bcc_4 - E)/np.abs(E0) for E_bcc_4 in E_bcc_list_4]
    except FileNotFoundError:
        print(rf"No data available for D_max = 4 (N_sweeps_b = None).")
    chi_max_b_list_5 = None
    try:
        with open(pkl_path_5, "rb") as pkl_file_5:
            E, chi_max_b_list_5, E_bc_list_5, trunc_errors_Lhs_bc_list_5, E_bcc_list_5, trunc_errors_Lhs_bcc_list_5 = pickle.load(pkl_file_5)
            deltaE_bc_list_5 = [(E_bc_5 - E)/np.abs(E0) for E_bc_5 in E_bc_list_5]
            deltaE_bcc_list_5 = [(E_bcc_5 - E)/np.abs(E0) for E_bcc_5 in E_bcc_list_5]
    except FileNotFoundError:
        print(rf"No data available for D_max = 5 (N_sweeps_b = None).")
    chi_max_b_list_6 = None
    try:
        with open(pkl_path_6, "rb") as pkl_file_6:
            E, chi_max_b_list_6, E_bc_list_6, trunc_errors_Lhs_bc_list_6, E_bcc_list_6, trunc_errors_Lhs_bcc_list_6 = pickle.load(pkl_file_6)
            deltaE_bc_list_6 = [(E_bc_6 - E)/np.abs(E0) for E_bc_6 in E_bc_list_6]
            deltaE_bcc_list_6 = [(E_bcc_6 - E)/np.abs(E0) for E_bcc_6 in E_bcc_list_6]
    except FileNotFoundError:
        print(rf"No data available for D_max = 6 (N_sweeps_b = None).")
    # plot energies
    if chi_max_b_list_2 is not None:
        chi_max_b_list_2_pos, deltE_bcc_list_2_pos, chi_max_b_list_2_neg, deltE_bcc_list_2_neg = separate_pos_neg_values(chi_max_b_list_2, deltaE_bcc_list_2)
        axes[0].semilogy(chi_max_b_list_2_pos, deltE_bcc_list_2_pos, ".", color="gold")
        axes[0].semilogy(chi_max_b_list_2_neg, np.abs(deltE_bcc_list_2_neg), "x", color="gold")
        energy_2_bcc_col, = axes[0].semilogy(chi_max_b_list_2, np.abs(deltaE_bcc_list_2), "-", color="gold", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    # plot energies
    if chi_max_b_list_3 is not None:
        chi_max_b_list_3_pos, deltE_bcc_list_3_pos, chi_max_b_list_3_neg, deltE_bcc_list_3_neg = separate_pos_neg_values(chi_max_b_list_3, deltaE_bcc_list_3)
        axes[0].semilogy(chi_max_b_list_3_pos, deltE_bcc_list_3_pos, ".", color="yellowgreen")
        axes[0].semilogy(chi_max_b_list_3_neg, np.abs(deltE_bcc_list_3_neg), "x", color="yellowgreen")
        energy_3_bcc_col, = axes[0].semilogy(chi_max_b_list_3, np.abs(deltaE_bcc_list_3), "-", color="yellowgreen", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    # violet indigo
    if chi_max_b_list_4 is not None:
        chi_max_b_list_4_pos, deltE_bcc_list_4_pos, chi_max_b_list_4_neg, deltE_bcc_list_4_neg = separate_pos_neg_values(chi_max_b_list_4, deltaE_bcc_list_4)
        axes[0].semilogy(chi_max_b_list_4_pos, deltE_bcc_list_4_pos, ".", color="limegreen")
        axes[0].semilogy(chi_max_b_list_4_neg, np.abs(deltE_bcc_list_4_neg), "x", color="limegreen")
        energy_4_bcc_col, = axes[0].semilogy(chi_max_b_list_4, np.abs(deltaE_bcc_list_4), "-", color="limegreen", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    if chi_max_b_list_5 is not None:
        chi_max_b_list_5_pos, deltE_bcc_list_5_pos, chi_max_b_list_5_neg, deltE_bcc_list_5_neg = separate_pos_neg_values(chi_max_b_list_5, deltaE_bcc_list_5)
        axes[0].semilogy(chi_max_b_list_5_pos, deltE_bcc_list_5_pos, ".", color="green")
        axes[0].semilogy(chi_max_b_list_5_neg, np.abs(deltE_bcc_list_5_neg), "x", color="green")
        energy_5_bcc_col, = axes[0].semilogy(chi_max_b_list_5, np.abs(deltaE_bcc_list_5), "-", color="green", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    if chi_max_b_list_6 is not None:
        chi_max_b_list_6_pos, deltE_bcc_list_6_pos, chi_max_b_list_6_neg, deltE_bcc_list_6_neg = separate_pos_neg_values(chi_max_b_list_6, deltaE_bcc_list_6)
        axes[0].semilogy(chi_max_b_list_6_pos, deltE_bcc_list_6_pos, ".", color="darkslategray")
        axes[0].semilogy(chi_max_b_list_6_neg, np.abs(deltE_bcc_list_6_neg), "x", color="darkslategray")
        energy_6_bcc_col, = axes[0].semilogy(chi_max_b_list_6, np.abs(deltaE_bcc_list_6), "-", color="darkslategray", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    # g = 3.0
    g = 3.0
    E0 = -227.308603832
    axes[1].set_title(rf"(b) $g = {g}$")
    axes[1].set_xlabel(r"$\chi_{\text{max,b}}$", fontsize=12)
    axes[1].set_ylabel(r"$\frac{\vert E_{\text{bc}} - E_{\text{YB}} \vert}{\vert E_{\text{YB}} \vert}$", fontsize=13)
    axes[1].set_ylim(0.8*1e-5, 1.5*1e-1)
    # N_sweeps_b = 3 (variational BoundaryCompression)
    # load data
    script_path = Path(__file__).resolve().parent
    pkl_path_2 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{2}_{12}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{3}_{18}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{4}_{24}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{5}_{30}_{3}.pkl"
    pkl_path_6 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{6}_{36}_{3}.pkl" 
    chi_max_b_list_2 = None   
    try:
        with open(pkl_path_2, "rb") as pkl_file_2:
            E, chi_max_b_list_2, E_bc_list_2, trunc_errors_Lhs_bc_list_2, E_bcc_list_2, trunc_errors_Lhs_bcc_list_2 = pickle.load(pkl_file_2)
            deltaE_bc_list_2 = [(E_bc_2 - E)/np.abs(E0) for E_bc_2 in E_bc_list_2]
            deltaE_bcc_list_2 = [(E_bcc_2 - E)/np.abs(E0) for E_bcc_2 in E_bcc_list_2]
    except FileNotFoundError:
        print("No data available for D_max = 2 (N_sweeps_b = 3).")
    chi_max_b_list_3 = None   
    try:
        with open(pkl_path_3, "rb") as pkl_file_3:
            E, chi_max_b_list_3, E_bc_list_3, trunc_errors_Lhs_bc_list_3, E_bcc_list_3, trunc_errors_Lhs_bcc_list_3 = pickle.load(pkl_file_3)
            deltaE_bc_list_3 = [(E_bc_3 - E)/np.abs(E0) for E_bc_3 in E_bc_list_3]
            deltaE_bcc_list_3 = [(E_bcc_3 - E)/np.abs(E0) for E_bcc_3 in E_bcc_list_3]
    except FileNotFoundError:
        print("No data available for D_max = 3 (N_sweeps_b = 3).")
    chi_max_b_list_4 = None
    try:
        with open(pkl_path_4, "rb") as pkl_file_4:
            E, chi_max_b_list_4, E_bc_list_4, trunc_errors_Lhs_bc_list_4, E_bcc_list_4, trunc_errors_Lhs_bcc_list_4 = pickle.load(pkl_file_4)
            deltaE_bc_list_4 = [(E_bc_4 - E)/np.abs(E0) for E_bc_4 in E_bc_list_4]
            deltaE_bcc_list_4 = [(E_bcc_4 - E)/np.abs(E0) for E_bcc_4 in E_bcc_list_4]
    except FileNotFoundError:
        print("No data available for D_max = 4 (N_sweeps_b = 3).")
    chi_max_b_list_5 = None
    try:
        with open(pkl_path_5, "rb") as pkl_file_5:
            E, chi_max_b_list_5, E_bc_list_5, trunc_errors_Lhs_bc_list_5, E_bcc_list_5, trunc_errors_Lhs_bcc_list_5 = pickle.load(pkl_file_5)
            deltaE_bc_list_5 = [(E_bc_5 - E)/np.abs(E0) for E_bc_5 in E_bc_list_5]
            deltaE_bcc_list_5 = [(E_bcc_5 - E)/np.abs(E0) for E_bcc_5 in E_bcc_list_5]
    except FileNotFoundError:
        print("No data available for D_max = 5 (N_sweeps_b = 3).")
    chi_max_b_list_6 = None
    try:
        with open(pkl_path_6, "rb") as pkl_file_6:
            E, chi_max_b_list_6, E_bc_list_6, trunc_errors_Lhs_bc_list_6, E_bcc_list_6, trunc_errors_Lhs_bcc_list_6 = pickle.load(pkl_file_6)
            deltaE_bc_list_6 = [(E_bc_6 - E)/np.abs(E0) for E_bc_6 in E_bc_list_6]
            deltaE_bcc_list_6 = [(E_bcc_6 - E)/np.abs(E0) for E_bcc_6 in E_bcc_list_6]
    except FileNotFoundError:
        print(rf"No data available for D_max = 6 (N_sweeps_b = 3).")
    # plot energies
    if chi_max_b_list_2 is not None:
        chi_max_b_list_2_pos, deltE_bcc_list_2_pos, chi_max_b_list_2_neg, deltE_bcc_list_2_neg = separate_pos_neg_values(chi_max_b_list_2, deltaE_bcc_list_2)
        axes[1].semilogy(chi_max_b_list_2_pos, deltE_bcc_list_2_pos, ".", color="gold")
        axes[1].semilogy(chi_max_b_list_2_neg, np.abs(deltE_bcc_list_2_neg), "x", color="gold")
        energy_2_bcc_var, = axes[1].semilogy(chi_max_b_list_2, np.abs(deltaE_bcc_list_2), "--", color="gold", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_3 is not None:
        chi_max_b_list_3_pos, deltE_bcc_list_3_pos, chi_max_b_list_3_neg, deltE_bcc_list_3_neg = separate_pos_neg_values(chi_max_b_list_3, deltaE_bcc_list_3)
        axes[1].semilogy(chi_max_b_list_3_pos, deltE_bcc_list_3_pos, ".", color="yellowgreen")
        axes[1].semilogy(chi_max_b_list_3_neg, np.abs(deltE_bcc_list_3_neg), "x", color="yellowgreen")
        energy_3_bcc_var, = axes[1].semilogy(chi_max_b_list_3, np.abs(deltaE_bcc_list_3), "--", color="yellowgreen", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_4 is not None:
        chi_max_b_list_4_pos, deltE_bcc_list_4_pos, chi_max_b_list_4_neg, deltE_bcc_list_4_neg = separate_pos_neg_values(chi_max_b_list_4, deltaE_bcc_list_4)
        axes[1].semilogy(chi_max_b_list_4_pos, deltE_bcc_list_4_pos, ".", color="limegreen")
        axes[1].semilogy(chi_max_b_list_4_neg, np.abs(deltE_bcc_list_4_neg), "x", color="limegreen")
        energy_4_bcc_var, = axes[1].semilogy(chi_max_b_list_4, np.abs(deltaE_bcc_list_4), "--", color="limegreen", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_5 is not None:
        chi_max_b_list_5_pos, deltE_bcc_list_5_pos, chi_max_b_list_5_neg, deltE_bcc_list_5_neg = separate_pos_neg_values(chi_max_b_list_5, deltaE_bcc_list_5)
        axes[1].semilogy(chi_max_b_list_5_pos, deltE_bcc_list_5_pos, ".", color="green")
        axes[1].semilogy(chi_max_b_list_5_neg, np.abs(deltE_bcc_list_5_neg), "x", color="green")
        energy_5_bcc_var, = axes[1].semilogy(chi_max_b_list_5, np.abs(deltaE_bcc_list_5), "--", color="green", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    if chi_max_b_list_6 is not None:
        chi_max_b_list_6_pos, deltE_bcc_list_6_pos, chi_max_b_list_6_neg, deltE_bcc_list_6_neg = separate_pos_neg_values(chi_max_b_list_6, deltaE_bcc_list_6)
        axes[1].semilogy(chi_max_b_list_6_pos, deltE_bcc_list_6_pos, ".", color="darkslategray")
        axes[1].semilogy(chi_max_b_list_6_neg, np.abs(deltE_bcc_list_6_neg), "x", color="darkslategray")
        energy_6_bcc_var, = axes[1].semilogy(chi_max_b_list_6, np.abs(deltaE_bcc_list_6), "--", color="darkslategray", \
                                               label=rf"variational BoundaryCompression, additively compressed")
    # N_sweeps_b = None (BoundaryColumnCompression)
    # load data
    script_path = Path(__file__).resolve().parent
    pkl_path_2 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{2}_{12}_{None}.pkl"
    pkl_path_3 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{3}_{18}_{None}.pkl"
    pkl_path_4 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{4}_{24}_{None}.pkl"
    pkl_path_5 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{5}_{30}_{None}.pkl"
    pkl_path_6 = script_path.parent / "data" / "boundary_compression" / f"bc_{Lx}_{Ly}_{g}_{6}_{36}_{None}.pkl" 
    chi_max_b_list_2 = None   
    try:
        with open(pkl_path_2, "rb") as pkl_file_2:
            E, chi_max_b_list_2, E_bc_list_2, trunc_errors_Lhs_bc_list_2, E_bcc_list_2, trunc_errors_Lhs_bcc_list_2 = pickle.load(pkl_file_2)
            deltaE_bc_list_2 = [(E_bc_2 - E)/np.abs(E0) for E_bc_2 in E_bc_list_2]
            deltaE_bcc_list_2 = [(E_bcc_2 - E)/np.abs(E0) for E_bcc_2 in E_bcc_list_2]
    except FileNotFoundError:
        print(rf"No data available for D_max = 2 (N_sweeps_b = None).")
    chi_max_b_list_3 = None   
    try:
        with open(pkl_path_3, "rb") as pkl_file_3:
            E, chi_max_b_list_3, E_bc_list_3, trunc_errors_Lhs_bc_list_3, E_bcc_list_3, trunc_errors_Lhs_bcc_list_3 = pickle.load(pkl_file_3)
            deltaE_bc_list_3 = [(E_bc_3 - E)/np.abs(E0) for E_bc_3 in E_bc_list_3]
            deltaE_bcc_list_3 = [(E_bcc_3 - E)/np.abs(E0) for E_bcc_3 in E_bcc_list_3]
    except FileNotFoundError:
        print(rf"No data available for D_max = 3 (N_sweeps_b = None).")
    chi_max_b_list_4 = None
    try:
        with open(pkl_path_4, "rb") as pkl_file_4:
            E, chi_max_b_list_4, E_bc_list_4, trunc_errors_Lhs_bc_list_4, E_bcc_list_4, trunc_errors_Lhs_bcc_list_4 = pickle.load(pkl_file_4)
            deltaE_bc_list_4 = [(E_bc_4 - E)/np.abs(E0) for E_bc_4 in E_bc_list_4]
            deltaE_bcc_list_4 = [(E_bcc_4 - E)/np.abs(E0) for E_bcc_4 in E_bcc_list_4]
    except FileNotFoundError:
        print(rf"No data available for D_max = 4 (N_sweeps_b = None).")
    chi_max_b_list_5 = None
    try:
        with open(pkl_path_5, "rb") as pkl_file_5:
            E, chi_max_b_list_5, E_bc_list_5, trunc_errors_Lhs_bc_list_5, E_bcc_list_5, trunc_errors_Lhs_bcc_list_5 = pickle.load(pkl_file_5)
            deltaE_bc_list_5 = [(E_bc_5 - E)/np.abs(E0) for E_bc_5 in E_bc_list_5]
            deltaE_bcc_list_5 = [(E_bcc_5 - E)/np.abs(E0) for E_bcc_5 in E_bcc_list_5]
    except FileNotFoundError:
        print(rf"No data available for D_max = 5 (N_sweeps_b = None).")
    chi_max_b_list_6 = None
    try:
        with open(pkl_path_6, "rb") as pkl_file_6:
            E, chi_max_b_list_6, E_bc_list_6, trunc_errors_Lhs_bc_list_6, E_bcc_list_6, trunc_errors_Lhs_bcc_list_6 = pickle.load(pkl_file_6)
            deltaE_bc_list_6 = [(E_bc_6 - E)/np.abs(E0) for E_bc_6 in E_bc_list_6]
            deltaE_bcc_list_6 = [(E_bcc_6 - E)/np.abs(E0) for E_bcc_6 in E_bcc_list_6]
    except FileNotFoundError:
        print(rf"No data available for D_max = 6 (N_sweeps_b = None).")
    # plot energies
    if chi_max_b_list_2 is not None:
        chi_max_b_list_2_pos, deltE_bcc_list_2_pos, chi_max_b_list_2_neg, deltE_bcc_list_2_neg = separate_pos_neg_values(chi_max_b_list_2, deltaE_bcc_list_2)
        axes[1].semilogy(chi_max_b_list_2_pos, deltE_bcc_list_2_pos, ".", color="gold")
        axes[1].semilogy(chi_max_b_list_2_neg, np.abs(deltE_bcc_list_2_neg), "x", color="gold")
        energy_2_bcc_col, = axes[1].semilogy(chi_max_b_list_2, np.abs(deltaE_bcc_list_2), "-", color="gold", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    # plot energies
    if chi_max_b_list_3 is not None:
        chi_max_b_list_3_pos, deltE_bcc_list_3_pos, chi_max_b_list_3_neg, deltE_bcc_list_3_neg = separate_pos_neg_values(chi_max_b_list_3, deltaE_bcc_list_3)
        axes[1].semilogy(chi_max_b_list_3_pos, deltE_bcc_list_3_pos, ".", color="yellowgreen")
        axes[1].semilogy(chi_max_b_list_3_neg, np.abs(deltE_bcc_list_3_neg), "x", color="yellowgreen")
        energy_3_bcc_col, = axes[1].semilogy(chi_max_b_list_3, np.abs(deltaE_bcc_list_3), "-", color="yellowgreen", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    # violet indigo
    if chi_max_b_list_4 is not None:
        chi_max_b_list_4_pos, deltE_bcc_list_4_pos, chi_max_b_list_4_neg, deltE_bcc_list_4_neg = separate_pos_neg_values(chi_max_b_list_4, deltaE_bcc_list_4)
        axes[1].semilogy(chi_max_b_list_4_pos, deltE_bcc_list_4_pos, ".", color="limegreen")
        axes[1].semilogy(chi_max_b_list_4_neg, np.abs(deltE_bcc_list_4_neg), "x", color="limegreen")
        energy_4_bcc_col, = axes[1].semilogy(chi_max_b_list_4, np.abs(deltaE_bcc_list_4), "-", color="limegreen", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    if chi_max_b_list_5 is not None:
        chi_max_b_list_5_pos, deltE_bcc_list_5_pos, chi_max_b_list_5_neg, deltE_bcc_list_5_neg = separate_pos_neg_values(chi_max_b_list_5, deltaE_bcc_list_5)
        axes[1].semilogy(chi_max_b_list_5_pos, deltE_bcc_list_5_pos, ".", color="green")
        axes[1].semilogy(chi_max_b_list_5_neg, np.abs(deltE_bcc_list_5_neg), "x", color="green")
        energy_5_bcc_col, = axes[1].semilogy(chi_max_b_list_5, np.abs(deltaE_bcc_list_5), "-", color="green", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    if chi_max_b_list_6 is not None:
        chi_max_b_list_6_pos, deltE_bcc_list_6_pos, chi_max_b_list_6_neg, deltE_bcc_list_6_neg = separate_pos_neg_values(chi_max_b_list_6, deltaE_bcc_list_6)
        axes[1].semilogy(chi_max_b_list_6_pos, deltE_bcc_list_6_pos, ".", color="darkslategray")
        axes[1].semilogy(chi_max_b_list_6_neg, np.abs(deltE_bcc_list_6_neg), "x", color="darkslategray")
        energy_6_bcc_col, = axes[1].semilogy(chi_max_b_list_6, np.abs(deltaE_bcc_list_6), "-", color="darkslategray", \
                                               label=rf"BoundaryColumnCompression, additively compressed")
    # ticks and legends 
    axes[0].set_xticks(chi_max_b_list_2)
    axes[0].set_xticklabels([])
    axes[1].set_xticks(chi_max_b_list_2)
    axes[1].set_xticklabels(chi_max_b_list_2)
    fig.legend(handles=[energy_2_bcc_var, energy_3_bcc_var, energy_4_bcc_var, energy_5_bcc_var, energy_6_bcc_var], \
               labels=[r"$D_{\text{max}} = 2$", \
                       r"$D_{\text{max}} = 3$", \
                       r"$D_{\text{max}} = 4$", \
                       r"$D_{\text{max}} = 5$", \
                       r"$D_{\text{max}} = 6$"], \
               loc="center left", \
               bbox_to_anchor=(0.9, 0.77), \
               title=r"Variational boundary compression")
    fig.legend(handles=[energy_2_bcc_col, energy_3_bcc_col, energy_4_bcc_col, energy_5_bcc_col, energy_6_bcc_col], \
               labels=[r"$D_{\text{max}} = 2$", \
                       r"$D_{\text{max}} = 3$", \
                       r"$D_{\text{max}} = 4$", \
                       r"$D_{\text{max}} = 5$", \
                       r"$D_{\text{max}} = 6$"], \
               loc="center left", \
               bbox_to_anchor=(0.9, 0.55), \
               title=r"New bulk-weighted boundary compression")
    # save figures
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return


def separate_pos_neg_values(chi_max_bs, deltaEs):
    """Separate positive and negative values of the energy differences deltaEs and return them 
    together with the corresponding bond dimensions chi_max_bs."""
    assert len(chi_max_bs) == len(deltaEs)
    chi_max_bs = np.array(chi_max_bs)
    deltaEs = np.array(deltaEs)
    inds_pos = deltaEs > 0
    inds_neg = deltaEs <= 0
    return chi_max_bs[inds_pos], deltaEs[inds_pos], chi_max_bs[inds_neg], deltaEs[inds_neg]