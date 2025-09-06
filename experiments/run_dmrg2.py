import sys
from pathlib import Path
import pickle
import cProfile

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..src.isometric_peps.a_iso_peps.src.isoTPS.square.isoTPS import isoTPS_Square as DiagonalIsometricPEPS
from ..src.isometric_peps.b_model import TFIModelDiagonalSquare
from ..src.isometric_peps.d_expectation_values import subtract_energy_offset_mpos
from ..src.isometric_peps.g_dmrg2 import DMRGSquared
from ..src.isometric_peps.h_excitations2_overlap import get_overlap_wavefunction_iso_peps


def run_dmrg2(Lx, Ly, g, D_max, chi_max_c, chi_max_b, N_sweeps_c, N_sweeps_b, N_sweeps, zeroE, \
              profile=False, seed=0):
    """Initialize an iso_peps with all spins up (plus random perturbations) and perform N_sweeps 
    DMRG^2 sweeps to find the ground state of the TFI model with transverse field g. Redirect prints 
    to log file and safe iso_peps and energies after each sweep in pkl file."""
    # exact ground state energies (from exact diagonalization or extrapolated 1d DMRG)
    tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
    E0 = None
    psi0 = None
    if 2*Lx*Ly <= 20:
        H = tfi_model.get_H()
        E0, psi0 = sparse.linalg.eigsh(H, k=1, which="SA")
        E0 = E0[0]
        psi0 = psi0[:, 0]
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
    elif g == 3.0:
        if Lx == Ly == 6:
            E0 = -227.308603832
    # unique and reproducible seed
    np.random.seed((seed + Lx + 100 * Ly) % (2**32))
    # maximal column bond dimension
    if chi_max_c == "6_D_max":
        chi_max_c = 6*D_max
    # for use on cluster: str(None) -> None
    if N_sweeps_b == "None":
        N_sweeps_b = None
    # log, pkl and profile path
    script_path = Path(__file__).resolve().parent
    file_base = f"dmrg_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{chi_max_b}_{N_sweeps_c}_{N_sweeps_b}_{N_sweeps}"
    log_path = script_path.parent / "data" / "dmrg2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "dmrg2" / f"{file_base}.pkl"
    if profile:
        profile_path = script_path.parent / "data" / "dmrg2" / f"{file_base}.pstat"
        profiler = cProfile.Profile()
        profiler.enable()
    # dmrg2 sweeps
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}, " \
              + f"D_max = {D_max}, chi_max_c = {chi_max_c}, chi_max_b = {chi_max_b}. \n")
        if E0 is not None:
            print(f"E0_exact = {E0}. \n")
            eps = 1./np.abs(E0)
        else:
            eps = 1./(2 * Lx * Ly)
        iso_peps = DiagonalIsometricPEPS.from_perturbed_qubit_product_state(Lx, Ly, D_max, chi_max_c, \
                                                                            spin_orientation="up", \
                                                                            eps=eps)
        """
        iso_peps = DiagonalIsometricPEPS.from_random_state(Lx, Ly, D_max, chi_max_c)
        file_base_tebd = f"tebd_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{0.05}_{100}"
        pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
        with open(pkl_path_tebd, "rb") as pkl_file_tebd:
            iso_peps = pickle.load(pkl_file_tebd)
        iso_peps_copy = DiagonalIsometricPEPS(iso_peps.Lx, iso_peps.Ly, D_max=iso_peps.D_max, \
                                              chi_factor=iso_peps.chi_factor, chi_max=iso_peps.chi_max, \
                                              d=iso_peps.d, shifting_options=iso_peps.shifting_options, \
                                              yb_options=iso_peps.yb_options, \
                                              tebd_options=iso_peps.tebd_options)
        iso_peps_copy._init_as_copy(iso_peps)
        iso_peps = iso_peps_copy
        iso_peps.move_orthogonality_column_to(0)
        """
        h_mpos = tfi_model.get_h_mpos()
        es = iso_peps.copy().get_column_expectation_values(h_mpos)
        E = np.sum(es)
        print(f"es = {es}.")
        print(f"E = {E}.")
        if E0 is not None:
            print(f"E - E0 = {E - E0}.")
        if zeroE:
            assert E0 is not None
            es_subtract = [(es[n]/E)*E0 for n in range(len(es))]
            subtract_energy_offset_mpos(h_mpos, es_subtract)
            print(f"subtract energy E0_exact evenly from all columns.")
        if psi0 is not None:
            overlap = get_overlap_wavefunction_iso_peps(psi0, iso_peps.copy())
            print(f"<iso_peps|psi0> = {overlap}.")
        print("")
        dmrg2 = DMRGSquared(iso_peps, h_mpos, chi_max_b, N_sweeps_c, N_sweeps_b)
        print("")
        iso_peps_list = []
        expectation_value_list = []
        if psi0 is not None:
            overlap_list = []
        for i in range(N_sweeps):
            print(f"DMRGSquared sweep {i+1} ...")
            dmrg2.run(1)
            iso_peps_list.append(dmrg2.psi)
            Es_list = dmrg2.Es
            Es_updated_list = dmrg2.Es_updated
            expectation_value = np.sum(dmrg2.psi.copy().get_column_expectation_values(h_mpos))
            if zeroE:
                Es_list = [E+E0 for E in Es_list]
                Es_updated_list = [E_updated+E0 for E_updated in Es_updated_list]
                expectation_value += E0
            expectation_value_list.append(expectation_value)
            if psi0 is not None:
                overlap = get_overlap_wavefunction_iso_peps(psi0, dmrg2.psi.copy())
                overlap_list.append(overlap)
            if E0 is not None:
                print(f"=> <H> - E0 = {expectation_value-E0}.")
                if psi0 is not None:
                    print(f"=> <iso_peps0|psi0> = {overlap}.")
                print("")
            else:
                print(f"=> <H> = {expectation_value}. \n")
            with open(pkl_path, "wb") as pkl_file:
                if E0 is not None:
                    pickle.dump((E0, Es_list, Es_updated_list, expectation_value_list, iso_peps_list), pkl_file)
                else:
                    pickle.dump((Es_list, Es_updated_list, expectation_value_list, iso_peps_list), pkl_file)
        if E0 is not None:
            print(f"<H>s - E0 = {np.array(expectation_value_list)-E0}")
            if psi0 is not None:
                print(f"<iso_peps0|psi0>s = {np.array(overlap_list)}")
        else:
            print(f"<H>s = {np.array(expectation_value_list)}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    if profile:
        profiler.disable()
        profiler.dump_stats(profile_path)
    return dmrg2.psi


def plot_dmrg2(g, N_sweeps_plot, chi_max_b_2, chi_max_b_3, chi_max_b_4, chi_max_b_5, chi_max_b_6):
    """After having ran test_dmrg2, plot the energies before and after each column dmrg update, and 
    the final expectation value."""
    # parameters
    Lx = 6
    Ly = 6
    Nx = 2*Lx+1 + 2*Lx
    ns_labels = (list(range(2*Lx+1)) + list(reversed(range(2*Lx)))) * N_sweeps_plot
    ns = np.array(range(Nx * N_sweeps_plot))
    ns_before_after = np.repeat(ns, 2)
    # figure 
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    script_path = Path(__file__).resolve().parent
    png_path = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_2.png"
    fig.text(0.01, 0.5, 
             rf"$\text{{DMRG}}^2$ at $g = {g}$ (ii)", 
             va='center', ha='center', rotation='vertical',
             fontsize=11,
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    # $\bf{{Reduced \: bMPS \: bond \: dimension}}$ 
    ax[0].set_title(rf"(a) New bulk-weighted boundary compression")
    ax[0].set_ylabel(r"$\frac{\vert E(n_x)-E_0 \vert}{\vert E_0 \vert}$", fontsize=13)
    ax[1].set_title(rf"(b) Variational boundary compression")
    ax[1].set_ylabel(r"$\frac{\vert E(n_x)-E_0 \vert}{\vert E_0 \vert}$", fontsize=13)
    ax[1].set_xlabel(r"Orthogonality column $n_x$")
    ax[0].set_ylim(0.8*1e-6, 2.5e-1)
    ax[1].set_ylim(0.8*1e-6, 2.5e-1)
    ns_ticks_filtered = []
    ns_labels_filtered = []
    for n in range(Nx * N_sweeps_plot):
        if ns_labels[n]%Lx == 0:
            ns_ticks_filtered.append(n)
            ns_labels_filtered.append(ns_labels[n])
    ax[0].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[1].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ## bulk-weighted compression
    N_sweeps_b = None
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{12}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{18}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{24}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{5}_{30}_{chi_max_b_5}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_6 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{6}_{36}_{chi_max_b_6}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # D_max = 5
    if chi_max_b_5 is not None:
        try:
            with open(pkl_path_5, "rb") as pkl_file_5:
                E0, Es_list_5, Es_updated_list_5, expectation_value_list_5, _ = pickle.load(pkl_file_5)
            Es_before_5 = np.array(Es_list_5[:Nx * N_sweeps_plot])
            Es_after_5 = np.array(Es_updated_list_5[:Nx * N_sweeps_plot])
            Es_before_after_5 = np.ravel(np.column_stack((Es_before_5, Es_after_5)))
        except FileNotFoundError:
            print("No data available for D_max = 5.")
    # D_max = 6
    if chi_max_b_6 is not None:
        try:
            with open(pkl_path_6, "rb") as pkl_file_6:
                E0, Es_list_6, Es_updated_list_6, expectation_value_list_6, _ = pickle.load(pkl_file_6)
            Es_before_6 = np.array(Es_list_6[:Nx * N_sweeps_plot])
            Es_after_6 = np.array(Es_updated_list_6[:Nx * N_sweeps_plot])
            Es_before_after_6 = np.ravel(np.column_stack((Es_before_6, Es_after_6)))
        except FileNotFoundError:
            print("No data available for D_max = 6.")
    # plot energies against orthogonality columns
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[0].semilogy(ns_before_after, np.abs(Es_before_after_2-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[0].semilogy(ns, np.abs(Es_before_2-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[0].semilogy(ns, np.abs(Es_after_2-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[0].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[0].semilogy(ns_before_after, np.abs(Es_before_after_3-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[0].semilogy(ns, np.abs(Es_before_3-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[0].semilogy(ns, np.abs(Es_after_3-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[0].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[0].semilogy(ns_before_after, np.abs(Es_before_after_4-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[0].semilogy(ns, np.abs(Es_before_4-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[0].semilogy(ns, np.abs(Es_after_4-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[0].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # D_max = 5
    if chi_max_b_5 is not None:
        ax[0].semilogy(ns_before_after, np.abs(Es_before_after_5-E0)/np.abs(E0), "-", color="blue")
        before_5, = ax[0].semilogy(ns, np.abs(Es_before_5-E0)/np.abs(E0), ".", color="lightskyblue", markersize=8, label=r"$E(n)$ before column DMRG")
        after_5, = ax[0].semilogy(ns, np.abs(Es_after_5-E0)/np.abs(E0), "x", color="darkblue", label=r"$E(n)$ after column DMRG")
        expectation_5 = ax[0].axhline(y=np.abs(expectation_value_list_5[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="blue", label=r"$\langle H \rangle$ for final state")
    # D_max = 6
    if chi_max_b_6 is not None:
        ax[0].semilogy(ns_before_after, np.abs(Es_before_after_6-E0)/np.abs(E0), "-", color="green")
        before_6, = ax[0].semilogy(ns, np.abs(Es_before_6-E0)/np.abs(E0), ".", color="lightgreen", markersize=8, label=r"$E(n)$ before column DMRG")
        after_6, = ax[0].semilogy(ns, np.abs(Es_after_6-E0)/np.abs(E0), "x", color="darkgreen", label=r"$E(n)$ after column DMRG")
        expectation_6 = ax[0].axhline(y=np.abs(expectation_value_list_6[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="green", label=r"$\langle H \rangle$ for final state")
    ## variational compression
    N_sweeps_b = 3
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{12}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{18}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{24}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{5}_{30}_{chi_max_b_5}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_6 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{6}_{36}_{chi_max_b_6}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # D_max = 5
    if chi_max_b_5 is not None:
        try:
            with open(pkl_path_5, "rb") as pkl_file_5:
                E0, Es_list_5, Es_updated_list_5, expectation_value_list_5, _ = pickle.load(pkl_file_5)
            Es_before_5 = np.array(Es_list_5[:Nx * N_sweeps_plot])
            Es_after_5 = np.array(Es_updated_list_5[:Nx * N_sweeps_plot])
            Es_before_after_5 = np.ravel(np.column_stack((Es_before_5, Es_after_5)))
        except FileNotFoundError:
            print("No data available for D_max = 5.")
    # D_max = 6
    if chi_max_b_6 is not None:
        try:
            with open(pkl_path_6, "rb") as pkl_file_6:
                E0, Es_list_6, Es_updated_list_6, expectation_value_list_6, _ = pickle.load(pkl_file_6)
            Es_before_6 = np.array(Es_list_6[:Nx * N_sweeps_plot])
            Es_after_6 = np.array(Es_updated_list_6[:Nx * N_sweeps_plot])
            Es_before_after_6 = np.ravel(np.column_stack((Es_before_6, Es_after_6)))
        except FileNotFoundError:
            print("No data available for D_max = 6.")
    # plot energies against orthogonality columns
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[1].semilogy(ns_before_after, np.abs(Es_before_after_2-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[1].semilogy(ns, np.abs(Es_before_2-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[1].semilogy(ns, np.abs(Es_after_2-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[1].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[1].semilogy(ns_before_after, np.abs(Es_before_after_3-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[1].semilogy(ns, np.abs(Es_before_3-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[1].semilogy(ns, np.abs(Es_after_3-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[1].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[1].semilogy(ns_before_after, np.abs(Es_before_after_4-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[1].semilogy(ns, np.abs(Es_before_4-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[1].semilogy(ns, np.abs(Es_after_4-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[1].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # D_max = 5
    if chi_max_b_5 is not None:
        ax[1].semilogy(ns_before_after, np.abs(Es_before_after_5-E0)/np.abs(E0), "-", color="blue")
        before_5, = ax[1].semilogy(ns, np.abs(Es_before_5-E0)/np.abs(E0), ".", color="lightskyblue", markersize=8, label=r"$E(n)$ before column DMRG")
        after_5, = ax[1].semilogy(ns, np.abs(Es_after_5-E0)/np.abs(E0), "x", color="darkblue", label=r"$E(n)$ after column DMRG")
        expectation_5 = ax[1].axhline(y=np.abs(expectation_value_list_5[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="blue", label=r"$\langle H \rangle$ for final state")
    # D_max = 6
    if chi_max_b_6 is not None:
        ax[1].semilogy(ns_before_after, np.abs(Es_before_after_6-E0)/np.abs(E0), "-", color="green")
        before_6, = ax[1].semilogy(ns, np.abs(Es_before_6-E0)/np.abs(E0), ".", color="lightgreen", markersize=8, label=r"$E(n)$ before column DMRG")
        after_6, = ax[1].semilogy(ns, np.abs(Es_after_6-E0)/np.abs(E0), "x", color="darkgreen", label=r"$E(n)$ after column DMRG")
        expectation_6 = ax[1].axhline(y=np.abs(expectation_value_list_6[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="green", label=r"$\langle H \rangle$ for final state")
    # legend and save
    # y: -0.03, -0.15; x: 0.12, 0.37, 0.62(3)
    if chi_max_b_2 is not None:
        fig.legend(handles=[before_2, after_2, expectation_2], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.12, -0.03), \
                title=rf"$D_{{max}} = {2}$, $\chi_{{max,c}} = {12}$, $\chi_{{max,b}} = {chi_max_b_2}$")
                # $\mathbf{{\chi_{{max,b}} = {chi_max_b_2}}}$
    if chi_max_b_3 is not None:
        fig.legend(handles=[before_3, after_3, expectation_3], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.37, -0.03), \
                title=rf"$D_{{max}} = {3}$, $\chi_{{max,c}} = {18}$, $\chi_{{max,b}} = {chi_max_b_3}$")
    if chi_max_b_4 is not None:
        fig.legend(handles=[before_4, after_4, expectation_4], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.62, -0.03), \
                title=rf"$D_{{max}} = {4}$, $\chi_{{max,c}} = {24}$, $\chi_{{max,b}} = {chi_max_b_4}$")
    if chi_max_b_5 is not None:
        fig.legend(handles=[before_5, after_5, expectation_5], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.12, -0.15), \
                title=rf"$D_{{max}} = {5}$, $\chi_{{max,c}} = {30}$, $\chi_{{max,b}} = {chi_max_b_5}$")
    if chi_max_b_6 is not None:
        fig.legend(handles=[before_6, after_6, expectation_6], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.37, -0.15), \
                title=rf"$D_{{max}} = {6}$, $\chi_{{max,c}} = {36}$, $\chi_{{max,b}} = {chi_max_b_6}$")
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles=handles)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return


def plot_dmrg2_comparison_column():
    # parameters
    g = 3.5
    Lx = 6
    Ly = 6
    chi_max_b_2 = 24
    chi_max_b_3 = 54
    chi_max_b_4 = 96
    Nx = 2*Lx+1 + 2*Lx
    N_sweeps_plot = 3
    ns_labels = (list(range(2*Lx+1)) + list(reversed(range(2*Lx)))) * N_sweeps_plot
    ns = np.array(range(Nx * N_sweeps_plot))
    ns_before_after = np.repeat(ns, 2)
    # figure 
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    script_path = Path(__file__).resolve().parent
    png_path = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_3.png"
    fig.text(0.01, 0.5, 
             rf"$\text{{DMRG}}^2$ at $g = {g}$ (iii)", 
             va='center', ha='center', rotation='vertical',
             fontsize=11,
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    fig.text(0.5, 0.9, 
             r"(a) New bulk-weighted boundary compression", 
             va='center', ha='center',
             fontsize=12)
    fig.text(0.31, 0.86, 
             r"$\chi_{max,c} = 6 D_{max}$", 
             va='center', ha='center',
             fontsize=11)
    fig.text(0.72, 0.86, 
             r"$\chi_{max,c} = 3 D_{max}$", 
             va='center', ha='center',
             fontsize=11)
    fig.text(0.5, 0.48, 
             r"(b) Variational boundary compression", 
             va='center', ha='center',
             fontsize=12)
    fig.text(0.31, 0.44, 
             r"$\chi_{max,c} = 6 D_{max}$", 
             va='center', ha='center',
             fontsize=11)
    fig.text(0.72, 0.44, 
             r"$\chi_{max,c} = 3 D_{max}$", 
             va='center', ha='center',
             fontsize=11)
    ax[0, 0].set_ylabel(r"$\frac{\vert E(n_x)-E_0 \vert}{\vert E_0 \vert}$", fontsize=13)
    ax[1, 0].set_ylabel(r"$\frac{\vert E(n_x)-E_0 \vert}{\vert E_0 \vert}$", fontsize=13)
    ax[1, 0].set_xlabel(r"Orthogonality column $n_x$")
    ax[1, 1].set_xlabel(r"Orthogonality column $n_x$")
    ax[0, 0].set_ylim(0.8*1e-6, 2.5e-1)
    ax[0, 1].set_ylim(0.8*1e-6, 2.5e-1)
    ax[1, 0].set_ylim(0.8*1e-6, 2.5e-1)
    ax[1, 1].set_ylim(0.8*1e-6, 2.5e-1)
    ax[0, 0].tick_params(labelbottom=False)
    ax[0, 1].tick_params(labelbottom=False)
    ax[0, 1].tick_params(labelleft=False)
    ax[1, 1].tick_params(labelleft=False)
    ns_ticks_filtered = []
    ns_labels_filtered = []
    for n in range(Nx * N_sweeps_plot):
        if ns_labels[n]%Lx == 0:
            ns_ticks_filtered.append(n)
            ns_labels_filtered.append(ns_labels[n])
    ax[0, 0].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[0, 1].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[1, 0].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[1, 1].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ### standard column dimension
    chi_max_c_2 = 12
    chi_max_c_3 = 18
    chi_max_c_4 = 24
    ## bulk-weighted compression
    N_sweeps_b = None
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{chi_max_c_2}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{chi_max_c_3}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{chi_max_c_4}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # plot energies against orthogonality columns
    Nx_start = 37
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[0, 0].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[0, 0].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[0, 0].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    ## variational compression
    N_sweeps_b = 3
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{chi_max_c_2}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{chi_max_c_3}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{chi_max_c_4}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # plot energies against orthogonality columns
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[1, 0].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[1, 0].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[1, 0].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    ### standard column dimension
    chi_max_c_2 = 6
    chi_max_c_3 = 9
    chi_max_c_4 = 12
    ## bulk-weighted compression
    N_sweeps_b = None
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{chi_max_c_2}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{chi_max_c_3}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{chi_max_c_4}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # plot energies against orthogonality columns
    Nx_start = 37
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[0, 1].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[0, 1].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[0, 1].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    ## variational compression
    N_sweeps_b = 3
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{chi_max_c_2}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{chi_max_c_3}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{chi_max_c_4}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # plot energies against orthogonality columns
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[1, 1].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[1, 1].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[1, 1].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # legend and save
    if chi_max_b_2 is not None:
        fig.legend(handles=[before_2, after_2, expectation_2], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.12, -0.03), \
                title=rf"$D_{{max}} = {2}$, $\chi_{{max,b}} = {24}$")
                # $\mathbf{{\chi_{{max,b}} = {chi_max_b_2}}}$
    if chi_max_b_3 is not None:
        fig.legend(handles=[before_3, after_3, expectation_3], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.37, -0.03), \
                title=rf"$D_{{max}} = {3}$, $\chi_{{max,b}} = {54}$")
    if chi_max_b_4 is not None:
        fig.legend(handles=[before_4, after_4, expectation_4], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.62, -0.03), \
                title=rf"$D_{{max}} = {4}$, $\chi_{{max,b}} = {96}$")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return


def plot_dmrg2_comparison_boundary():
    # parameters
    g = 3.0
    Lx = 6
    Ly = 6
    Nx = 2*Lx+1 + 2*Lx
    N_sweeps_plot = 3
    ns_labels = (list(range(2*Lx+1)) + list(reversed(range(2*Lx)))) * N_sweeps_plot
    ns = np.array(range(Nx * N_sweeps_plot))
    ns_before_after = np.repeat(ns, 2)
    # figure 
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    script_path = Path(__file__).resolve().parent
    png_path = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_2.png"
    fig.text(0.01, 0.5, 
             rf"$\text{{DMRG}}^2$ at $g = {g}$ (ii)", 
             va='center', ha='center', rotation='vertical',
             fontsize=11,
             bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    fig.text(0.5, 0.9, 
             r"(a) New bulk-weighted boundary compression", 
             va='center', ha='center',
             fontsize=12)
    fig.text(0.31, 0.86, 
             r"$\chi_{max,b} = 6 D_{max}^2$", 
             va='center', ha='center',
             fontsize=11)
    fig.text(0.72, 0.86, 
             r"$\chi_{max,b} = 3 D_{max}^2$", 
             va='center', ha='center',
             fontsize=11)
    fig.text(0.5, 0.48, 
             r"(b) Variational boundary compression", 
             va='center', ha='center',
             fontsize=12)
    fig.text(0.31, 0.44, 
             r"$\chi_{max,b} = 6 D_{max}^2$", 
             va='center', ha='center',
             fontsize=11)
    fig.text(0.72, 0.44, 
             r"$\chi_{max,b} = 3 D_{max}^2$", 
             va='center', ha='center',
             fontsize=11)
    ax[0, 0].set_ylabel(r"$\frac{\vert E(n_x)-E_0 \vert}{\vert E_0 \vert}$", fontsize=13)
    ax[1, 0].set_ylabel(r"$\frac{\vert E(n_x)-E_0 \vert}{\vert E_0 \vert}$", fontsize=13)
    ax[1, 0].set_xlabel(r"Orthogonality column $n_x$")
    ax[1, 1].set_xlabel(r"Orthogonality column $n_x$")
    ax[0, 0].set_ylim(0.8*1e-6, 2.5e-1)
    ax[0, 1].set_ylim(0.8*1e-6, 2.5e-1)
    ax[1, 0].set_ylim(0.8*1e-6, 2.5e-1)
    ax[1, 1].set_ylim(0.8*1e-6, 2.5e-1)
    ax[0, 0].tick_params(labelbottom=False)
    ax[0, 1].tick_params(labelbottom=False)
    ax[0, 1].tick_params(labelleft=False)
    ax[1, 1].tick_params(labelleft=False)
    ns_ticks_filtered = []
    ns_labels_filtered = []
    for n in range(Nx * N_sweeps_plot):
        if ns_labels[n]%Lx == 0:
            ns_ticks_filtered.append(n)
            ns_labels_filtered.append(ns_labels[n])
    ax[0, 0].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[0, 1].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[1, 0].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ax[1, 1].set_xticks(ns_ticks_filtered, labels=[f"{n}" for n in ns_labels_filtered])
    ### standard boundary dimension
    chi_max_b_2 = 24
    chi_max_b_3 = 54
    chi_max_b_4 = 96
    chi_max_b_5 = 150
    ## bulk-weighted compression
    N_sweeps_b = None
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{12}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{18}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{24}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{5}_{30}_{chi_max_b_5}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # D_max = 5
    if chi_max_b_5 is not None:
        try:
            with open(pkl_path_5, "rb") as pkl_file_5:
                E0, Es_list_5, Es_updated_list_5, expectation_value_list_5, _ = pickle.load(pkl_file_5)
            Es_before_5 = np.array(Es_list_5[:Nx * N_sweeps_plot])
            Es_after_5 = np.array(Es_updated_list_5[:Nx * N_sweeps_plot])
            Es_before_after_5 = np.ravel(np.column_stack((Es_before_5, Es_after_5)))
        except FileNotFoundError:
            print("No data available for D_max = 5.")
    # plot energies against orthogonality columns
    Nx_start = 37
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[0, 0].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[0, 0].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[0, 0].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # D_max = 5
    if chi_max_b_5 is not None:
        ax[0, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_5[2*Nx_start:]-E0)/np.abs(E0), "-", color="blue")
        before_5, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_before_5[Nx_start:]-E0)/np.abs(E0), ".", color="lightskyblue", markersize=8, label=r"$E(n)$ before column DMRG")
        after_5, = ax[0, 0].semilogy(ns[Nx_start:], np.abs(Es_after_5[Nx_start:]-E0)/np.abs(E0), "x", color="darkblue", label=r"$E(n)$ after column DMRG")
        expectation_5 = ax[0, 0].axhline(y=np.abs(expectation_value_list_5[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="blue", label=r"$\langle H \rangle$ for final state")
    ## variational compression
    N_sweeps_b = 3
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{12}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{18}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{24}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{5}_{30}_{chi_max_b_5}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # D_max = 5
    if chi_max_b_5 is not None:
        try:
            with open(pkl_path_5, "rb") as pkl_file_5:
                E0, Es_list_5, Es_updated_list_5, expectation_value_list_5, _ = pickle.load(pkl_file_5)
            Es_before_5 = np.array(Es_list_5[:Nx * N_sweeps_plot])
            Es_after_5 = np.array(Es_updated_list_5[:Nx * N_sweeps_plot])
            Es_before_after_5 = np.ravel(np.column_stack((Es_before_5, Es_after_5)))
        except FileNotFoundError:
            print("No data available for D_max = 5.")
    # plot energies against orthogonality columns
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[1, 0].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[1, 0].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[1, 0].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # D_max = 5
    if chi_max_b_5 is not None:
        ax[1, 0].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_5[2*Nx_start:]-E0)/np.abs(E0), "-", color="blue")
        before_5, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_before_5[Nx_start:]-E0)/np.abs(E0), ".", color="lightskyblue", markersize=8, label=r"$E(n)$ before column DMRG")
        after_5, = ax[1, 0].semilogy(ns[Nx_start:], np.abs(Es_after_5[Nx_start:]-E0)/np.abs(E0), "x", color="darkblue", label=r"$E(n)$ after column DMRG")
        expectation_5 = ax[1, 0].axhline(y=np.abs(expectation_value_list_5[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="blue", label=r"$\langle H \rangle$ for final state")
    ### smaller boundary dimension
    chi_max_b_2 = 12
    chi_max_b_3 = 27
    chi_max_b_4 = 48
    chi_max_b_5 = 75
    ## bulk-weighted compression
    N_sweeps_b = None
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{12}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{18}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{24}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{5}_{30}_{chi_max_b_5}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # D_max = 5
    if chi_max_b_5 is not None:
        try:
            with open(pkl_path_5, "rb") as pkl_file_5:
                E0, Es_list_5, Es_updated_list_5, expectation_value_list_5, _ = pickle.load(pkl_file_5)
            Es_before_5 = np.array(Es_list_5[:Nx * N_sweeps_plot])
            Es_after_5 = np.array(Es_updated_list_5[:Nx * N_sweeps_plot])
            Es_before_after_5 = np.ravel(np.column_stack((Es_before_5, Es_after_5)))
        except FileNotFoundError:
            print("No data available for D_max = 5.")
    # plot energies against orthogonality columns
    Nx_start = 37
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[0, 1].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[0, 1].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[0, 1].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # D_max = 5
    if chi_max_b_5 is not None:
        ax[0, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_5[2*Nx_start:]-E0)/np.abs(E0), "-", color="blue")
        before_5, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_before_5[Nx_start:]-E0)/np.abs(E0), ".", color="lightskyblue", markersize=8, label=r"$E(n)$ before column DMRG")
        after_5, = ax[0, 1].semilogy(ns[Nx_start:], np.abs(Es_after_5[Nx_start:]-E0)/np.abs(E0), "x", color="darkblue", label=r"$E(n)$ after column DMRG")
        expectation_5 = ax[0, 1].axhline(y=np.abs(expectation_value_list_5[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="blue", label=r"$\langle H \rangle$ for final state")
    ## variational compression
    N_sweeps_b = 3
    # load energy data
    pkl_path_2 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{2}_{12}_{chi_max_b_2}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_3 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{3}_{18}_{chi_max_b_3}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_4 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{4}_{24}_{chi_max_b_4}_{3}_{N_sweeps_b}_{3}.pkl"
    pkl_path_5 = script_path.parent / "data" / "dmrg2" / f"dmrg_{Lx}_{Ly}_{g}_{5}_{30}_{chi_max_b_5}_{3}_{N_sweeps_b}_{3}.pkl"
    # D_max = 2
    if chi_max_b_2 is not None:
        try:
            with open(pkl_path_2, "rb") as pkl_file_2:
                E0, Es_list_2, Es_updated_list_2, expectation_value_list_2, _ = pickle.load(pkl_file_2)
            Es_before_2 = np.array(Es_list_2[:Nx * N_sweeps_plot])
            Es_after_2 = np.array(Es_updated_list_2[:Nx * N_sweeps_plot])
            Es_before_after_2 = np.ravel(np.column_stack((Es_before_2, Es_after_2)))
        except FileNotFoundError:
            print("No data available for D_max = 2.")
    # D_max = 3
    if chi_max_b_3 is not None:
        try:
            with open(pkl_path_3, "rb") as pkl_file_3:
                E0, Es_list_3, Es_updated_list_3, expectation_value_list_3, _ = pickle.load(pkl_file_3)
            Es_before_3 = np.array(Es_list_3[:Nx * N_sweeps_plot])
            Es_after_3 = np.array(Es_updated_list_3[:Nx * N_sweeps_plot])
            Es_before_after_3 = np.ravel(np.column_stack((Es_before_3, Es_after_3)))
        except FileNotFoundError:
            print("No data available for D_max = 3.")
    # D_max = 4
    if chi_max_b_4 is not None:
        try:
            with open(pkl_path_4, "rb") as pkl_file_4:
                E0, Es_list_4, Es_updated_list_4, expectation_value_list_4, _ = pickle.load(pkl_file_4)
            Es_before_4 = np.array(Es_list_4[:Nx * N_sweeps_plot])
            Es_after_4 = np.array(Es_updated_list_4[:Nx * N_sweeps_plot])
            Es_before_after_4 = np.ravel(np.column_stack((Es_before_4, Es_after_4)))
        except FileNotFoundError:
            print("No data available for D_max = 4.")
    # D_max = 5
    if chi_max_b_5 is not None:
        try:
            with open(pkl_path_5, "rb") as pkl_file_5:
                E0, Es_list_5, Es_updated_list_5, expectation_value_list_5, _ = pickle.load(pkl_file_5)
            Es_before_5 = np.array(Es_list_5[:Nx * N_sweeps_plot])
            Es_after_5 = np.array(Es_updated_list_5[:Nx * N_sweeps_plot])
            Es_before_after_5 = np.ravel(np.column_stack((Es_before_5, Es_after_5)))
        except FileNotFoundError:
            print("No data available for D_max = 5.")
    # plot energies against orthogonality columns
    # D_max = 2
    if chi_max_b_2 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_2[2*Nx_start:]-E0)/np.abs(E0), "-", color="red")
        before_2, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_2[Nx_start:]-E0)/np.abs(E0), ".", color="lightcoral", markersize=8, label=r"$E(n)$ before column DMRG")
        after_2,  = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_2[Nx_start:]-E0)/np.abs(E0), "x", color="darkred", label=r"$E(n)$ after column DMRG")
        expectation_2 = ax[1, 1].axhline(y=np.abs(expectation_value_list_2[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="red", label=r"$\langle H \rangle$ for final state")
    # D_max = 3
    if chi_max_b_3 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_3[2*Nx_start:]-E0)/np.abs(E0), "-", color="orange")
        before_3, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_3[Nx_start:]-E0)/np.abs(E0), ".", color="sandybrown", markersize=8, label=r"$E(n)$ before column DMRG")
        after_3,  = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_3[Nx_start:]-E0)/np.abs(E0), "x", color="saddlebrown", label=r"$E(n)$ after column DMRG")
        expectation_3 = ax[1, 1].axhline(y=np.abs(expectation_value_list_3[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="orange", label=r"$\langle H \rangle$ for final state")
    # D_max = 4
    if chi_max_b_4 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_4[2*Nx_start:]-E0)/np.abs(E0), "-", color="purple")
        before_4, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_4[Nx_start:]-E0)/np.abs(E0), ".", color="violet", markersize=8, label=r"$E(n)$ before column DMRG")
        after_4, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_4[Nx_start:]-E0)/np.abs(E0), "x", color="indigo", label=r"$E(n)$ after column DMRG")
        expectation_4 = ax[1, 1].axhline(y=np.abs(expectation_value_list_4[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="purple", label=r"$\langle H \rangle$ for final state")
    # D_max = 5
    if chi_max_b_5 is not None:
        ax[1, 1].semilogy(ns_before_after[2*Nx_start:], np.abs(Es_before_after_5[2*Nx_start:]-E0)/np.abs(E0), "-", color="blue")
        before_5, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_before_5[Nx_start:]-E0)/np.abs(E0), ".", color="lightskyblue", markersize=8, label=r"$E(n)$ before column DMRG")
        after_5, = ax[1, 1].semilogy(ns[Nx_start:], np.abs(Es_after_5[Nx_start:]-E0)/np.abs(E0), "x", color="darkblue", label=r"$E(n)$ after column DMRG")
        expectation_5 = ax[1, 1].axhline(y=np.abs(expectation_value_list_5[N_sweeps_plot-1]-E0)/np.abs(E0), linestyle="-", color="blue", label=r"$\langle H \rangle$ for final state")
    # legend and save
    if chi_max_b_2 is not None:
        fig.legend(handles=[before_2, after_2, expectation_2], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.12, -0.03), \
                title=rf"$D_{{max}} = {2}$, $\chi_{{max,c}} = {12}$")
                # $\mathbf{{\chi_{{max,b}} = {chi_max_b_2}}}$
    if chi_max_b_3 is not None:
        fig.legend(handles=[before_3, after_3, expectation_3], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.37, -0.03), \
                title=rf"$D_{{max}} = {3}$, $\chi_{{max,c}} = {18}$")
    if chi_max_b_4 is not None:
        fig.legend(handles=[before_4, after_4, expectation_4], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.62, -0.03), \
                title=rf"$D_{{max}} = {4}$, $\chi_{{max,c}} = {24}$")
    if chi_max_b_5 is not None:
        fig.legend(handles=[before_5, after_5, expectation_5], \
                labels=[r"$E(n_x)$ before column DMRG", r"$E(n_x)$ after column DMRG", r"$\langle H \rangle$ for final state"], \
                loc="center left", \
                bbox_to_anchor=(0.12, -0.15), \
                title=rf"$D_{{max}} = {5}$, $\chi_{{max,c}} = {30}$")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return