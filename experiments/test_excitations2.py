import sys
from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from copy import deepcopy

from ..src.mps.a_mps import MPS
from ..src.mps.b_model_finite import TFIModelFinite
from ..src.mps.c_dmrg import dmrg_algorithm
from ..src.mps.d_excitations import VariationalQuasiparticleExcitationEngine

from ..src.isometric_peps.b_model import DiagonalSquareLattice, TFIModelDiagonalSquare, TFIModelDiagonalSquareTenpy
from ..src.isometric_peps.h_excitations2_overlap import ExcitedIsometricPEPS as ExcitedIsometricPEPSOverlap, \
                                                        get_overlap_wavefunction_iso_peps
from ..src.isometric_peps.i_excitations2 import VariationalQuasiparticleExcitationsEngine, \
                                                ExcitedIsometricPEPS, \
                                                Heff, \
                                                extract_all_isometric_configurations, get_ADs_AUs_ACs
from ..src.isometric_peps.j_excitations2_middle import VariationalQuasiparticleExcitationsEngineMiddle, get_iso_peps_mps_overlap


# PERTURBATION THEORY

def plot_single_particle(L1, L2, L3, g, k, uniform):
    ess_bond_1, Es_1 = TFIModelDiagonalSquare(L1, L1, g).get_es_bond_single_particle(k, uniform)
    print(f"L1 = {L1}: es = {Es_1}.")
    ess_bond_2, Es_2 = TFIModelDiagonalSquare(L2, L2, g).get_es_bond_single_particle(k, uniform)
    print(f"L2 = {L2}: es = {Es_2}.")
    ess_bond_3, Es_3 = TFIModelDiagonalSquare(L3, L3, g).get_es_bond_single_particle(k, uniform)
    print(f"L3 = {L3}: es = {Es_3}.")
    if L3 <= 3:
        H = TFIModelDiagonalSquare(L3, L3, g).get_H()
        Es_exact, psis_exact = sparse.linalg.eigsh(H, k=k+1, which="SA")
        print(f"es_exact = {Es_exact[1:]-Es_exact[0]}.")
    vmin_1 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_1)
    vmax_1 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_1)
    vmin_2 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_2)
    vmax_2 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_2)
    vmin_3 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_3)
    vmax_3 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_3)
    fig, axes = plt.subplots(k, 3, figsize=(3*3, 3*k))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.15, hspace=0.15)
    axes[0, 0].set_title(rf"$\Downarrow L_x = L_y = {L1}$", fontsize=13)
    axes[0, 1].set_title(rf"$\Downarrow L_x = L_y = {L2}$", fontsize=13)
    axes[0, 2].set_title(rf"$\Downarrow L_x = L_y = {L3}$", fontsize=13)
    for i in range(k):
        axes[i, 0].set_ylabel(rf"$\Downarrow$ excitation {i+1}", fontsize=13)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_yticklabels([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_yticklabels([])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_xticklabels([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_yticklabels([])
    for i in range(k):
        ax1 = axes[i, 0]
        im_1 = ax1.imshow(np.real(ess_bond_1[i]).T, origin='lower', vmin=vmin_1, vmax=vmax_1)
        ax1.text(0.5, 0.95, rf"$\epsilon_{{{i+1}}} = {Es_1[i]:.5f}$", transform=ax1.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        ax2 = axes[i, 1]
        im_2 = ax2.imshow(np.real(ess_bond_2[i]).T, origin='lower', vmin=vmin_2, vmax=vmax_2)
        ax2.text(0.5, 0.95, rf"$\epsilon_{{{i+1}}} = {Es_2[i]:.5f}$", transform=ax2.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        ax3 = axes[i, 2]
        im_3 = ax3.imshow(np.real(ess_bond_3[i]).T, origin='lower', vmin=vmin_3, vmax=vmax_3)
        ax3.text(0.5, 0.95, rf"$\epsilon_{{{i+1}}} = {Es_3[i]:.5f}$", transform=ax3.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # save
    script_path = Path(__file__).resolve().parent
    if uniform:
        file_base_png = file_base_png = f"excitations_{g}_single_particle_uniform"
    else:
        file_base_png = file_base_png = f"excitations_{g}_single_particle"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


# MPS

def run_excitations2_mps(Lx, Ly, g, D_max, k):
    """For system sizes Lx and Ly, snake an MPS through the diagonal square lattice (down -> up if
    Lx >= Ly and left -> right if Ly > Lx). For transverse field g and maximal bond dimension D_max, 
    compute the ground state with DMRG and the first k excitations on top. If the number of spins 
    does not exceed 20, compare with exact diagonalization. Also compute the (uniform) local bond
    energies for all states."""
    if Lx >= Ly:
        order = "down_to_up"
    elif Ly > Lx:
        order = "left_to_right"
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max}_{order}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        N = 2 * Lx * Ly
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        # exact diagonalization if at most 20 spins
        if N <= 20:
            print(f"Exact diagonalization:")
            H = tfi_model.get_H()
            H_bonds = tfi_model.get_H_bonds_array()
            Es_exact, psis_exact = sparse.linalg.eigsh(H, k=k+1, which="SA")
            psis_exact = [psis_exact[:, i] for i in range(k+1)]
            print(f"- E{0}_exact = {Es_exact[0]}.")
            for i in range(1, k+1):
                print(f"- E{i}_exact = {Es_exact[i]} (e{i}_exact = {Es_exact[i]-Es_exact[0]}).")
            ess_bond_exact = []
            for i in range(k+1):
                psi = psis_exact[i]
                es_bond = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                for bx in range(2*Lx-1):
                    for by in range(2*Ly-1):
                        es_bond[bx][by] = np.inner(np.conj(psi), H_bonds[bx][by] @ psi)
                ess_bond_exact.append(es_bond)
                print(f"computed es_bond_{i}.")
            print("")
        # mps
        print(f"MPS with D_max = {D_max}:")
        tfi_model_tenpy = TFIModelDiagonalSquareTenpy.initialize(Lx, Ly, g, order)
        Ws = tfi_model_tenpy.get_np_mpo()
        guess_mps0 = MPS.from_desired_bond_dimension(N, D_max)
        E0_mps, mps0, _ = dmrg_algorithm(Ws, guess_mps0, D_max, eps=1.e-15, num_runs=10)
        excitation_engine = VariationalQuasiparticleExcitationEngine(mps0, Ws)
        es, empss = excitation_engine.run(k)
        Es_mps = [E0_mps]
        psis_mps = [mps0]
        if k == 1:
            Es_mps.append(es+E0_mps)
            psis_mps.append(empss)
        elif k > 1:
            for i in range(k):
                Es_mps.append(es[i]+E0_mps)
                psis_mps.append(empss[i])
        ind_sorted = np.argsort(Es_mps)
        Es_mps = [Es_mps[i] for i in ind_sorted]
        psis_mps = [psis_mps[i] for i in ind_sorted]
        # bond energies
        ess_bond_mps = []
        ess_bond_mps_uniform = []
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((psis_mps, Es_mps, ess_bond_mps, ess_bond_mps_uniform), pkl_file)
        print("")
        print("Bond energies.")
        N = 2 * Lx * Ly
        tfi_model = TFIModelFinite(N, g)
        if order == "down_to_up":
            def get_number_NN(n):
                if n == 0 or n == 2*Lx*Ly-1:
                    return 1
                elif (n < Ly) or (n > 2*Lx*Ly-1-Ly) or (n % (2*Ly) == 0) or ((n+1) % (2*Ly) == 0):
                    return 2 
                else:
                    return 4
            for i in range(k+1):
                print(f"- State {i}.")
                mps = psis_mps[i]
                # sum(es_bond) = E
                print("1) sum(es_bond) = E")
                es_bond = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                n = 0
                for nx in range(2*Lx-1):
                    for y in range(Ly):
                        print(f"bond {n}.")
                        m = n + Ly
                        if nx%2 == 0 and y > 0:
                            es_bond[nx][2*y-1] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m-1, g/get_number_NN(n), g/get_number_NN(m-1)))
                        es_bond[nx][2*y] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m, g/get_number_NN(n), g/get_number_NN(m)))
                        if nx%2 == 1 and y < Ly-1:
                            es_bond[nx][2*y+1] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m+1, g/get_number_NN(n), g/get_number_NN(m+1)))
                        n += 1
                assert n == N - Ly
                ess_bond_mps.append(es_bond)
                print(f"=> deltaE = {np.sum(es_bond)-Es_mps[i]}.")
                if N <= 20:
                    deltas = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                    for bx in range(len(es_bond)):
                        for by in range(len(es_bond[0])):
                            deltas[bx][by] = es_bond[bx][by] - ess_bond_exact[i][bx][by]
                    print(np.array(deltas))
                # uniform bond energies
                print("2) uniform bond energies")
                es_bond_uniform = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                n = 0
                for nx in range(2*Lx-1):
                    for y in range(Ly):
                        print(f"bond {n}.")
                        m = n + Ly
                        if nx%2 == 0 and y > 0:
                            es_bond_uniform[nx][2*y-1] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m-1, g/2, g/2))
                        es_bond_uniform[nx][2*y] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m, g/2, g/2))
                        if nx%2 == 1 and y < Ly-1:
                            es_bond_uniform[nx][2*y+1] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m+1, g/2, g/2))
                        n += 1
                assert n == N - Ly
                ess_bond_mps_uniform.append(es_bond_uniform)
                print("")
                with open(pkl_path, "wb") as pkl_file:
                    pickle.dump((psis_mps, Es_mps, ess_bond_mps, ess_bond_mps_uniform), pkl_file)
        elif order == "left_to_right":
            def get_number_NN(n):
                if n == 0 or n == 2*Lx*Ly-1:
                    return 1
                elif (n < Lx) or (n > 2*Lx*Ly-1-Lx) or (n % (2*Lx) == 0) or ((n+1) % (2*Lx) == 0):
                    return 2 
                else:
                    return 4
            for i in range(k+1):
                print(f"- State {i}.")
                mps = psis_mps[i]
                # sum(es_bond) = E
                print("1) sum(es_bond) = E")
                es_bond = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                n = 0
                for ny in range(2*Ly-1):
                    for x in range(Lx):
                        print(f"bond {n}.")
                        m = n + Lx
                        if ny%2 == 0 and x > 0:
                            es_bond[2*x-1][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m-1, g/get_number_NN(n), g/get_number_NN(m-1)))
                        es_bond[2*x][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m, g/get_number_NN(n), g/get_number_NN(m)))
                        if ny%2 == 1 and x < Lx-1:
                            es_bond[2*x+1][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m+1, g/get_number_NN(n), g/get_number_NN(m+1)))
                        n += 1
                assert n == N - Lx
                ess_bond_mps.append(es_bond)
                print(f"=> deltaE = {np.sum(es_bond)-Es_mps[i]}.")
                if N <= 20:
                    deltas = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                    for bx in range(len(es_bond)):
                        for by in range(len(es_bond[0])):
                            deltas[bx][by] = es_bond[bx][by] - ess_bond_exact[i][bx][by]
                    print(np.array(deltas))
                # uniform bond energies
                print("2) uniform bond energies")
                es_bond_uniform = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                n = 0
                for ny in range(2*Ly-1):
                    for x in range(Lx):
                        print(f"bond {n}.")
                        m = n + Lx
                        if ny%2 == 0 and x > 0:
                            es_bond_uniform[2*x-1][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m-1, g/2, g/2))
                        es_bond_uniform[2*x][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m, g/2, g/2))
                        if ny%2 == 1 and x < Lx-1:
                            es_bond_uniform[2*x+1][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, m+1, g/2, g/2))
                        n += 1
                assert n == N - Lx
                ess_bond_mps_uniform.append(es_bond_uniform)
                print("")
                with open(pkl_path, "wb") as pkl_file:
                    pickle.dump((psis_mps, Es_mps, ess_bond_mps, ess_bond_mps_uniform), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations2_mps(Lx, Ly, g, D_max, k, uniform):
    if Lx >= Ly:
        order = "down_to_up"
    elif Ly > Lx:
        order = "left_to_right"
    # load data for mps
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max}_{order}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_mps, ess_bond_mps, ess_bond_mps_uniform = pickle.load(pkl_file)
    # restrict data to first k excitations and subtract ground state energy
    if uniform:
        ess_bond_mps = ess_bond_mps_uniform
    ess_bond_mps = ess_bond_mps[:(k+1)]
    es_bond_vac = deepcopy(ess_bond_mps[0])
    for i in range(k+1):
        for bx in range(len(es_bond_vac)):
            for by in range(len(es_bond_vac[0])):
                ess_bond_mps[i][bx][by] -= es_bond_vac[bx][by]
    ess_bond_mps = ess_bond_mps[1:]
    es_mps = [E_mps-Es_mps[0] for E_mps in Es_mps[1:]]
    # compare to perturbation theory
    ess_bond_pert, es_pert = TFIModelDiagonalSquare(Lx, Lx, g).get_es_bond_single_particle(k, uniform)
    # combine all bond energy data for global color scalinge
    ess_bond = ess_bond_mps + ess_bond_pert
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    # plot
    fig, axes = plt.subplots(k, 2, figsize=(7, k*3))
    fig.subplots_adjust(top=0.94)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    axes[0, 0].set_title(r"$\Downarrow$ MPS", fontsize=13)
    axes[0, 1].set_title(r"$\Downarrow$ Perturbation theory", fontsize=13)
    for i in range(k):
        axes[i, 0].set_ylabel(rf"$\Downarrow$ excitation {i+1}", fontsize=13)
    # mps
    for i in range(k):
        ax = axes[i, 0]
        im = ax.imshow(np.real(ess_bond_mps[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$e_{{{i+1}}} = {np.real(es_mps[i]):.4f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # perturbation theory
    for i in range(k):
        ax = axes[i, 1]
        im = ax.imshow(np.real(ess_bond_pert[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$e_{{{i+1}}} = {np.real(es_pert[i]):.4f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.03)
    # save
    if uniform:
        file_base_png = f"excitations_{Lx}_{Ly}_{g}_mps_uniform"
    else:
        file_base_png = f"excitations_{Lx}_{Ly}_{g}_mps"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


# OVERLAP WITH MPS

def run_excitations2_overlap_mps(Lx, Ly, g, D_max_mps, state_indices, D_max_iso, chi_max_b, nx):
    """For system sizes Lx, Ly and transverse field g, excite the isoPEPS ground state (received 
    from TEBD^2 or DMRG^2 with D_max_iso) by optimizing the overlap with the excited MPS of maximal 
    bond dimension D_max_mps. Do this for the state_indices excitations on top of the ground state. Also 
    compute the (uniform) bond energies for all states."""
    if Lx >= Ly:
        order = "down_to_up"
    elif Ly > Lx:
        order = "left_to_right"
    chi_max_c = 6 * D_max_iso
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max_iso}_{chi_max_c}_{chi_max_b}_{nx}_overlap_mps_{D_max_mps}_{state_indices}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        # define model and (bond) Hamiltonians
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        h_mpos = tfi_model.get_h_mpos()
        h_bonds = tfi_model.get_h_bonds()
        h_bonds_uniform = tfi_model.get_h_bonds_uniform()
        # load mps
        print(f"MPS with D_max = {D_max_mps}:")
        file_base_mps = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}_{order}"
        pkl_path_mps = script_path.parent / "data" / "excitations2" / f"{file_base_mps}.pkl"
        with open(pkl_path_mps, "rb") as pkl_file:
            psis_mps, Es_mps, _, _ = pickle.load(pkl_file)
        for i in [0] + state_indices:
            print(f"- E{i}_mps = {Es_mps[i]}.")
        print("")
        # load iso_peps ground state
        print(f"Isometric PEPS with D_max = {D_max_iso}, chi_max_c = {chi_max_c}:")
        file_base_dmrg = f"dmrg_{Lx}_{Ly}_{g}_{D_max_iso}_{chi_max_c}_{6*(D_max_iso**2)}_{3}_None_{2}_onesite"
        pkl_path_dmrg = script_path.parent / "data" / "dmrg2" / f"{file_base_dmrg}.pkl"
        with open(pkl_path_dmrg, "rb") as pkl_file:
            _, _, _, _, iso_peps_list = pickle.load(pkl_file)
            iso_peps0 = iso_peps_list[-1]
        ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
        ADs, AUs, ACs = get_ADs_AUs_ACs(ALs, CDs, CUs, CCs)
        E0_iso = np.sum(iso_peps0.copy().get_column_expectation_values(h_mpos))
        print(f"E0_iso = {E0_iso}, deltaE0 = {E0_iso - Es_mps[0]}.")
        es_bond0_list = iso_peps0.copy().get_bond_expectation_values(h_bonds)
        es_bond0 = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0[bx][by] = es_bond0_list[n]
        es_bond0_uniform_list = iso_peps0.copy().get_bond_expectation_values(h_bonds_uniform)
        es_bond0_uniform = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0_uniform[bx][by] = es_bond0_uniform_list[n]
        overlap0 = get_iso_peps_mps_overlap(psis_mps[0].ARs, ALs, ARs, ADs, AUs, ACs)
        print(f"=> |<mps{0}|iso_peps{0}>| = {np.abs(overlap0)}. \n")
        psis_iso = [iso_peps0]
        Es_iso = [E0_iso]
        ess_bond_iso = [es_bond0]
        ess_bond_iso_uniform = [es_bond0_uniform]
        overlaps = [overlap0]
        # compute iso_peps excitations
        bc = "variational"
        engine = VariationalQuasiparticleExcitationsEngineMiddle(D_max_iso, chi_max_c, ALs, ARs, CDs, CCs, CUs, h_mpos, bc, chi_max_b, nx)
        engine.initialize_compressed_hamiltonian_boundaries()
        for i in state_indices:
            print(f"Excitation {i}:")
            engine_copy = engine.copy()
            engine_copy.initialize_excitations_from_emps_overlap(psis_mps[i])
            psis_iso.append(engine_copy)
            overlaps.append(np.inner(np.conj(engine_copy.vecX), engine_copy.vecX))
            engine_copy.print_all_excitation_norms()
            E_iso = np.real_if_close(np.inner(np.conj(engine_copy.vecX), Heff(engine_copy)._matvec(engine_copy.vecX)))
            Es_iso.append(E_iso)
            print(f"=> E{i}_iso = {E_iso}, deltaE{i} = {E_iso - Es_mps[i]}.")
            engine_copy.initialize_compressed_excitation_boundaries()
            es_bond_iso = engine_copy.get_bond_expectation_values(h_bonds)
            ess_bond_iso.append(es_bond_iso)
            print(f"=> sum(es_bond) - E{i}_iso = {np.real_if_close(np.sum(es_bond_iso)-E_iso)}.")
            es_bond_iso_uniform = engine_copy.get_bond_expectation_values(h_bonds_uniform)
            ess_bond_iso_uniform.append(es_bond_iso_uniform)
            print("")
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((psis_iso, overlaps, Es_iso, ess_bond_iso, ess_bond_iso_uniform), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations2_overlap_mps(Lx, Ly, g, D_max_mps, chi_max_b_2, chi_max_b_3, chi_max_b_4, nx, state_indices, uniform=False):
    if Lx >= Ly:
        order = "down_to_up"
    elif Ly > Lx:
        order = "left_to_right"
    # load data for mps
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}_{order}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_mps, ess_bond_mps, ess_bond_mps_uniform = pickle.load(pkl_file)
    # load data for D_max = 2
    D_max = 2
    chi_max_c = 12
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{chi_max_b_2}_{nx}_overlap_mps_{D_max_mps}_{state_indices}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, overlaps_2, Es_iso_2, ess_bond_iso_2, ess_bond_iso_uniform_2 = pickle.load(pkl_file)
    # load data for D_max = 3
    D_max = 3
    chi_max_c = 18
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{chi_max_b_3}_{nx}_overlap_mps_{D_max_mps}_{state_indices}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, overlaps_3, Es_iso_3, ess_bond_iso_3, ess_bond_iso_uniform_3 = pickle.load(pkl_file)
    # load data for D_max = 4
    D_max = 4
    chi_max_c = 24
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{chi_max_b_4}_{nx}_overlap_mps_{D_max_mps}_{state_indices}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, overlaps_4, Es_iso_4, ess_bond_iso_4, ess_bond_iso_uniform_4 = pickle.load(pkl_file)
    if uniform:
        ess_bond_mps = ess_bond_mps_uniform
        ess_bond_iso_2 = ess_bond_iso_uniform_2
        ess_bond_iso_3 = ess_bond_iso_uniform_3
        ess_bond_iso_4 = ess_bond_iso_uniform_4
    k = len(ess_bond_iso_2)
    # restrict mps data to state_indices
    state_indices_complete = [0] + state_indices
    ess_bond_mps = [ess_bond_mps[index] for index in state_indices_complete]
    # subtract mps ground state energies
    es_bond_vac = deepcopy(ess_bond_mps[0])
    for i in range(k):
        for bx in range(len(es_bond_vac)):
            for by in range(len(es_bond_vac[0])):
                ess_bond_mps[i][bx][by] -= es_bond_vac[bx][by]
                ess_bond_iso_2[i][bx][by] -= es_bond_vac[bx][by]
                ess_bond_iso_3[i][bx][by] -= es_bond_vac[bx][by]
                ess_bond_iso_4[i][bx][by] -= es_bond_vac[bx][by]
    # combine all bond energy data for global color scaling
    ess_bond = ess_bond_mps + ess_bond_iso_2 + ess_bond_iso_3 + ess_bond_iso_4
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    # plot
    fig, axes = plt.subplots(k, 4, figsize=(4*3, k*3))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.15, hspace=0.15)
    axes[0, 0].set_title(r"$\Downarrow$ MPS", fontsize=13)
    axes[0, 1].set_title(r"$\Downarrow D_{\text{max}} = 2$", fontsize=13)
    axes[0, 2].set_title(r"$\Downarrow D_{\text{max}} = 3$", fontsize=13)
    axes[0, 3].set_title(r"$\Downarrow D_{\text{max}} = 4$", fontsize=13)
    for i in range(k):
        if i == 0:
            axes[i, 0].set_ylabel(r"$\Downarrow$ GS", fontsize=13)
        else:
            axes[i, 0].set_ylabel(rf"$\Downarrow$ excitation {state_indices_complete[i]}", fontsize=13)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    # mps
    cmap = "viridis"  # viridis, plasma, inferno, magma, cividis
    for i in range(k):
        ax = axes[i, 0]
        im = ax.imshow(np.real(ess_bond_mps[i]).T, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.text(0.5, 0.95, rf"$\vert \psi_{{{state_indices_complete[i]}}} \rangle$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # D_max = 2
    for i in range(k):
        ax = axes[i, 1]
        im = ax.imshow(np.real(ess_bond_iso_2[i]).T, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.text(0.5, 0.95, rf"${np.abs(overlaps_2[i]):.3f}\vert \psi_{{{state_indices_complete[i]}}} \rangle$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # D_max = 3
    for i in range(k):
        ax = axes[i, 2]
        im = ax.imshow(np.real(ess_bond_iso_3[i]).T, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.text(0.5, 0.95, rf"${np.abs(overlaps_3[i]):.4f}\vert \psi_{{{state_indices_complete[i]}}} \rangle$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # D_max = 4
    for i in range(k):
        ax = axes[i, 3]
        im = ax.imshow(np.real(ess_bond_iso_4[i]).T, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.text(0.5, 0.95, rf"${np.abs(overlaps_4[i]):.5f}\vert \psi_{{{state_indices_complete[i]}}} \rangle$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.03)
    # save
    if uniform:
        file_base_png = f"excitations_{Lx}_{Ly}_{g}_overlap_mps_{nx}_{state_indices}_uniform"
    else:
        file_base_png = f"excitations_{Lx}_{Ly}_{g}_overlap_mps_{nx}_{state_indices}"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


# EFFECTIVE HAMILTONIAN

def run_excitations2_effective(Lx, Ly, g, D_max, k, bc, chi_max_b_1, chi_max_b_2, nx, D_max_mps=None, eps_b=1.e-15):
    """For system sizes Lx, Ly and transverse field g, excite the ground state (received from TEBD^2 
    or DMRG^2 with D_max) by diagonalizing the effective Hamiltonian. To compress the boundaries,
    use bc "variational" or "column", chi_max_b_1 and eps_b. Do this for the first k excitations on 
    top of the ground state. Also compute the bond energies for all states, with chi_max_b_2."""
    chi_max_c = 6 * D_max
    if D_max_mps == "None":
        D_max_mps = None
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b_1}_{chi_max_b_2}_{nx}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        # define model and (bond) Hamiltonians
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        h_mpos = tfi_model.get_h_mpos()
        h_bonds = tfi_model.get_h_bonds()
        h_bonds_uniform = tfi_model.get_h_bonds_uniform()
        # load mps for reference
        if D_max_mps is not None:
            if Lx >= Ly:
                order = "down_to_up"
            elif Ly > Lx:
                order = "left_to_right"
            print(f"MPS with D_max = {D_max_mps}:")
            file_base_mps = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}_{order}"
            pkl_path_mps = script_path.parent / "data" / "excitations2" / f"{file_base_mps}.pkl"
            with open(pkl_path_mps, "rb") as pkl_file:
                _, Es_mps, _, _ = pickle.load(pkl_file)
            for i in range(k+1):
                print(f"- E{i}_mps = {Es_mps[i]}.")
            print("")
        # load iso_peps ground state
        print(f"Isometric PEPS with D_max = {D_max}, chi_max_c = {chi_max_c}:")
        file_base_dmrg = f"dmrg_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{6*(D_max**2)}_{3}_None_{2}_onesite"
        pkl_path_dmrg = script_path.parent / "data" / "dmrg2" / f"{file_base_dmrg}.pkl"
        with open(pkl_path_dmrg, "rb") as pkl_file:
            _, _, _, _, iso_peps_list = pickle.load(pkl_file)
            iso_peps0 = iso_peps_list[-1]
        E0_iso = np.sum(iso_peps0.copy().get_column_expectation_values(h_mpos))
        if D_max_mps is None:
            print(f"- E0_iso = {E0_iso}.")
        else:
            print(f"- E0_iso = {E0_iso}, deltaE0 = {E0_iso-Es_mps[0]}.")
        es_bond0_list = iso_peps0.copy().get_bond_expectation_values(h_bonds)
        es_bond0 = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0[bx][by] = es_bond0_list[n]
        es_bond0_uniform_list = iso_peps0.copy().get_bond_expectation_values(h_bonds_uniform)
        es_bond0_uniform = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0_uniform[bx][by] = es_bond0_uniform_list[n]
        psis_iso = [iso_peps0]
        Es_iso = [E0_iso]
        ess_bond_iso = [es_bond0]
        ess_bond_iso_uniform = [es_bond0_uniform]
        # compute iso_peps excitations
        ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
        engine = VariationalQuasiparticleExcitationsEngineMiddle(D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, h_mpos, bc, chi_max_b_1, nx)
        engine.initialize_compressed_hamiltonian_boundaries()
        Es, vecXs = engine.run(k, N=None)
        for i in range(1, k+1):
            Es_iso.append(Es[i-1])
            if D_max_mps is None:
                print(f"- E{i}_iso = {Es_iso[i]}.")
            else:
                print(f"- E{i}_iso = {Es_iso[i]}, deltaE{i} = {Es_iso[i]-Es_mps[i]}.")
        print("")
        engine.chi_max_b = chi_max_b_2
        for i in range(1, k+1):
            engine_copy = engine.copy()
            engine_copy.initialize_excitations(vecXs[i-1])
            psis_iso.append(engine_copy)
            engine_copy.initialize_compressed_excitation_boundaries()
            es_bond_iso = engine_copy.get_bond_expectation_values(h_bonds)
            ess_bond_iso.append(es_bond_iso)
            print(f"=> deltaE{i} = {np.sum(es_bond_iso)-Es_iso[i]}.")
            es_bond_iso_uniform = engine_copy.get_bond_expectation_values(h_bonds_uniform)
            ess_bond_iso_uniform.append(es_bond_iso_uniform)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((psis_iso, Es_iso, ess_bond_iso, ess_bond_iso_uniform), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations2_effective(Lx, Ly, g, D_max_mps, k, nx, uniform=None):
    if Lx >= Ly:
        order = "down_to_up"
    elif Ly > Lx:
        order = "left_to_right"
    # global parameters
    bc = "variational"
    # load data for mps
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}_{order}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_mps, ess_bond_mps, ess_bond_mps_uniform = pickle.load(pkl_file)
    # load data for D_max = 2
    D_max = 2
    chi_max_c = 12
    chi_max_b_1 = 24
    chi_max_b_2 = 96
    #chi_max_b_1 = chi_max_b_2 = D_max**4
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b_1}_{chi_max_b_2}_{nx}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_2, ess_bond_iso_2, ess_bond_iso_uniform_2 = pickle.load(pkl_file)
    # load data for D_max = 3
    D_max = 3
    chi_max_c = 18
    chi_max_b_1 = 54
    chi_max_b_2 = 150
    #chi_max_b_1 = chi_max_b_2 = D_max**4
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b_1}_{chi_max_b_2}_{nx}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_3, ess_bond_iso_3, ess_bond_iso_uniform_3 = pickle.load(pkl_file)
    # load data for D_max = 4
    D_max = 4
    chi_max_c = 24
    chi_max_b_1 = 96
    chi_max_b_2 = 216
    #chi_max_b_1 = chi_max_b_2 = D_max**4
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b_1}_{chi_max_b_2}_{nx}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_4, ess_bond_iso_4, ess_bond_iso_uniform_4 = pickle.load(pkl_file)
    # restrict data to ground state and first k excitations
    if uniform:
        ess_bond_mps = ess_bond_mps_uniform
        ess_bond_iso_2 = ess_bond_iso_uniform_2
        ess_bond_iso_3 = ess_bond_iso_uniform_3
        ess_bond_iso_4 = ess_bond_iso_uniform_4
    Es_mps = Es_mps[:(k+1)]
    ess_bond_mps = ess_bond_mps[:(k+1)]
    Es_iso_2 = Es_iso_2[:(k+1)]
    ess_bond_iso_2 = ess_bond_iso_2[:(k+1)]
    Es_iso_3 = Es_iso_3[:(k+1)]
    ess_bond_iso_3 = ess_bond_iso_3[:(k+1)]
    Es_iso_4 = Es_iso_4[:(k+1)]
    ess_bond_iso_4 = ess_bond_iso_4[:(k+1)]
    # subtract mps ground state energies
    es_bond_vac = deepcopy(ess_bond_mps[0])
    for i in range(k+1):
        for bx in range(len(es_bond_vac)):
            for by in range(len(es_bond_vac[0])):
                ess_bond_mps[i][bx][by] -= es_bond_vac[bx][by]
                ess_bond_iso_2[i][bx][by] -= es_bond_vac[bx][by]
                ess_bond_iso_3[i][bx][by] -= es_bond_vac[bx][by]
                ess_bond_iso_4[i][bx][by] -= es_bond_vac[bx][by]
    # combine all bond energy data for global color scaling
    ess_bond = ess_bond_mps + ess_bond_iso_2 + ess_bond_iso_3 + ess_bond_iso_4
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    # plot
    fig, axes = plt.subplots(k+1, 4, figsize=(4*3, (k+1)*3))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.15, hspace=0.15)
    axes[0, 0].set_title(r"$\Downarrow$ MPS", fontsize=13)
    axes[0, 1].set_title(r"$\Downarrow D_{\text{max}} = 2$", fontsize=13)
    axes[0, 2].set_title(r"$\Downarrow D_{\text{max}} = 3$", fontsize=13)
    axes[0, 3].set_title(r"$\Downarrow D_{\text{max}} = 4$", fontsize=13)
    for i in range(k+1):
        if i == 0:
            axes[i, 0].set_ylabel(r"$\Downarrow$ GS", fontsize=13)
        else:
            axes[i, 0].set_ylabel(rf"$\Downarrow$ excitation {i}", fontsize=13)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    # mps
    for i in range(k+1):
        ax = axes[i, 0] 
        im = ax.imshow(np.real(ess_bond_mps[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$E_{{{i}}} = {np.real(Es_mps[i]):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # D_max = 2
    for i in range(k+1):
        ax = axes[i, 1]
        im = ax.imshow(np.real(ess_bond_iso_2[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E_{{{i}}} = {(np.real(Es_iso_2[i] - Es_mps[i])):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # D_max = 3
    for i in range(k+1):
        ax = axes[i, 2]
        im = ax.imshow(np.real(ess_bond_iso_3[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E_{{{i}}} = {(np.real(Es_iso_3[i] - Es_mps[i])):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    # D_max = 4
    for i in range(k+1):
        ax = axes[i, 3]
        im = ax.imshow(np.real(ess_bond_iso_4[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E_{{{i}}} = {(np.real(Es_iso_4[i] - Es_mps[i])):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.03)
    # save
    if uniform:
        file_base_png = f"excitations_{Lx}_{Ly}_{g}_effective_{nx}_uniform"
    else:
        file_base_png = f"excitations_{Lx}_{Ly}_{g}_effective_{nx}"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")




def run_excitations2_overlap_wavefunction(Lx, Ly, g, D_max, k):
    """For system sizes Lx, Ly and transverse field g, excite the isoPEPS ground state (received 
    from TEBD^2 or DMRG^2 with D_max) by optimizing the overlap with the exact wavefunction. Do this 
    for the first k excitations on top of the ground state."""
    chi_max_c = 6 * D_max
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_wavefunction"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        N = 2 * Lx * Ly
        assert N <= 20, "No more than 20 spins for exact diagonalization!"
        # model and (bond) Hamiltonians
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        H = tfi_model.get_H()
        H_bonds = tfi_model.get_H_bonds_array()
        h_mpos = tfi_model.get_h_mpos()
        h_bonds = tfi_model.get_h_bonds()
        # exact diagonalization
        print(f"Exact diagonalization:")
        Es_exact, psis_exact = sparse.linalg.eigsh(H, k=k+1, which="SA")
        psis_exact = [psis_exact[:, i] for i in range(k+1)]
        ess_bond_exact = []
        for i in range(k+1):
            print(f"- E{i}_exact = {Es_exact[i]}.")
            psi = psis_exact[i]
            es_bond = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
            for bx in range(2*Lx-1):
                for by in range(2*Ly-1):
                    es_bond[bx][by] = np.inner(np.conj(psi), H_bonds[bx][by] @ psi)
            ess_bond_exact.append(es_bond)
        print("")
        # iso_peps ground state
        if D_max == 2:
            chi_max_b = 16
        elif D_max == 3:
            chi_max_b = 81
        elif D_max == 4:
            chi_max_b = 256
        elif D_max == 6:
            chi_max_b = 800
        print(f"Isometric PEPS with D_max = {D_max}, chi_max_c = {chi_max_c} (from DMRG^2 with chi_max_b = {chi_max_b}):")
        file_base_dmrg = f"dmrg_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{chi_max_b}_{3}_{3}_{3}"
        pkl_path_dmrg = script_path.parent / "data" / "dmrg2" / f"{file_base_dmrg}.pkl"
        with open(pkl_path_dmrg, "rb") as pkl_file:
            _, _, _, _, iso_peps_list = pickle.load(pkl_file)
            iso_peps0 = iso_peps_list[-1]
        overlap0 = get_overlap_wavefunction_iso_peps(psis_exact[0], iso_peps0)
        print(f"=> |<psi{0}_exact|iso_peps{0}>| = {np.abs(overlap0)}.")
        E0_iso = np.sum(iso_peps0.copy().get_column_expectation_values(h_mpos))
        print(f"=> E0_iso = {E0_iso}.")
        es_bond0_list = iso_peps0.copy().get_bond_expectation_values(h_bonds)
        es_bond0 = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0[bx][by] = es_bond0_list[n]
        psis_iso = [iso_peps0]
        Es_iso = [E0_iso]
        overlaps = [overlap0]
        ess_bond_iso = [es_bond0]
        print("")
        # iso_peps excitations
        for i in range(1, k+1):
            print(f"Excitation {i}:")
            excited_iso_peps = ExcitedIsometricPEPSOverlap.optimized_from_excited_wavefunction(D_max_iso, chi_max_c, ALs, ARs, CDs, CCs, CUs, psis_exact[i])
            psis_iso.append(excited_iso_peps)
            total_overlap2 = 0.
            for j in range(k+1):
                overlap = excited_iso_peps.get_overlap_with_excited_wavefunction(psis_exact[j])
                print(f"=> |<psi{j}_exact|iso_peps{i}>| = {np.abs(overlap)}.")
                total_overlap2 += overlap**2
                if j == i:
                    overlaps.append(overlap)
            print(f"sum_j |<psi(j)_exact|iso_peps{i}>|^2 = {total_overlap2}.")
            E = excited_iso_peps.get_energy(H)
            print(f"=> E{i} = {E}. \n")
            Es_iso.append(E)
            es_bond = excited_iso_peps.get_bond_energies(H_bonds)
            ess_bond_iso.append(es_bond)
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((psis_exact, Es_exact, ess_bond_exact, psis_iso, Es_iso, ess_bond_iso, overlaps), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations2_overlap_wavefunction():
    Lx = 3
    Ly = 3
    g = 3.5
    subtract = True
    script_path = Path(__file__).resolve().parent
    # load data for D_max = 2
    D_max = 2
    chi_max_c = 12
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_wavefunction"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, _, ess_bond_exact, _, _, ess_bond_iso_2, overlaps_2 = pickle.load(pkl_file)
    # load data for D_max = 4
    D_max = 4
    chi_max_c = 24
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_wavefunction"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, _, _, _, _, ess_bond_iso_4, overlaps_4 = pickle.load(pkl_file)
    # load data for D_max = 6
    D_max = 6
    chi_max_c = 36
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_wavefunction"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, _, _, _, _, ess_bond_iso_6, overlaps_6 = pickle.load(pkl_file)
    k = len(ess_bond_exact) - 1
    if subtract:
        es_bond_vac = deepcopy(ess_bond_exact[0])
        for i in range(k+1):
            for bx in range(len(es_bond_vac)):
                for by in range(len(es_bond_vac[0])):
                    ess_bond_exact[i][bx][by] -= es_bond_vac[bx][by]
                    ess_bond_iso_2[i][bx][by] -= es_bond_vac[bx][by]
                    ess_bond_iso_4[i][bx][by] -= es_bond_vac[bx][by]
                    ess_bond_iso_6[i][bx][by] = 0
    # combine all bond energy data for global color scaling
    ess_bond = ess_bond_exact + ess_bond_iso_2 + ess_bond_iso_4 + ess_bond_iso_6
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    # plot for ED, 2, 4, 6
    fig, axes = plt.subplots(5, 4, figsize=(4*3, 5*3))
    """
    fig.suptitle(r"isoPEPS excitations from wavefunction overlap", fontsize=15, \
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    """
    fig.text(0.075, 0.525, 
            r"isoPEPS excitations from wavefunction overlap", 
            va='center', ha='center', rotation='vertical',
            fontsize=15,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    fig.subplots_adjust(top=0.94)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(range(2*Lx-1))
            ax.set_xticklabels(range(1, 2*Lx))
            ax.set_yticks(range(2*Ly-1))
            ax.set_yticklabels(range(1, 2*Ly))
    for i in range(5):
        axes[i, 0].set_title(rf"$\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 0].imshow(np.real(ess_bond_exact[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == 4:
            axes[i, 0].set_xlabel(r"$\Uparrow \text{Exact diagonalization}$", fontsize=13)
        if i == 0:
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    for i in range(5):
        overlap = np.abs(overlaps_2[i])
        if i == 0:
            axes[i, 1].set_title(rf"${overlap:.7f}\vert \psi_{{{i}}} \rangle$")
        else:
            axes[i, 1].set_title(rf"${overlap:.4f}\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 1].imshow(np.real(ess_bond_iso_2[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == 4:
            axes[i, 1].set_xlabel(r"$\Uparrow D_{\text{max}} = 2$", fontsize=14)
    for i in range(5):
        overlap = np.abs(overlaps_4[i])
        if i == 0:
            axes[i, 2].set_title(rf"${overlap:.7f}\vert \psi_{{{i}}} \rangle$")
        else:
            axes[i, 2].set_title(rf"${overlap:.4f}\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 2].imshow(np.real(ess_bond_iso_4[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == 4:
            axes[i, 2].set_xlabel(r"$\Uparrow D_{\text{max}} = 4$", fontsize=14)
    for i in range(5):
        overlap = np.abs(overlaps_6[i])
        if i == 0:
            axes[i, 3].set_title(rf"${overlap:.7f}\vert \psi_{{{i}}} \rangle$")
        else:
            axes[i, 3].set_title(rf"${overlap:.4f}\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 3].imshow(np.real(ess_bond_iso_6[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == 4:
            axes[i, 3].set_xlabel(r"$\Uparrow D_{\text{max}} = 6$", fontsize=14)
    file_base_png = f"excitations_{Lx}_{Ly}_{g}_overlap_wavefunction"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


def plot_gs_bond_energies():
    # parameters
    g = 3.5
    L1 = 5
    L2 = 7
    L3 = 10
    # figure
    script_path = Path(__file__).resolve().parent
    file_base_png = f"gs_bond_energies"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig, axes = plt.subplots(3, 6, figsize=(6*3, 3*3))
    fig.subplots_adjust(top=0.94)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    axes[0, 0].set_title(r"$\Downarrow$ MPS", fontsize=13)
    axes[0, 1].set_title(r"$\Downarrow D_{max} = 2$", fontsize=13)
    axes[0, 2].set_title(r"$\Downarrow D_{max} = 3$", fontsize=13)
    axes[0, 3].set_title(r"$\Downarrow D_{max} = 4$", fontsize=13)
    axes[0, 4].set_title(r"$\Downarrow D_{max} = 5$", fontsize=13)
    axes[0, 5].set_title(r"$\Downarrow D_{max} = 6$", fontsize=13)
    axes[0, 0].set_ylabel(rf"$\Downarrow L = {{{L1}}}$", fontsize=13)
    axes[1, 0].set_ylabel(rf"$\Downarrow L = {{{L2}}}$", fontsize=13)
    axes[2, 0].set_ylabel(rf"$\Downarrow L = {{{L3}}}$", fontsize=13)
    # plots
    Ls = [L3]
    for L in Ls:
        tfi_model = TFIModelDiagonalSquare(L, L, g)
        h_bonds = tfi_model.get_h_bonds()
        # load data for mps
        if L == 5:
            D_max_mps = 256
            i = 0
        elif L == 7:
            D_max_mps = 512
            i = 1
        elif L == 10:
            D_max_mps = 512
            i = 2
        file_base = f"excitations_{L}_{L}_{g}_mps_{D_max_mps}"
        pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
        with open(pkl_path, "rb") as pkl_file:
            _, Es_mps, ess_bond_mps = pickle.load(pkl_file)
        E0_mps = Es_mps[0]
        es_bond0_mps = ess_bond_mps[0]
        print(np.array(es_bond0_mps))
        # load data for iso_peps 
        # 2
        D_max_iso = 2
        chi_max_c = 6 * D_max_iso
        file_base_tebd = f"tebd_{L}_{L}_{g}_{D_max_iso}_{chi_max_c}_{0.08}_{10}"
        pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
        with open(pkl_path_tebd, "rb") as pkl_file:
            iso_peps0_2 = pickle.load(pkl_file)
        es_bond0_list_2 = iso_peps0_2.copy().get_bond_expectation_values(h_bonds)
        E0_2 = np.sum(es_bond0_list_2)
        es_bond0_2 = [[None] * (2*L-1) for _ in range(2*L-1)]
        for n in range((2*L-1)*(2*L-1)):
            bx, by = DiagonalSquareLattice(L, L).get_bond_vector(n)
            es_bond0_2[bx][by] = es_bond0_list_2[n]
        print("2 done.")
        # 3
        D_max_iso = 3
        chi_max_c = 6 * D_max_iso
        file_base_tebd = f"tebd_{L}_{L}_{g}_{D_max_iso}_{chi_max_c}_{0.08}_{10}"
        pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
        with open(pkl_path_tebd, "rb") as pkl_file:
            iso_peps0_3 = pickle.load(pkl_file)
        es_bond0_list_3 = iso_peps0_3.copy().get_bond_expectation_values(h_bonds)
        E0_3 = np.sum(es_bond0_list_3)
        es_bond0_3 = [[None] * (2*L-1) for _ in range(2*L-1)]
        for n in range((2*L-1)*(2*L-1)):
            bx, by = DiagonalSquareLattice(L, L).get_bond_vector(n)
            es_bond0_3[bx][by] = es_bond0_list_3[n]
        print("3 done.")
        print(np.array(es_bond0_3))
        """
        # 4
        D_max_iso = 4
        chi_max_c = 6 * D_max_iso
        file_base_tebd = f"tebd_{L}_{L}_{g}_{D_max_iso}_{chi_max_c}_{0.08}_{10}"
        pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
        with open(pkl_path_tebd, "rb") as pkl_file:
            iso_peps0_4 = pickle.load(pkl_file)
        es_bond0_list_4 = iso_peps0_4.copy().get_bond_expectation_values(h_bonds)
        E0_4 = np.sum(es_bond0_list_4)
        es_bond0_4 = [[None] * (2*L-1) for _ in range(2*L-1)]
        for n in range((2*L-1)*(2*L-1)):
            bx, by = DiagonalSquareLattice(L, L).get_bond_vector(n)
            es_bond0_4[bx][by] = es_bond0_list_4[n]
        print("4 done.")
        # 5
        D_max_iso = 5
        chi_max_c = 6 * D_max_iso
        file_base_tebd = f"tebd_{L}_{L}_{g}_{D_max_iso}_{chi_max_c}_{0.08}_{10}"
        pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
        with open(pkl_path_tebd, "rb") as pkl_file:
            iso_peps0_5 = pickle.load(pkl_file)
        es_bond0_list_5 = iso_peps0_5.copy().get_bond_expectation_values(h_bonds)
        E0_5 = np.sum(es_bond0_list_5)
        es_bond0_5 = [[None] * (2*L-1) for _ in range(2*L-1)]
        for n in range((2*L-1)*(2*L-1)):
            bx, by = DiagonalSquareLattice(L, L).get_bond_vector(n)
            es_bond0_5[bx][by] = es_bond0_list_5[n]
        print("5 done.")
        # 6
        D_max_iso = 6
        chi_max_c = 6 * D_max_iso
        file_base_tebd = f"tebd_{L}_{L}_{g}_{D_max_iso}_{chi_max_c}_{0.08}_{10}"
        pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
        with open(pkl_path_tebd, "rb") as pkl_file:
            iso_peps0_6 = pickle.load(pkl_file)
        es_bond0_list_6 = iso_peps0_6.copy().get_bond_expectation_values(h_bonds)
        E0_6 = np.sum(es_bond0_list_6)
        es_bond0_6 = [[None] * (2*L-1) for _ in range(2*L-1)]
        for n in range((2*L-1)*(2*L-1)):
            bx, by = DiagonalSquareLattice(L, L).get_bond_vector(n)
            es_bond0_6[bx][by] = es_bond0_list_6[n]
        print("6 done.")
        """

        es_bond_vac = deepcopy(es_bond0_mps)
        for bx in range(len(es_bond_vac)):
            for by in range(len(es_bond_vac[0])):
                es_bond0_mps[bx][by] -= es_bond_vac[bx][by]
                es_bond0_2[bx][by] -= es_bond_vac[bx][by]
                es_bond0_3[bx][by] -= es_bond_vac[bx][by]
                #es_bond0_4[bx][by] -= es_bond_vac[bx][by]
                #es_bond0_5[bx][by] -= es_bond_vac[bx][by]
                #es_bond0_6[bx][by] -= es_bond_vac[bx][by]

        vmin = np.min(es_bond0_mps + es_bond0_2 + es_bond0_3)
        vmax = np.max(es_bond0_mps + es_bond0_2 + es_bond0_3)
        #vmin = np.min(es_bond0_mps + es_bond0_2 + es_bond0_3 + es_bond0_4 + es_bond0_5 + es_bond0_6)
        #vmax = np.max(es_bond0_mps + es_bond0_2 + es_bond0_3 + es_bond0_4 + es_bond0_5 + es_bond0_6)

        # mps
        ax = axes[i, 0] 
        im = ax.imshow(np.real(es_bond0_mps).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$E_0 = {np.real(E0_mps):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        # D_max = 2
        ax = axes[i, 1]
        im = ax.imshow(np.real(es_bond0_2).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E = {(np.real(E0_2 - E0_mps)):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        # D_max = 3
        """
        ax = axes[i, 2]
        im = ax.imshow(np.real(es_bond0_3).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E = {(np.real(E0_3 - E0_mps)):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        # D_max = 4
        ax = axes[i, 3]
        im = ax.imshow(np.real(es_bond0_4).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E = {(np.real(E0_4 - E0_mps)):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        # D_max = 5
        ax = axes[i, 4]
        im = ax.imshow(np.real(es_bond0_5).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E = {(np.real(E0_5 - E0_mps)):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        # D_max = 6
        ax = axes[i, 5]
        im = ax.imshow(np.real(es_bond0_6).T, origin='lower', vmin=vmin, vmax=vmax)
        ax.text(0.5, 0.95, rf"$\Delta E = {(np.real(E0_6 - E0_mps)):.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="white", weight="bold")
        """
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.047, pad=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


def test_energy_compression_convergence(Lx, Ly, g, D_max, bc, chi_max_bs, D_max_mps=None, eps_b=1.e-15):
    if D_max_mps == "None":
        D_max_mps = None
    chi_max_c = 6 * D_max
    # define model and Hamiltonian
    tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
    h_mpos = tfi_model.get_h_mpos()
    h_bonds = tfi_model.get_h_bonds()
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_energy_compression_{bc}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        # load iso_peps ground state and excitations optimized from wavefunction or mps overlap
        if 2*Lx*Ly <= 20:
            # load wavefunctions and corresponding iso_peps
            file_base_iso = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_wavefunction"
            pkl_path_iso = script_path.parent / "data" / "excitations2" / f"{file_base_iso}.pkl"
            with open(pkl_path_iso, "rb") as pkl_file:
                psis_exact, Es_exact, ess_bond_exact, psis_iso, Es_iso, ess_bond_iso, overlaps = pickle.load(pkl_file)
            print("Exact diagonalization:")
            for i in range(len(Es_exact)):
                print(f"- E{i}_exact = {Es_exact[i]}.")
            print("")
            print(f"Isometric PEPS with D_max = {D_max}, chi_max_c = {chi_max_c}:")
            for i in range(len(Es_iso)):
                print(f"- E{i}_iso = {Es_iso[i]}, overlap{i} = {np.abs(overlaps[i])}.")
            es_bond0 = ess_bond_iso[0]
        elif D_max_mps is not None:
            # load mps
            file_base_mps = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}"
            pkl_path_mps = script_path.parent / "data" / "excitations2" / f"{file_base_mps}.pkl"
            with open(pkl_path_mps, "rb") as pkl_file:
                _, Es_mps, _ = pickle.load(pkl_file)
            # load corresponding iso_peps
            file_base_iso = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_mps_{D_max_mps}"
            pkl_path_iso = script_path.parent / "data" / "excitations2" / f"{file_base_iso}.pkl"
            with open(pkl_path_iso, "rb") as pkl_file:
                psis_iso, Es_iso, ess_bond_iso, overlaps = pickle.load(pkl_file)
            print(f"MPS with D_max = {D_max_mps}:")
            for i in range(len(Es_mps)):
                print(f"- E{i}_mps = {Es_mps[i]}.")
            print("")
            print(f"Isometric PEPS with D_max = {D_max}, chi_max_c = {chi_max_c}:")
            for i in range(len(Es_iso)):
                print(f"- E{i}_iso = {Es_iso[i]}, overlap{i} = {np.abs(overlaps[i])}.")
            es_bond0 = ess_bond_iso[0]
        else:
            file_base_tebd = f"tebd_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{0.08}_{10}"
            pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
            with open(pkl_path_tebd, "rb") as pkl_file:
                iso_peps0 = pickle.load(pkl_file)
            E0_iso = np.sum(iso_peps0.copy().get_column_expectation_values(h_mpos))
            print(f"Isometric PEPS with D_max = {D_max}, chi_max_c = {chi_max_c}:")
            print(f"- E{0}_iso = {E0_iso}.")
            es_bond0_list = iso_peps0.copy().get_bond_expectation_values(h_bonds)
            es_bond0 = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
            for n in range((2*Lx-1)*(2*Ly-1)):
                bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
                es_bond0[bx][by] = es_bond0_list[n]
        print("")
        # compute energies for random and first excitation
        Es_matvec = []
        Es_bond = []
        ess_bond = []
        Es_random_matvec = []
        Es_random_bond = []
        ess_bond_random = []
        chi_max_bs_actual = []
        for chi_max_b in chi_max_bs:
            if (2*Lx*Ly <= 20) or (D_max_mps is not None):
                e_iso_peps_overlap = psis_iso[1]
                e_iso_peps = ExcitedIsometricPEPS.from_ExcitedIsometricPEPSOverlap(e_iso_peps_overlap, bc, chi_max_b)
                ALs, ARs, CDs, CCs, CUs = e_iso_peps.ALs, e_iso_peps.ARs, e_iso_peps.CDs, e_iso_peps.CCs, e_iso_peps.CUs
            else:
                ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
            engine = VariationalQuasiparticleExcitationsEngine(D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, h_mpos, bc, chi_max_b, eps_b)
            # first excitation
            if (2*Lx*Ly <= 20) or (D_max_mps is not None):
                print(f"1) First excitation with {overlaps[1]} overlap")
                vecX = np.hstack([e_iso_peps.vecX, e_iso_peps.vecX_column])
                E_matvec = np.inner(np.conj(vecX), Heff(engine)._matvec(vecX))
                Es_matvec.append(E_matvec)
                print(f"=> E_matvec = {E_matvec}.")
                e_iso_peps.initialize_compressed_boundaries()
                es_bond = e_iso_peps.get_bond_expectation_values(h_bonds)
                E_bond = np.sum(es_bond)
                Es_bond.append(E_bond)
                ess_bond.append(es_bond)
                print(f"=> E_bond = {E_bond}.")
            # random excitation
            print("2) Random excitation")
            np.random.seed(0)
            vecX_random = np.random.randn(engine.shape_vecX + engine.shape_vecX_column) \
                          + 1.j * np.random.randn(engine.shape_vecX + engine.shape_vecX_column)
            vecX_random /= np.linalg.norm(vecX_random)
            E_random_matvec = np.inner(np.conj(vecX_random), Heff(engine)._matvec(vecX_random))
            Es_random_matvec.append(E_random_matvec)
            print(f"=> E_random_matvec = {E_random_matvec}.")
            e_iso_peps_random = ExcitedIsometricPEPS(D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, \
                                                     vecX_random, bc, chi_max_b, eps_b)
            e_iso_peps_random.initialize_compressed_boundaries()
            es_bond_random = e_iso_peps_random.get_bond_expectation_values(h_bonds)
            E_random_bond = np.sum(es_bond_random)
            Es_random_bond.append(E_random_bond)
            ess_bond_random.append(es_bond_random)
            print(f"=> E_random_bond = {E_random_bond}. \n")
            chi_max_bs_actual.append(chi_max_b)
            if (2*Lx*Ly <= 20) or (D_max_mps is not None):
                with open(pkl_path, "wb") as pkl_file:
                    pickle.dump((chi_max_bs_actual, Es_random_matvec, Es_random_bond, ess_bond_random, Es_matvec, Es_bond, ess_bond, es_bond0), pkl_file)
            else:
                with open(pkl_path, "wb") as pkl_file:
                    pickle.dump((chi_max_bs_actual, Es_random_matvec, Es_random_bond, ess_bond_random, es_bond0), pkl_file)
        print("")
        print(f"for chi_max_bs = {chi_max_bs}:")
        if (2*Lx*Ly <= 20) or (D_max_mps is not None):
            print(f"=> Es_matvec = {np.real_if_close(Es_matvec)}.")
            print(f"=> Es_bond = {np.real_if_close(Es_bond)}.")
        print(f"=> Es_random_matvec = {np.real_if_close(Es_random_matvec)}.")
        print(f"=> Es_random_bond = {np.real_if_close(Es_random_bond)}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_energy_compression_convergence(Lx, Ly, g, D_max, bc, D_max_mps=None):
    chi_max_c = 6 * D_max
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_energy_compression_{bc}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    if (2*Lx*Ly <= 20) or (D_max_mps is not None):
        with open(pkl_path, "rb") as pkl_file:
            chi_max_bs, Es_random_matvec, Es_random_bond, ess_bond_random, Es_matvec, Es_bond, ess_bond, es_bond0 = pickle.load(pkl_file)
    else:
        with open(pkl_path, "rb") as pkl_file:
            chi_max_bs, Es_random_matvec, Es_random_bond, ess_bond_random, es_bond0 = pickle.load(pkl_file)
            Es_matvec = Es_random_matvec
            Es_bond = Es_random_bond
            ess_bond = ess_bond_random
    N_chi = len(chi_max_bs)
    # total energy
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    axes.set_xlabel(r"$\chi_{b}$")
    axes.set_ylabel(r"$\vert E \vert$")
    axes.plot(chi_max_bs, np.abs(Es_matvec), "x-", color="blue", label="matvec")
    axes.plot(chi_max_bs, np.abs(Es_bond), ".-", color="green", label="bond")
    axes.legend(loc="best")
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    """
    # bond energies
    for i in range(N_chi):
        for bx in range(len(es_bond0)):
            for by in range(len(es_bond0[0])):
                ess_bond[i][bx][by] -= es_bond0[bx][by]
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    fig, axes = plt.subplots(N_chi, 1, figsize=(3, N_chi*3))
    for ax in axes:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
    for i in range(N_chi):
        chi_max_b = chi_max_bs[i]
        im = axes[i].imshow(np.real(ess_bond[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(rf"$\chi_b = {chi_max_b}$")
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.03)
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base}_2.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    """
    return


def compute_bond_energies(Lx, Ly, g, D_max, bc, chi_max_b_1, chi_max_b_2, chi_max_b_new):
    chi_max_c = 6 * D_max
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_variational_{chi_max_b_1}_{chi_max_b_2}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        psis, Es, ess_bond, overlapss = pickle.load(pkl_file)
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b_1}_{chi_max_b_2}_bond_energies_{chi_max_b_new}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        k = len(psis) - 1
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        h_bonds = tfi_model.get_h_bonds()
        for i in range(1, k+1):
            print(f"- Excitation {i}:")
            e_iso_peps = psis[i]
            e_iso_peps.chi_max_b = chi_max_b_new
            e_iso_peps.initialize_compressed_boundaries()
            es_bond_iso = e_iso_peps.get_bond_expectation_values(h_bonds)
            print(f"=> deltaE = {np.sum(es_bond_iso)-Es[i]}.")
            ess_bond[i] = es_bond_iso
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((psis, Es, ess_bond, overlapss), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return