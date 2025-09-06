import sys
from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from ..src.mps.a_mps import MPS
from ..src.mps.b_model_finite import TFIModelFinite
from ..src.mps.c_dmrg import dmrg_algorithm
from ..src.mps.d_excitations import VariationalQuasiparticleExcitationEngine

from ..src.isometric_peps.b_model import DiagonalSquareLattice, TFIModelDiagonalSquare, TFIModelDiagonalSquareTenpy
from ..src.isometric_peps.h_excitations2_overlap import ExcitedIsometricPEPS as ExcitedIsometricPEPSOverlap, \
                                                        get_overlap_mps_iso_peps, \
                                                        get_overlap_wavefunction_iso_peps
from ..src.isometric_peps.i_excitations2 import VariationalQuasiparticleExcitationsEngine, \
                                                ExcitedIsometricPEPS, \
                                                Heff, \
                                                extract_all_isometric_configurations
    

# 1) Overlap with exact wavefunction

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
        H_bonds = tfi_model.get_H_bonds_uniform()
        h_mpos = tfi_model.get_h_mpos()
        h_bonds = tfi_model.get_h_bonds_uniform()
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
            excited_iso_peps = ExcitedIsometricPEPSOverlap.optimized_from_excited_wavefunction(iso_peps0, psis_exact[i])
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


# 2) MPS

def run_excitations2_mps(Lx, Ly, g, D_max, k):
    """For system sizes Lx and Ly, snake an MPS through the diagonal square lattice (down -> up if
    Lx > Ly and left -> right if Ly >= Lx). For transverse field g and maximal bond dimension D_max, 
    compute the ground state with DMRG and the first k excitations on top. If the number of spins 
    does not exceed 20, compare with exact diagonalization. Also compute the uniform local bond
    energies for all states."""
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max}"
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
            Es_exact, psis_exact = sparse.linalg.eigsh(H, k=k+1, which="SA")
            psis_exact = [psis_exact[:, i] for i in range(k+1)]
            print(f"- E{0}_exact = {Es_exact[0]}.")
            for i in range(1, k+1):
                print(f"- E{i}_exact = {Es_exact[i]} (e{i}_exact = {Es_exact[i]-Es_exact[0]}).")
            print("")
        # mps
        print(f"MPS with D_max = {D_max}:")
        if Lx > Ly:
            order = "down_to_up"
        elif Ly >= Lx:
            order = "left_to_right"
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
        if Ly >= Lx:
            print("")
            print("Bond energies.")
            N = 2 * Lx * Ly
            tfi_model = TFIModelFinite(N, g)
            ess_bond_mps = []
            for i in range(k+1):
                print(f"- State {i}.")
                mps = psis_mps[i]
                es_bond = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
                n = 0
                for ny in range(2*Ly-1):
                    for x in range(Lx):
                        print(f"bond {n}.")
                        if ny%2 == 0 and x > 0:
                            es_bond[2*x-1][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, n+Lx-1))
                        es_bond[2*x][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, n+Lx))
                        if ny%2 == 1 and x < Lx-1:
                            es_bond[2*x+1][ny] = mps.get_mpo_expectation_value(tfi_model.get_bond_mpo(n, n+Lx+1))
                        n += 1
                assert n == N - Lx
                ess_bond_mps.append(es_bond)
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((psis_mps, Es_mps, ess_bond_mps), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return


# 3) Overlap with MPS

def run_excitations2_overlap_mps(Lx, Ly, g, D_max_mps, D_max_iso, k):
    """For system sizes Lx, Ly and transverse field g, excite the isoPEPS ground state (received 
    from TEBD^2 or DMRG^2 with D_max_iso) by optimizing the overlap with the excited MPS of maximal 
    bond dimension D_max_mps. Do this for the first k excitations on top of the ground state. Also 
    compute the uniform local bond energies for all states."""
    chi_max_c = 6 * D_max_iso
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max_iso}_{chi_max_c}_overlap_mps_{D_max_mps}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        # define model and (bond) Hamiltonians
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        h_mpos = tfi_model.get_h_mpos()
        h_bonds = tfi_model.get_h_bonds_uniform()
        # load mps
        print(f"MPS with D_max = {D_max_mps}:")
        file_base_mps = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}"
        pkl_path_mps = script_path.parent / "data" / "excitations2" / f"{file_base_mps}.pkl"
        with open(pkl_path_mps, "rb") as pkl_file:
            psis_mps, Es_mps, _ = pickle.load(pkl_file)
        for i in range(k+1):
            print(f"- E{i}_mps = {Es_mps[i]}.")
        print("")
        # load iso_peps ground state
        print(f"Isometric PEPS with D_max = {D_max_iso}, chi_max_c = {chi_max_c}:")
        if D_max_iso == 2:
            file_base_tebd = f"tebd_{Lx}_{Ly}_{g}_{D_max_iso}_{chi_max_c}_{0.05}_{100}"
            pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
            with open(pkl_path_tebd, "rb") as pkl_file:
                iso_peps0 = pickle.load(pkl_file)
        elif D_max_iso > 2:
            file_base_dmrg = f"dmrg_{Lx}_{Ly}_{g}_{D_max_iso}_{chi_max_c}_{6*(D_max_iso**2)}_{3}_{None}_{3}"
            pkl_path_dmrg = script_path.parent / "data" / "dmrg2" / f"{file_base_dmrg}.pkl"
            with open(pkl_path_dmrg, "rb") as pkl_file:
                _, _, _, _, iso_peps_list = pickle.load(pkl_file)
                iso_peps0 = iso_peps_list[-1]
        E0_iso = np.sum(iso_peps0.copy().get_column_expectation_values(h_mpos))
        print(f"E0_iso = {E0_iso}.")
        es_bond0_list = iso_peps0.copy().get_bond_expectation_values(h_bonds)
        es_bond0 = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0[bx][by] = es_bond0_list[n]
        overlap0 = get_overlap_mps_iso_peps(psis_mps[0].ARs, iso_peps0)
        print(f"=> |<mps{0}|iso_peps{0}>| = {np.abs(overlap0)}. \n")
        psis_iso = [iso_peps0]
        Es_iso = [E0_iso]
        ess_bond_iso = [es_bond0]
        overlaps = [overlap0]
        # compute iso_peps excitations
        bc = "variational"
        chi_max_b = 6 * (D_max_iso**2)
        engine = VariationalQuasiparticleExcitationsEngine(iso_peps0, h_mpos, bc, chi_max_b)
        print("")
        for i in range(1, k+1):
            print(f"Excitation {i}:")
            e_iso_peps_overlap, overlap = ExcitedIsometricPEPSOverlap.optimized_from_excited_mps(iso_peps0, psis_mps[i])
            psis_iso.append(e_iso_peps_overlap)
            overlaps.append(overlap)
            e_iso_peps = ExcitedIsometricPEPS.from_ExcitedIsometricPEPSOverlap(e_iso_peps_overlap, bc, chi_max_b)
            vecX = np.hstack([e_iso_peps.vecX, e_iso_peps.vecX_column])
            E_iso = np.real_if_close(np.inner(np.conj(vecX), Heff(engine)._matvec(vecX)))
            Es_iso.append(E_iso)
            print(f"=> E{i}_iso = {E_iso}.")
            e_iso_peps.initialize_compressed_boundaries()
            es_bond_iso = e_iso_peps.get_bond_expectation_values(h_bonds)
            ess_bond_iso.append(es_bond_iso)
            print("")
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((psis_iso, Es_iso, ess_bond_iso, overlaps), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations2_overlap_mps(k):
    # global parameters
    Lx = 5
    Ly = 5
    g = 3.5
    D_max_mps = 256
    # load data for mps
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_mps, ess_bond_mps = pickle.load(pkl_file)
    # load data for D_max = 2
    D_max = 2
    chi_max_c = 12
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_mps_{D_max_mps}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_2, ess_bond_iso_2, overlaps_2 = pickle.load(pkl_file)
    # load data for D_max = 3
    D_max = 3
    chi_max_c = 18
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_overlap_mps_{D_max_mps}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_3, ess_bond_iso_3, overlaps_3 = pickle.load(pkl_file)
    # restrict data to first k excitations
    ess_bond_mps = ess_bond_mps[:(k+1)]
    ess_bond_iso_2 = ess_bond_iso_2[:(k+1)]
    ess_bond_iso_3 = ess_bond_iso_3[:(k+1)]
    # combine all bond energy data for global color scaling
    ess_bond = ess_bond_mps + ess_bond_iso_2 + ess_bond_iso_3 
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    # plot
    fig, axes = plt.subplots(k+1, 3, figsize=(3*3, (k+1)*3))
    """
    fig.suptitle(rf"isoPEPS excitations from MPS overlap", fontsize=15, \
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    """
    fig.text(0.06, 0.525, 
            rf"isoPEPS excitations from MPS overlap", 
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
    # mps
    for i in range(k+1):
        axes[i, 0].set_title(rf"$\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 0].imshow(np.real(ess_bond_mps[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == k:
            axes[i, 0].set_xlabel(r"$\Uparrow \text{MPS}$", fontsize=14)
        if i == 0:
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    # D_max = 2
    for i in range(k+1):
        overlap = np.abs(overlaps_2[i])
        if i == 0:
            axes[i, 1].set_title(rf"${overlap:.4f}\vert \psi_{{{i}}} \rangle$")
        else:
            axes[i, 1].set_title(rf"${overlap:.3f}\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 1].imshow(np.real(ess_bond_iso_2[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == k:
            axes[i, 1].set_xlabel(r"$\Uparrow D_{\text{max}} = 2$", fontsize=14)
    # D_max = 3
    for i in range(k+1):
        overlap = np.abs(overlaps_3[i])
        if i == 0:
            axes[i, 2].set_title(rf"${overlap:.4f}\vert \psi_{{{i}}} \rangle$")
        else:
            axes[i, 2].set_title(rf"${overlap:.3f}\vert \psi_{{{i}}} \rangle$")
        im = axes[i, 2].imshow(np.real(ess_bond_iso_3[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == k:
            axes[i, 2].set_xlabel(r"$\Uparrow D_{\text{max}} = 3$", fontsize=14)
    # save
    file_base_png = f"excitations_{Lx}_{Ly}_{g}_overlap_mps"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


# 4) Effective Hamiltonian

def run_excitations2_effective(Lx, Ly, g, D_max, bc, chi_max_b, k, eps_b=1.e-15):
    """For system sizes Lx, Ly and transverse field g, excite the ground state (received from TEBD^2 
    or DMRG^2 with D_max) by diagonalizing the effective Hamiltonian. To compress the boundaries,
    use bc "variational" or "column", chi_max_b and eps_b. Do this for the first k excitations on 
    top of the ground state. Also compute the uniform local bond energies for all states."""
    chi_max_c = 6 * D_max
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b}"
    log_path = script_path.parent / "data" / "excitations2" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"Lx = {Lx}, Ly = {Ly}, g = {g}. \n")
        # define model and (bond) Hamiltonians
        tfi_model = TFIModelDiagonalSquare(Lx, Ly, g)
        h_mpos = tfi_model.get_h_mpos()
        h_bonds = tfi_model.get_h_bonds_uniform()
        # load mps for reference
        D_max_mps = 256
        print(f"MPS with D_max = {D_max_mps}:")
        file_base_mps = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}"
        pkl_path_mps = script_path.parent / "data" / "excitations2" / f"{file_base_mps}.pkl"
        with open(pkl_path_mps, "rb") as pkl_file:
            psis_mps, Es_mps, _ = pickle.load(pkl_file)
        for i in range(k+1):
            print(f"- E{i}_mps = {Es_mps[i]}.")
        print("")
        # load iso_peps ground state
        print(f"Isometric PEPS with D_max = {D_max}, chi_max_c = {chi_max_c}:")
        if D_max == 2:
            file_base_tebd = f"tebd_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{0.05}_{100}"
            pkl_path_tebd = script_path.parent / "data" / "tebd2" / f"{file_base_tebd}.pkl"
            with open(pkl_path_tebd, "rb") as pkl_file:
                iso_peps0 = pickle.load(pkl_file)
        elif D_max > 2:
            file_base_dmrg = f"dmrg_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_{6*(D_max**2)}_{3}_{None}_{3}"
            pkl_path_dmrg = script_path.parent / "data" / "dmrg2" / f"{file_base_dmrg}.pkl"
            with open(pkl_path_dmrg, "rb") as pkl_file:
                _, _, _, _, iso_peps_list = pickle.load(pkl_file)
                iso_peps0 = iso_peps_list[-1]
        E0_iso = np.sum(iso_peps0.copy().get_column_expectation_values(h_mpos))
        print(f"E0_iso = {E0_iso}.")
        es_bond0_list = iso_peps0.copy().get_bond_expectation_values(h_bonds)
        es_bond0 = [[None] * (2*Ly-1) for _ in range(2*Lx-1)]
        for n in range((2*Lx-1)*(2*Ly-1)):
            bx, by = DiagonalSquareLattice(Lx, Ly).get_bond_vector(n)
            es_bond0[bx][by] = es_bond0_list[n]
        overlap0 = get_overlap_mps_iso_peps(psis_mps[0].ARs, iso_peps0)
        print(f"=> |<mps{0}|iso_peps{0}>| = {np.abs(overlap0)}. \n")
        psis_iso = [iso_peps0]
        Es_iso = [E0_iso]
        ess_bond_iso = [es_bond0]
        overlapss = [overlap0]
        # compute iso_peps excitations
        print(f"boundary compression = {bc}, chi_max_b = {chi_max_b}.")
        engine = VariationalQuasiparticleExcitationsEngine(iso_peps0, h_mpos, bc, chi_max_b, eps_b)
        print("")
        Es, vecXs = engine.run(k)
        ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
        e_iso_pepss = [ExcitedIsometricPEPS(D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, \
                                            vecXs[i], bc, chi_max_b, eps_b) for i in range(k)]
        for i in range(k):
            Es_iso.append(Es[i])
            psis_iso.append(e_iso_pepss[i])
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((psis_iso, Es_iso, ess_bond_iso, overlapss), pkl_file)
        print("")
        for i in range(1, k+1):
            print(f"- Excitation {i}:")
            e_iso_peps = psis_iso[i]
            e_iso_peps.initialize_compressed_boundaries()
            es_bond_iso = e_iso_peps.get_bond_expectation_values(h_bonds)
            ess_bond_iso.append(es_bond_iso)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((psis_iso, Es_iso, ess_bond_iso, overlapss), pkl_file)
        print("")
        for i in range(1, k+1):
            print(f"- Excitation {i}:")
            e_iso_peps = psis_iso[i]
            overlaps = []
            e_iso_peps_overlap = ExcitedIsometricPEPSOverlap.from_Xs_form2(iso_peps0, e_iso_peps.Xs, e_iso_peps.Xs_column)
            sum_overlap2 = 0.
            for j in range(1, len(psis_mps)):
                overlap = e_iso_peps_overlap.get_overlap_with_excited_mps(psis_mps[j])
                print(f"=> |<mps{j}|e_iso_peps{i}>| = {np.abs(overlap)}.")
                overlaps.append(overlap)
                sum_overlap2 += np.abs(overlap)**2
            overlapss.append(overlaps)
            print(f"=> sum_j |<mps(j)|e_iso_peps{i}>|^2 = {sum_overlap2}.")
            print("")
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((psis_iso, Es_iso, ess_bond_iso, overlapss), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations2_effective(k):
    # global parameters
    Lx = 5
    Ly = 5
    g = 3.5
    D_max_mps = 256
    bc = "variational"
    # load data for mps
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_{Lx}_{Ly}_{g}_mps_{D_max_mps}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_mps, ess_bond_mps = pickle.load(pkl_file)
    # load data for D_max = 2
    D_max = 2
    chi_max_c = 12
    chi_max_b = 24
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_2, ess_bond_iso_2, overlapss_2 = pickle.load(pkl_file)
    # load data for D_max = 3
    D_max = 3
    chi_max_c = 18
    chi_max_b = 54
    file_base = f"excitations_{Lx}_{Ly}_{g}_{D_max}_{chi_max_c}_effective_{bc}_{chi_max_b}"
    pkl_path = script_path.parent / "data" / "excitations2" / f"{file_base}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        _, Es_iso_3, ess_bond_iso_3, overlapss_3 = pickle.load(pkl_file)
    # restrict data to ground state and first k excitations
    Es_mps = Es_mps[:(k+1)]
    ess_bond_mps = ess_bond_mps[:(k+1)]
    Es_iso_2 = Es_iso_2[:(k+1)]
    ess_bond_iso_2 = ess_bond_iso_2[:(k+1)]
    Es_iso_3 = Es_iso_3[:(k+1)]
    ess_bond_iso_3 = ess_bond_iso_3[:(k+1)]
    # combine all bond energy data for global color scaling
    ess_bond = ess_bond_mps + ess_bond_iso_2 + ess_bond_iso_3
    vmin = min(np.min(np.real(es_bond)) for es_bond in ess_bond)
    vmax = max(np.max(np.real(es_bond)) for es_bond in ess_bond)
    # plot
    fig, axes = plt.subplots(k+1, 3, figsize=(3*3, (k+1)*3))
    """
    fig.suptitle(rf"isoPEPS excitations from effective Hamiltonian", fontsize=15, \
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    """
    fig.text(0.06, 0.525, 
            rf"isoPEPS excitations from effective Hamiltonian", 
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
    # mps
    for i in range(k+1):
        axes[i, 0].set_title(rf"$E_{{{i}}} = {Es_mps[i]:.3f}$")
        im = axes[i, 0].imshow(np.real(ess_bond_mps[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == k:
            axes[i, 0].set_xlabel(r"$\Uparrow \text{MPS}$", fontsize=14)
        if i == 0:
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02)
    # D_max = 2
    for i in range(k+1):
        deltaE = Es_iso_2[i] - Es_mps[i]
        axes[i, 1].set_title(rf"$\Delta E_{{{i}}} = {deltaE:.3f}$")
        im = axes[i, 1].imshow(np.real(ess_bond_iso_2[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == k:
            axes[i, 1].set_xlabel(r"$\Uparrow D_{\text{max}} = 2$", fontsize=14)
    # D_max = 3
    for i in range(k+1):
        deltaE = Es_iso_3[i] - Es_mps[i]
        axes[i, 2].set_title(rf"$\Delta E_{{{i}}} = {deltaE:.3f}$")
        im = axes[i, 2].imshow(np.real(ess_bond_iso_3[i]).T, origin='lower', vmin=vmin, vmax=vmax)
        if i == k:
            axes[i, 2].set_xlabel(r"$\Uparrow D_{\text{max}} = 3$", fontsize=14)
    # save
    file_base_png = f"excitations_{Lx}_{Ly}_{g}_effective"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


# 5) Perturbation theory

def plot_single_particle(g, k):
    ess_bond_100, Es_100 = TFIModelDiagonalSquare(100, 100, g).get_es_bond_single_particle(k)
    ess_bond_10, Es_10 = TFIModelDiagonalSquare(10, 10, g).get_es_bond_single_particle(k)
    ess_bond_5, Es_5 = TFIModelDiagonalSquare(5, 5, g).get_es_bond_single_particle(k)
    ess_bond_3, Es_3 = TFIModelDiagonalSquare(3, 3, g).get_es_bond_single_particle(k)
    vmin_100 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_100)
    vmax_100 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_100)
    vmin_10 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_10)
    vmax_10 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_10)
    vmin_5 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_5)
    vmax_5 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_5)
    vmin_3 = min(np.min(np.real(es_bond)) for es_bond in ess_bond_3)
    vmax_3 = max(np.max(np.real(es_bond)) for es_bond in ess_bond_3)
    fig, axes = plt.subplots(k, 4, figsize=(4*3, 3*k))
    fig.text(0.09, 0.525, 
            "Excitations from single particle model", 
            va='center', ha='center', rotation='vertical',
            fontsize=15,
            bbox=dict(facecolor='none', edgecolor='black', boxstyle='square,pad=0.3'))
    fig.subplots_adjust(top=0.93)
    for i in range(k):
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
        axes[i, 3].set_xticks([])
        axes[i, 3].set_xticklabels([])
        axes[i, 3].set_yticks([])
        axes[i, 3].set_yticklabels([])
    for i in range(k):
        if i == 0:
            axes[i, 0].set_title(rf"$k = (1, 1) \:\:\: \epsilon_k = {Es_100[i]:.6f}$")
            axes[i, 1].set_title(rf"$k = (1, 1) \:\:\: \epsilon_k = {Es_10[i]:.5f}$")
            axes[i, 2].set_title(rf"$k = (1, 1) \:\:\: \epsilon_k = {Es_5[i]:.4f}$")
            axes[i, 3].set_title(rf"$k = (1, 1) \:\:\: \epsilon_k = {Es_3[i]:.3f}$")
        elif i == 1:
            axes[i, 0].set_title(rf"$k = (2, 1) \:\:\: \epsilon_k = {Es_100[i]:.6f}$")
            axes[i, 1].set_title(rf"$k = (2, 1) \:\:\: \epsilon_k = {Es_10[i]:.5f}$")
            axes[i, 2].set_title(rf"$k = (2, 1) \:\:\: \epsilon_k = {Es_5[i]:.4f}$")
            axes[i, 3].set_title(rf"$k = (2, 1) \:\:\: \epsilon_k = {Es_3[i]:.3f}$")
        elif i == 2:
            axes[i, 0].set_title(rf"$k = (1, 2) \:\:\: \epsilon_k = {Es_100[i]:.6f}$")
            axes[i, 1].set_title(rf"$k = (1, 2) \:\:\: \epsilon_k = {Es_10[i]:.5f}$")
            axes[i, 2].set_title(rf"$k = (1, 2) \:\:\: \epsilon_k = {Es_5[i]:.4f}$")
            axes[i, 3].set_title(rf"$k = (1, 2) \:\:\: \epsilon_k = {Es_3[i]:.3f}$")
        elif i == 3:
            axes[i, 0].set_title(rf"$k = (2, 2) \:\:\: \epsilon_k = {Es_100[i]:.6f}$")
            axes[i, 1].set_title(rf"$k = (2, 2) \:\:\: \epsilon_k = {Es_10[i]:.5f}$")
            axes[i, 2].set_title(rf"$k = (2, 2) \:\:\: \epsilon_k = {Es_5[i]:.4f}$")
            axes[i, 3].set_title(rf"$k = (2, 2) \:\:\: \epsilon_k = {Es_3[i]:.3f}$")
            axes[i, 0].set_xlabel(rf"$\Uparrow L_x = L_y = 100$", fontsize=14)
            axes[i, 1].set_xlabel(rf"$\Uparrow L_x = L_y = 10$", fontsize=14)
            axes[i, 2].set_xlabel(rf"$\Uparrow L_x = L_y = 5$", fontsize=14)
            axes[i, 3].set_xlabel(rf"$\Uparrow L_x = L_y = 3$", fontsize=14)
        im_100 = axes[i, 0].imshow(np.real(ess_bond_100[i]).T, origin='lower', vmin=vmin_100, vmax=vmax_100)
        im_10 = axes[i, 1].imshow(np.real(ess_bond_10[i]).T, origin='lower', vmin=vmin_10, vmax=vmax_10)
        im_5 = axes[i, 2].imshow(np.real(ess_bond_5[i]).T, origin='lower', vmin=vmin_5, vmax=vmax_5)
        im_3 = axes[i, 3].imshow(np.real(ess_bond_3[i]).T, origin='lower', vmin=vmin_3, vmax=vmax_3)
    # save
    script_path = Path(__file__).resolve().parent
    file_base_png = file_base_png = f"excitations_{g}_single_particle"
    png_path = script_path.parent / "data" / "excitations2" / f"{file_base_png}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")