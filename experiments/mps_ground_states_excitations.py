import sys
from pathlib import Path
import pickle

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..src.uniform_mps.a_umps import UniformMPS
from ..src.uniform_mps.b_model_infinite import TFIModelInfinite
from ..src.uniform_mps.c_vumps import vumps_algorithm
from ..src.uniform_mps.d_uexcitations import VariationalPlaneWaveExcitationEngine

from ..src.mps.a_mps import MPS
from ..src.mps.b_model_finite import TFIModelFinite
from ..src.mps.c_dmrg import dmrg_algorithm
from ..src.mps.d_excitations import VariationalQuasiparticleExcitationEngine, ExcitedMPS


def plot_tfi_phase_diagram():
    plt.rcParams.update({
        "text.usetex": True,            # use LaTeX
        "font.family": "serif",         # serif font for physics look
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}"
    })
    # png path
    script_path = Path(__file__).resolve().parent
    png_path = script_path.parent / "data" / "mps_ground_states_excitations" / "tfi_phase_diagram.png"
    # magnetization
    def get_magnetization(g):
        if g <= 1.:
            return (1. - g**2)**(1/8)
        else:
            return 0.
    gs = np.arange(0., 2., step=0.001)
    ms = [get_magnetization(g) for g in gs]
    # figure
    fig, ax = plt.subplots(figsize=(5, 2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 1.3)
    ax.annotate("", xy=(2.05, 0.), xytext=(-0.015, 0),
                arrowprops=dict(arrowstyle="->", lw=1.), zorder=0)
    ax.annotate("", xy=(0, 1.3), xytext=(0, -0.02),
                arrowprops=dict(arrowstyle="->", lw=1.), zorder=1)
    ax.plot(gs, ms, "-", color="purple", zorder=2)
    # axis labels
    ax.text(2., -0.05, r"$g/J$", ha="center", va="top", fontsize=12)
    ax.text(1., -0.05, r"$(g/J)_c$", ha="center", va="top", fontsize=12)
    ax.text(-0.05, 1.25, r"$\vert \langle \sigma^x \rangle \vert$", ha="right", va="center", fontsize=12)
    ax.text(-0.05, 0., r"0", ha="right", va="center", fontsize=12)
    ax.text(-0.05, 1., r"1", ha="right", va="center", fontsize=12)
    # states
    ax.text(0.25, 0.65, r"SSB", ha="center", va="top", fontsize=12)
    ax.text(0.25, 0.5, r"$\vert \rightarrow \rightarrow \rightarrow \rangle$", ha="center", va="top", fontsize=12)
    ax.text(0.25, 0.35, r"or", ha="center", va="top", fontsize=12)
    ax.text(0.25, 0.2, r"$\vert \leftarrow \leftarrow \leftarrow \rangle$", ha="center", va="top", fontsize=12)
    ax.text(1.8, 0.2, r"$\vert \uparrow \uparrow \uparrow \rangle$", ha="center", va="top", fontsize=12)
    # save
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return


def run_vumps_energy(g, Ds, tol, maxruns):
    """For the TFI model on an infinite chain with transverse field g, find the ground states with 
    VUMPS using bond dimensions Ds. Converge up to tolerance tol in gradient norm but perform 
    maximally maxruns vumps updates. Save the ground state energy densities in pkl file."""
    tol = float(tol)
    script_path = Path(__file__).resolve().parent
    file_base = f"vumps_energy_{g}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"g = {g}, tol = {tol}, maxruns = {maxruns}. \n")
        tfi_model_infinite = TFIModelInfinite(g)
        e_exact = tfi_model_infinite.get_exact_gs_energy_density()
        h = tfi_model_infinite.get_h()
        Ds_done = []
        es = []
        delta_es = []
        for D in Ds:
            print(f"D = {D}.")
            guess_umps = UniformMPS.from_desired_bond_dimension(D)
            e, _, _ = vumps_algorithm(h, guess_umps, tol, maxruns)
            delta_e = np.abs(e - e_exact)
            print(f"|e - e_exact| = {delta_e}. \n")
            Ds_done.append(D)
            es.append(e)
            delta_es.append(delta_e)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((Ds_done, es), pkl_file)
        print(f"delta_es = {np.array(delta_es)}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def run_vumps_magnetization(D, tol, maxruns, gs=None):
    """For the TFI model on an infinite chain with transverse fields gs, find the ground states with 
    VUMPS using bond dimension D. Converge up to tolerance tol in gradient norm but perform 
    maximally maxruns vumps updates. Save the ground state magnetization densities in pkl file."""
    tol = float(tol)
    if gs is None:
        gs = [1.e-5, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.994, 0.998, \
              1.002, 1.006, 1.01, 1.05, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]
    script_path = Path(__file__).resolve().parent
    file_base = f"vumps_magnetization_{D}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"D = {D}, tol = {tol}, maxruns = {maxruns}. \n")
        gs_done = []
        mxs = []
        guess_umps = UniformMPS.from_desired_bond_dimension(D)
        print("")
        for g in gs:
            print(f"g = {g}.")
            tfi_model_infinite = TFIModelInfinite(g)
            h = tfi_model_infinite.get_h()
            _, umps, _ = vumps_algorithm(h, guess_umps, tol, maxruns)
            mx = umps.get_site_expectation_value(tfi_model_infinite.sigma_x)
            print(f"mx = {mx}. \n")
            gs_done.append(g)
            mxs.append(mx)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((gs_done, mxs), pkl_file)
        print(f"mxs = {np.array(mxs)}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_vumps(D):
    """Plot the VUMPS ground state energy density convergence (for g = 0.5, 1.0, 1.5) and the 
    quantum phase diagram with magnetization mx as order parameter."""
    script_path = Path(__file__).resolve().parent
    # figure 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    png_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"vumps.png"
    ax[0].set_title("(a)")
    ax[0].set_xlabel(r"bond dimension $D$")
    ax[0].set_ylabel(r"$\vert e - e_{\text{exact}} \vert$")
    ax[1].set_title("(b)")
    ax[1].set_xlabel(r"transverse field $g$")
    ax[1].set_ylabel(r"$\vert \langle \sigma^x \rangle \vert$")
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    ax[1].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # load energy data
    pkl_path_05 = script_path.parent / "data" / "mps_ground_states_excitations" / "vumps_energy_0.5.pkl"
    pkl_path_10 = script_path.parent / "data" / "mps_ground_states_excitations" / "vumps_energy_1.0.pkl"
    pkl_path_15 = script_path.parent / "data" / "mps_ground_states_excitations" / "vumps_energy_1.5.pkl"
    delta_es_05 = None
    try:
        with open(pkl_path_05, "rb") as pkl_file_05:
            Ds_05, es_05 = pickle.load(pkl_file_05)
            tfi_model_infinite = TFIModelInfinite(g=0.5)
            e_exact_05 = tfi_model_infinite.get_exact_gs_energy_density()
            delta_es_05 = [np.abs(e_05 - e_exact_05) for e_05 in es_05]
    except FileNotFoundError:
        print("No data available for g = 0.5")
    delta_es_10 = None
    try:
        with open(pkl_path_10, "rb") as pkl_file_10:
            Ds_10, es_10 = pickle.load(pkl_file_10)
            tfi_model_infinite = TFIModelInfinite(g=1.0)
            e_exact_10 = tfi_model_infinite.get_exact_gs_energy_density()
            delta_es_10 = [np.abs(e_10 - e_exact_10) for e_10 in es_10]
    except FileNotFoundError:
        print("No data available for g = 1.0")
    delta_es_15 = None
    try:
        with open(pkl_path_15, "rb") as pkl_file_15:
            Ds_15, es_15 = pickle.load(pkl_file_15)
            tfi_model_infinite = TFIModelInfinite(g=1.5)
            e_exact_15 = tfi_model_infinite.get_exact_gs_energy_density()
            delta_es_15 = [np.abs(e_15 - e_exact_15) for e_15 in es_15]
    except FileNotFoundError:
        print("No data available for g = 1.5")
    # plot energies against bond dimensions
    if delta_es_05 is not None:
        ax[0].loglog(Ds_05, delta_es_05, ".-", color="green", label=r"$g = 0.5$")
    if delta_es_10 is not None:
        ax[0].loglog(Ds_10, delta_es_10, ".-", color="red", label=r"$g = 1.0$")
    if delta_es_15 is not None:
        ax[0].loglog(Ds_15, delta_es_15, ".-", color="blue", label=r"$g = 1.5$")
    # load magnetization data
    pkl_path_m = script_path.parent / "data" / "mps_ground_states_excitations" / f"vumps_magnetization_{D}.pkl"
    mxs = None
    try:
        with open(pkl_path_m, "rb") as pkl_file_m:
            gs, mxs = pickle.load(pkl_file_m)
    except FileNotFoundError:
        print("No data available for magnetization")
    # plot magnetizations against transverse fields
    if mxs is not None:
        ax[1].axvline(x=1.0, color="black")
        ax[1].plot(gs, np.abs(mxs), ".-", color="purple", label=rf"$D = {D}$")
    # legend and save
    ax[1].legend(loc='best')
    ax[0].legend(loc='best')
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return

def run_uexcitations_dispersion(g, D, tol, maxruns, ps=None):
    """For the TFI model on an infinite chain with transverse field g, find the ground state with 
    VUMPS using bond dimension D. Converge up to tolerance tol in gradient norm but perform 
    maximally maxruns vumps updates. Then find variational plane wave excitations on top, for 
    momenta ps within the first BZ (-pi, pi]. Save the dispersion in pkl file."""
    tol = float(tol)
    if ps == None or ps == "None":
        ps = list(np.arange(-np.pi, 0, np.pi/10)) + [0] + list(np.arange(0, np.pi + np.pi/10, np.pi/10))
    script_path = Path(__file__).resolve().parent
    file_base = f"uexcitations_dispersion_{g}_{D}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"g = {g}, D = {D}, tol = {tol}. \n")
        tfi_model_infinite = TFIModelInfinite(g)
        h = tfi_model_infinite.get_h()
        guess_umps0 = UniformMPS.from_desired_bond_dimension(D)
        _, umps0, _ = vumps_algorithm(h, guess_umps0, tol, maxruns)
        print("")
        # paramagnetic phase (single spin flip excitations) and critical point (gapless excitation)
        if g >= 1.:  
            es = []
            for p in ps:
                excitation_engine = VariationalPlaneWaveExcitationEngine(umps0, h, p)
                e, _ = excitation_engine.run(k=1)
                es.append(e)
            print(f"Found dispersion relation for single particle excitations on top of ground state.")
        elif 0 <= g < 1:  # ferromagnetic phase (topological domain wall excitations)
            mx0 = umps0.get_site_expectation_value(tfi_model_infinite.sigma_x)
            for i in range(100):
                guess_umps0_tilde = UniformMPS.from_desired_bond_dimension(D)
                _, umps0_tilde, _ = vumps_algorithm(h, guess_umps0_tilde, tol, maxruns)
                mx0_tilde = umps0_tilde.get_site_expectation_value(tfi_model_infinite.sigma_x)
                if np.abs(mx0 + mx0_tilde) < 1.e-5:
                    print("")
                    print(f"Found mx0={mx0} and mx0_tilde={mx0_tilde} after {i+2} ground state searchs.")
                    es = []
                    for p in ps:
                        excitation_engine = VariationalPlaneWaveExcitationEngine(umps0, h, p, umps0_tilde)
                        e, _ = excitation_engine.run(k=1)
                        es.append(e)
                    print(f"Found dispersion relation for single particle excitations on top of ground state.")
                    break
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((ps, es), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_uexcitations_dispersion(D):
    """Plot the dispersion relation of the lowest lying single particle excitations in the first
    Brillouin zone (for g = 0.5, 1.0, 1.5)."""
    script_path = Path(__file__).resolve().parent
    # figure 
    fig, ax = plt.subplots(figsize=(7, 4))
    png_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"uexcitations_dispersion.png"
    ax.set_xlabel(r"momentum $p$")
    ax.set_xticks(np.arange(-np.pi, 3*np.pi/2, np.pi/2), [r"$-\pi$",r"$-\pi /2$", r"$0$", r"$\pi /2$",r"$\pi$"])
    ax.set_ylabel(r"$\epsilon_p$")
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # load dispersion data
    pkl_path_05 = script_path.parent / "data" / "mps_ground_states_excitations" / f"uexcitations_dispersion_0.5_{D}.pkl"
    pkl_path_10 = script_path.parent / "data" / "mps_ground_states_excitations" / f"uexcitations_dispersion_1.0_{D}.pkl"
    pkl_path_15 = script_path.parent / "data" / "mps_ground_states_excitations" / f"uexcitations_dispersion_1.5_{D}.pkl"
    es_05 = None
    try:
        with open(pkl_path_05, "rb") as pkl_file_05:
            ps_05, es_05 = pickle.load(pkl_file_05)
            tfi_model_infinite = TFIModelInfinite(g=0.5)
            ps_exact_05, es_exact_05 = tfi_model_infinite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 0.5")
    es_10 = None
    try:
        with open(pkl_path_10, "rb") as pkl_file_10:
            ps_10, es_10 = pickle.load(pkl_file_10)
            tfi_model_infinite = TFIModelInfinite(g=1.0)
            ps_exact_10, es_exact_10 = tfi_model_infinite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 1.0")
    es_15 = None
    try:
        with open(pkl_path_15, "rb") as pkl_file_15:
            ps_15, es_15 = pickle.load(pkl_file_15)
            tfi_model_infinite = TFIModelInfinite(g=1.5)
            ps_exact_15, es_exact_15 = tfi_model_infinite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 1.5")
    # plot dispersions
    ax.axhline(y=0., color="black")
    if es_05 is not None:
        ferro_exact, = ax.plot(ps_exact_05, es_exact_05, "-", color="lightgreen")
        ferro, = ax.plot(ps_05, es_05, ".", color="green")
    if es_10 is not None:
        crit_exact, = ax.plot(ps_exact_10, es_exact_10, "-", color="lightcoral")
        crit, = ax.plot(ps_10, es_10, ".", color="darkred")
    if es_15 is not None:
        para_exact, = ax.plot(ps_exact_15, es_exact_15, "-", color="lightskyblue")
        para, = ax.plot(ps_15, es_15, ".", color="blue")
    # legend and save
    fig.legend(handles=[ferro_exact, crit_exact, para_exact], \
               labels=[r"$g = 0.5$", r"$g = 1.0$", r"$g = 1.5$"], \
               loc="center left", \
               bbox_to_anchor=(0.9, 0.75), \
               title=r"$\epsilon_p^{\text{exact}} = 2 \sqrt{g^2 - 2g\cos(p) + 1}$")
    fig.legend(handles=[ferro, crit, para], \
               labels=[r"$g = 0.5$", r"$g = 1.0$", r"$g = 1.5$"], \
               loc="center left", \
               bbox_to_anchor=(0.9, 0.5), \
               title=f"VUMPS + VPWE (D = {D})")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return




def run_dmrg(N, bc, g, D_max, guess_mps0, num_runs=10, eps=1.e-14):
    """For the TFI model on a finite chain of length N with boundary conditions bc and transverse 
    field g, find the ground state with num_runs DMRG runs. Allow maximal bond dimension D_max and 
    discard any singular values smaller than eps. As initial guess, choose a random state with full
    bond dimensions or a product state equal to the ground state for extreme g."""
    eps = float(eps)
    print(f"N = {N}, bc = {bc}, g = {g}, D_max = {D_max}:")
    tfi_model_finite = TFIModelFinite(N, g)
    mpo = tfi_model_finite.get_mpo(bc)
    if guess_mps0 == "random":
        guess_mps0 = MPS.from_desired_bond_dimension(N, D_max)
    elif guess_mps0 == "product":
        if g >= 1:  # paramagnetic phase and critical point
            guess_mps0 = MPS.from_qubit_product_state(N, spin_orientation="up")
        elif 0 <= g < 1.:  # ferromagnetic phase 
            import random
            spin_orientations = ["right", "left"]
            spin_orientation = random.choice(spin_orientations)
            guess_mps0 = MPS.from_qubit_product_state(N, spin_orientation)
    E0, mps0, var0 = dmrg_algorithm(mpo, guess_mps0, D_max, eps, num_runs)
    return E0, mps0, var0

def run_dmrg_variance(N, g, D_maxs=None, bc="periodic", guess_mps0="random", num_runs=10, eps=1.e-14):
    """Compute the translation and Hamiltonian variances for maximal bond dimensions D_maxs and save
    them in pkl file."""
    if D_maxs is None:
        D_maxs = [2, 4, 8, 16, 32, 64, 128, 256]
    script_path = Path(__file__).resolve().parent
    file_base = f"dmrg_variance_{N}_{g}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    D_maxs_done = []
    vars_T = []
    exps_T = []
    vars_H = []
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"N = {N}, g = {g}, bc = {bc}, num_runs = {num_runs}, eps = {eps}. \n")
        for D_max in D_maxs:
            _, mps, var_H = run_dmrg(N, bc, g, D_max, guess_mps0, num_runs, eps)
            var_T = mps.get_translation_variance()
            exp_T = mps.get_translation_expectation_value()
            print(f"var(T) = {var_T}.")
            print(f"<T> = {exp_T}.")
            print("")
            D_maxs_done.append(D_max)
            vars_T.append(var_T)
            exps_T.append(exp_T)
            vars_H.append(var_H)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((D_maxs_done, vars_T, exps_T, vars_H), pkl_file)
        print(f"vars_T = {np.array(vars_T)}.")
        print(f"exps_T = {np.array(vars_T)}.")
        print(f"vars_H = {np.array(vars_H)}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def run_dmrg_magnetization(N, D_max, gs=None, bc="periodic", guess_mps0="random", num_runs=10, eps=1.e-14):
    """Compute the magnetizations for transverse field values gs and save them in pkl file."""
    if gs is None:
        gs = [1.e-5, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, \
              1.02, 1.04, 1.06, 1.08, 1.1, 1.13, 1.2, 1.4, 1.6, 1.8, 2.0]
    script_path = Path(__file__).resolve().parent
    file_base = f"dmrg_magnetization_{N}_{D_max}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"N = {N}, D_max = {D_max}, bc = {bc}, num_runs = {num_runs}, eps = {eps}. \n")
        gs_done = []
        mxs = []
        for g in gs:
            print(f"g = {g}.")
            _, mps, _ = run_dmrg(N, bc, g, D_max, guess_mps0, num_runs, eps)
            sigma_x = TFIModelFinite(N, g).sigma_x
            Cxs, _ = mps.get_correlation_functions(sigma_x, sigma_x, N//4, 3*N//4)
            mx = np.sqrt(Cxs[-1])
            print(f"mx = {mx}. \n")
            gs_done.append(g)
            mxs.append(mx)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((gs_done, mxs), pkl_file)
        print(f"mxs = {np.array(mxs)}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def run_dmrg_ssb(g, D_max, Ns=None, bc="periodic", guess_mps0="random", num_runs=10, eps=1.e-14):
    """In the ferromagnetic phase g < 1, compute the magnetizations for system sizes Ns to check
    till which system size DMRG is able to converge to the symmetric ground state."""
    if Ns is None:
        Ns = [20, 25, 30, 35, 40, 45, 50]
    script_path = Path(__file__).resolve().parent
    file_base = f"dmrg_ssb_{g}_{D_max}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"g = {g}, D_max = {D_max}, bc = {bc}, num_runs = {num_runs}, eps = {eps}. \n")
        Ns_done = []
        mxs = []    
        for N in Ns:  
            _, mps, _ = run_dmrg(N, bc, g, D_max, guess_mps0, num_runs, eps)
            sigma_x = TFIModelFinite(N, g).sigma_x
            mx = np.mean(mps.get_site_expectation_values([sigma_x] * N))
            print(f"mx = {mx}. \n")
            Ns_done.append(N)
            mxs.append(mx)
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump((Ns_done, mxs), pkl_file)
        print(f"mxs = {np.array(mxs)}.")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_dmrg(N_var, N_mag, D_max_mag, g_ssb):
    """For DMRG ground states, plot the convergences of the variances of translation operator and 
    PBC Hamiltonian against maximal bond dimensions (for g = 0.5, 1.0, 1.5). Also draw the quantum 
    phase diagram with magnetization mx as order parameter."""
    script_path = Path(__file__).resolve().parent
    # figure 
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1, 1], hspace=0.05, wspace=0.3)
    png_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"dmrg.png"
    # left side: translation and hamiltonian variances
    ax_left_top = fig.add_subplot(gs[0, 0])
    ax_left_top.set_title(rf"(a)")
    ax_left_top.set_ylabel(r"$\text{var}(T)$")
    ax_left_top.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    inset_ax_left_top = inset_axes(ax_left_top, width="40%", height="40%", loc='upper right',
                                   bbox_to_anchor=(-0.02, -0.03, 1, 1), bbox_transform=ax_left_top.transAxes, borderpad=0)
    inset_ax_left_top.set_ylabel(r"$\langle T \rangle$")
    inset_ax_left_top.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    inset_ax_left_top.set_ylim(0.965, 1.003)
    ax_left_top.tick_params(labelbottom=False)
    ax_left_bottom = fig.add_subplot(gs[1, 0])
    ax_left_bottom.set_xlabel(r"maximal bond dimension $D_{\text{max}}$")
    ax_left_bottom.set_ylabel(r"$\text{var}(H)$")
    ax_left_bottom.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # right side: magnetization
    ax_right = fig.add_subplot(gs[:, 1])  
    ax_right.set_title(rf"(b)")
    bbox_top = ax_left_top.get_position()
    bbox_bot = ax_left_bottom.get_position()
    y_center = (bbox_top.y0 + bbox_bot.y1) / 2
    height = (bbox_top.y1 - bbox_bot.y0) * 1.
    new_bbox = [bbox_top.x0 + 0.38, y_center - height / 2, 0.35, height]
    ax_right.set_position(new_bbox)
    ax_right.set_xlabel(r"transverse field $g$")
    ax_right.set_ylabel(r"$\langle \sigma^x_{N/4} \sigma^x_{3N/4} \rangle^{1/2}$")
    ax_right.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    inset_ax_right = inset_axes(ax_right, width="30%", height="30%", loc='lower left', \
                                bbox_to_anchor=(0.15, 0.15, 1, 1), bbox_transform=ax_right.transAxes, borderpad=0)
    inset_ax_right.set_title(rf"$g = {g_ssb}$")
    inset_ax_right.set_xlabel(r"$N$")
    inset_ax_right.set_xticks([20, 30, 40, 50])
    inset_ax_right.set_ylim(-0.1, 1.1)
    inset_ax_right.set_ylabel(r"$\vert \langle \sigma^x \rangle \vert$")
    inset_ax_right.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # load translation and Hamiltonian variance data
    pkl_path_05 = script_path.parent / "data" / "mps_ground_states_excitations" / f"dmrg_variance_{N_var}_{0.5}.pkl"
    pkl_path_10 = script_path.parent / "data" / "mps_ground_states_excitations" / f"dmrg_variance_{N_var}_{1.0}.pkl"
    pkl_path_15 = script_path.parent / "data" / "mps_ground_states_excitations" / f"dmrg_variance_{N_var}_{1.5}.pkl"
    vars_T_05 = None
    try:
        with open(pkl_path_05, "rb") as pkl_file_05:
            D_maxs_05, vars_T_05, exps_T_05, vars_H_05 = pickle.load(pkl_file_05)
    except FileNotFoundError:
        print("No data available for g = 0.5")
    vars_T_10 = None
    try:
        with open(pkl_path_10, "rb") as pkl_file_10:
            D_maxs_10, vars_T_10, exps_T_10, vars_H_10 = pickle.load(pkl_file_10)
    except FileNotFoundError:
        print("No data available for g = 1.0")
    vars_T_15 = None
    try:
        with open(pkl_path_15, "rb") as pkl_file_15:
            D_maxs_15, vars_T_15, exps_T_15, vars_H_15 = pickle.load(pkl_file_15)
    except FileNotFoundError:
        print("No data available for g = 1.5")
    # plot variances against bond dimensions
    if vars_T_05 is not None:
        ax_left_top.loglog(D_maxs_05, np.abs(vars_T_05), ".-", color="green", label=r"$g = 0.5$")
        plot_left_bottom_05, = ax_left_bottom.loglog(D_maxs_05, np.abs(vars_H_05), ".-", color="green", label=r"$g = 0.5$")
        inset_ax_left_top.semilogx(D_maxs_05, np.real(exps_T_05), ".-", color="green", label=r"$g = 0.5$")
    if vars_T_10 is not None:
        ax_left_top.loglog(D_maxs_10, np.abs(vars_T_10), ".-", color="red", label=r"$g = 1.0$")
        plot_left_bottom_10, = ax_left_bottom.loglog(D_maxs_10, np.abs(vars_H_10), ".-", color="red", label=r"$g = 1.0$")
        inset_ax_left_top.semilogx(D_maxs_10, np.real(exps_T_10), ".-", color="red", label=r"$g = 1.0$")
    if vars_T_15 is not None:
        ax_left_top.loglog(D_maxs_15, np.abs(vars_T_15), ".-", color="blue", label=r"$g = 1.5$")
        plot_left_bottom_15, = ax_left_bottom.loglog(D_maxs_15, np.abs(vars_H_15), ".-", color="blue", label=r"$g = 1.5$")
        inset_ax_left_top.semilogx(D_maxs_15, np.real(exps_T_15), ".-", color="blue", label=r"$g = 1.5$")
    # load magnetization data
    pkl_path_m = script_path.parent / "data" / "mps_ground_states_excitations" / f"dmrg_magnetization_{N_mag}_{D_max_mag}.pkl"
    mxs = None
    try:
        with open(pkl_path_m, "rb") as pkl_file_m:
            gs, mxs = pickle.load(pkl_file_m)
    except FileNotFoundError:
        print("No data available for magnetization")
    # plot magnetizations against transverse fields
    if mxs is not None:
        ax_right.axvline(x=1.0, color="black")
        ax_right.plot(gs, mxs, ".-", color="purple", label=rf"$N = {N_mag}$, $D_{{\max}} = {D_max_mag}$")
    # load ssb data
    pkl_path_m = script_path.parent / "data" / "mps_ground_states_excitations" / f"dmrg_ssb_{g_ssb}_{D_max_mag}.pkl"
    mxs = None
    try:
        with open(pkl_path_m, "rb") as pkl_file_m:
            Ns, mxs = pickle.load(pkl_file_m)
    except FileNotFoundError:
        print("No data available for ssb")
    # plot ssb
    if mxs is not None:
        inset_ax_right.plot(Ns, np.abs(mxs), ".-", color="green")
    # legend and save
    #ax_left_bottom.legend(loc='best')
    fig.legend(handles=[plot_left_bottom_05, plot_left_bottom_10, plot_left_bottom_15], \
               labels=[r"$g = 0.5$", r"$g = 1.0$", r"$g = 1.5$"], \
               loc="lower center", \
               bbox_to_anchor=(0.4, 0.235), \
               title=rf"$N = {N_var}$")
    ax_right.legend(loc='best')
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return

def get_degeneracies(Es, psis, tol):
    """For a list Es of eigenvalues and a list psis of corresponding eigenvectors, find degeneracies
    up to tolerance tol."""
    k = len(Es)
    H_deg = {}
    for i in range(k):
        E_i = Es[i]
        psi_i = psis[i]
        contained_i = False
        for E in H_deg:
            if np.allclose(E_i, E, rtol=tol, atol=0):
                H_deg[E].append(psi_i)
                contained_i = True
                break
        if not contained_i:
            H_deg[E_i] = [psi_i]
    return H_deg

def diagonalize_translation_operator(H_deg, get_translation_overlap):
    """For H_deg a dictionary with eigenenergies as keys and degenerate eigenvectors as values,
    build the translation matrix with get_translation_overlap and diagonalize it."""
    H_deg_diagt = {}
    for E in H_deg:
        psis = H_deg[E]
        l = len(psis)
        T_E = np.zeros((l, l), dtype=complex)
        for i in range(l):
            for j in range(l):
                T_E[i, j] = get_translation_overlap(psis[j], psis[i])
        ts, phis = np.linalg.eig(T_E)
        H_deg_diagt[E] = [ts, phis, psis]
    return H_deg_diagt

def t_to_p(t, N):
    """For a translation eigenvalue t = e^{i * (2pi/N) * p}, compute the integer momentum p."""
    return N/(2 * np.pi) * np.angle(t)

def run_excitations_pbc(N, g, D_max, k, deg_tol=1.e-5, guess_mps0="product", num_runs=10, eps=1.e-14):
    """For the TFI model on a finite chain of length N with periodic boundary conditions and 
    transverse field g, find the ground state with DMRG and k variational quasiparticle excitations 
    on top. For each degenerate subspace, diagonalize the translation operator."""
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_pbc_{N}_{g}_{D_max}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        tfi_model_finite = TFIModelFinite(N, g)
        # energy ground state with dmrg
        E0, mps0, _ = run_dmrg(N, "periodic", g, D_max, guess_mps0, num_runs, eps)
        # energy excitations and degeneracies
        mpo = tfi_model_finite.get_mpo(bc="periodic")
        excitation_engine = VariationalQuasiparticleExcitationEngine(mps0, mpo)
        if g >= 1.:  # paramagnetic phase (and critical point): unique symmetric ground state
            es_excited, empss = excitation_engine.run(k)
        elif 0 <= g < 1.:  # ferromagnetic phase: two-fold degenerate symmetry broken ground states
            es, empss = excitation_engine.run(k+1)
            es_excited, empss = es[1:], empss[1:]
        H_deg = get_degeneracies(es_excited, empss, deg_tol)  # {e: [empss]}
        empss_sup_p = [emps.copy() for emps in empss]  # superposition of states with definite momentum
        # translation diagonalization
        def get_translation_overlap(emps1, emps2):
            return emps1.get_translation_overlap(emps2)
        H_deg_diagt = diagonalize_translation_operator(H_deg, get_translation_overlap)
        # {e: [ts, phis, empss]}
        ps = []
        es = []
        empss = []
        for e in H_deg_diagt:
            ts_e, phis_e, empss_e = H_deg_diagt[e]
            ps_e = t_to_p(ts_e, N)
            l = len(ps_e)
            for i in range(l):
                ps.append(ps_e[i])
                phi_i = phis_e[:, i]
                vecX_i = np.zeros(empss_e[0].shape_vecX, dtype=complex)
                for j in range(l):
                    vecX_i += phi_i[j] * empss_e[j].vecX
                emps_i = ExcitedMPS(empss_e[0].ALs, empss_e[0].ARs, vecX_i)
                empss.append(emps_i)  # state with definite momentum
                e_i = emps_i.get_mpo_expectation_value(mpo) - E0
                assert np.allclose(e_i, e)
                es.append(e)
        print(f"Found {k} momentum resolved variational quasiparticle excitations on top of ground state.")
        print(f"ps = {np.array(ps)}.")
        print(f"es = {np.array(es)}.")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((empss_sup_p, ps, es, empss), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def run_excitations_pbc_ed(N, g, k):
    """For the TFI model on a finite chain of length N with periodic boundary conditions and 
    transverse field g, find the ground state and excitations with exact diagonalization. For each 
    degenerate subspace, diagonalize the translation operator."""
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_pbc_ed_{N}_{g}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        print(f"N = {N}, g = {g}:")
        tfi_model_finite = TFIModelFinite(N, g)
        # energy diagonalization and degeneracies
        H = tfi_model_finite.get_H(bc="periodic")
        if g >= 1.:  # paramagnetic phase (and critical point): unique symmetric ground state
            Es, psis = sparse.linalg.eigsh(H, k=k+1, which="SA")
        elif 0 <= g < 1.:  # ferromagnetic phase: two-fold degenerate symmetry broken ground states
            Es, psis = sparse.linalg.eigsh(H, k=k+2, which="SA")
            Es, psis = Es[1:], psis[:, 1:]
        E0 = Es[0]
        Es_excited = Es[1:]
        psis_excited = [psis[:, i] for i in range(1, k+1)]
        H_deg = get_degeneracies(Es_excited, psis_excited, tol=1.e-5)
        psis_sup_p = psis_excited  # superposition of states with definite momentum
        # translation diagonalization
        T = tfi_model_finite.get_T()
        def get_translation_overlap(psi, phi):
            return np.inner(np.conj(phi), T @ psi)
        H_deg_diagt = diagonalize_translation_operator(H_deg, get_translation_overlap)
        # {E: [ts, phis, psis]}
        ps = []
        es = []
        psis = []
        for E in H_deg_diagt:
            ts_E, phis_E, psis_E = H_deg_diagt[E]
            ps_E = t_to_p(ts_E, N)
            l = len(ps_E)
            for i in range(l):
                ps.append(ps_E[i])
                phi_i = phis_E[:, i]
                psi_i = np.zeros(np.shape(psis_E[0]), dtype=complex)
                for j in range(l):
                    psi_i += phi_i[j] * psis_E[j]
                psis.append(psi_i)  # state with definite momentum
                E_i = np.inner(np.conj(psi_i), H @ psi_i)
                assert np.allclose(E_i, E)
                es.append(E - E0)
        print("")
        print(f"Found {k} momentum resolved excitations with exact diagonalization.")
        print(f"ps = {np.array(ps)}.")
        print(f"es = {np.array(es)}.")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((psis_sup_p, ps, es, psis), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def run_excitations_dispersion(N, g, D_max, guess_mps0="product", num_runs=10, eps=1.e-14):
    """For the TFI model on a finite chain of length N with periodic boundary conditions and 
    transverse field g, find the ground state with DMRG and the single particle excitation 
    dispersion on top by simultaneously diagonalizing the effective quasiparticle Hamiltonian and 
    translation operator."""
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_dispersion_{N}_{g}_{D_max}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        tfi_model_finite = TFIModelFinite(N, g)
        # ground state with dmrg
        _, mps0, _ = run_dmrg(N, "periodic", g, D_max, guess_mps0, num_runs, eps)
        # variational excitation disperion relation
        mpo = tfi_model_finite.get_mpo(bc="periodic")
        ps = tfi_model_finite.get_ps()
        ps_actual = []
        es = []
        empss = []
        for p in ps:
            excitation_engine = VariationalQuasiparticleExcitationEngine(mps0, mpo, p)
            # ferromagnetic phase: two-fold degenerate symmetry broken ground states
            if p == 0 and 0 <= g < 1.: 
                es0, empss0 = excitation_engine.run(k=2)
                e = es0[1]
                emps = empss0[1]
                p_actual = N/(2 * np.pi) * np.angle(emps.get_translation_overlap())
                ps_actual.append(p_actual)
                es.append(e)
                empss.append(emps)
            else:
                e, emps = excitation_engine.run(k=1)
                p_actual = N/(2 * np.pi) * np.angle(emps.get_translation_overlap())
                if np.abs(p_actual + N//2) < 1.e-5:
                    p_actual += N
                ps_actual.append(p_actual)
                es.append(e)
                empss.append(emps)
            print(f"Found variational quasiparticle excitation with momentum {p_actual} and energy {e}.")
        print("")
        print(f"ps = {np.array(ps_actual)}.")
        print(f"es = {np.array(es)}.")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((ps_actual, es, empss), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations(N, D_max):
    script_path = Path(__file__).resolve().parent
    # figure 
    fig, ax = plt.subplots(3, 2, figsize=(10, 9))
    png_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations.png"
    # left column
    ax[0, 0].set_title("(a)")
    ax[1, 0].set_ylabel(r"excitation energy")
    ax[2, 0].set_xlabel(r"momentum $k$")
    for i in range(3):
        labels = [-N//2] + ["" for _ in range(N//4-1)] + [-N//4] + ["" for _ in range(N//4-1)] + [0] \
                 + ["" for _ in range(N//4-1)] + [N//4] + ["" for _ in range(N//4-1)]+ [N//2]
        ax[i, 0].set_xticks(np.arange(-N/2, N/2+1), labels=labels)
    ax[0, 0].set_ylim(-0.2, 4.2)
    ax[0, 0].set_yticks([0, 1, 2, 3, 4])
    ax[1, 0].set_yticks([0, 1, 2, 3, 4])
    ax[0, 0].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    ax[1, 0].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    ax[2, 0].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # load energy data
    pkl_path_05 = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_{N}_0.5_{D_max}.pkl"
    pkl_path_05_ed = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_ed_{N}_0.5.pkl"
    es_05 = None
    try:
        with open(pkl_path_05_ed, "rb") as pkl_file_05_ed:
            _, ps_05_ed, es_05_ed, _ = pickle.load(pkl_file_05_ed)
        with open(pkl_path_05, "rb") as pkl_file_05:
            _, ps_05, es_05, _ = pickle.load(pkl_file_05)
            tfi_model_finite = TFIModelFinite(N, g=0.5)
            _, _, ps_con_05, es_con_05 = tfi_model_finite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 0.5")
    pkl_path_10 = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_{N}_1.0_{D_max}.pkl"
    pkl_path_10_ed = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_ed_{N}_1.0.pkl"
    es_10 = None
    try:
        with open(pkl_path_10_ed, "rb") as pkl_file_10_ed:
            _, ps_10_ed, es_10_ed, _ = pickle.load(pkl_file_10_ed)
        with open(pkl_path_10, "rb") as pkl_file_10:
            _, ps_10, es_10, _ = pickle.load(pkl_file_10)
            tfi_model_finite = TFIModelFinite(N, g=1.0)
            _, _, ps_con_10, es_con_10 = tfi_model_finite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 1.0")
    pkl_path_15 = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_{N}_1.5_{D_max}.pkl"
    pkl_path_15_ed = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_ed_{N}_1.5.pkl"
    es_15 = None
    try:
        with open(pkl_path_15_ed, "rb") as pkl_file_15_ed:
            _, ps_15_ed, es_15_ed, _ = pickle.load(pkl_file_15_ed)
        with open(pkl_path_15, "rb") as pkl_file_15:
            _, ps_15, es_15, _ = pickle.load(pkl_file_15)
            tfi_model_finite = TFIModelFinite(N, g=1.5)
            _, _, ps_con_15, es_con_15 = tfi_model_finite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 1.5")
    # plot ground state energy
    for i in range(3):
        ax[i, 0].axhline(y=0., color="black")
    # plot energies against momenta
    if es_05 is not None:
        ax[0, 0].plot(ps_con_05, es_con_05, "-", color="lightgreen", label=r"$\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
        plot_ed, = ax[0, 0].plot(ps_05_ed[:-4], es_05_ed[:-4], "x", color="purple", label="Exact diagonalization")
        plot_05, = ax[0, 0].plot(ps_05[:-4], es_05[:-4], ".", color="green", label=r"DMRG + VQPE ($D_{\text{max}} = 64$)")
    if es_10 is not None:
        ax[1, 0].plot(ps_con_10, es_con_10, "-", color="lightcoral", label=r"$\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
        ax[1, 0].plot(ps_10_ed, es_10_ed, "x", color="purple", label="Exact diagonalization")
        plot_10, = ax[1, 0].plot(ps_10, es_10, ".", color="darkred", label=r"DMRG + VQPE ($D_{\text{max}} = 64$)")
    if es_15 is not None:
        ax[2, 0].plot(ps_con_15, es_con_15, "-", color="lightskyblue", label=r"$\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
        ax[2, 0].plot(ps_15_ed[:-4], es_15_ed[:-4], "x", color="purple", label="Exact diagonalization")
        plot_15, = ax[2, 0].plot(ps_15[:-4], es_15[:-4], ".", color="blue", label=r"DMRG + VQPE ($D_{\text{max}} = 64$)")
    # right colomn
    ax[0, 1].set_title("(b)")
    ax[1, 1].set_ylabel(r"$\epsilon_k$")
    ax[2, 1].set_xlabel(r"momentum $k$")
    for i in range(3):
        labels = [-N//2] + ["" for _ in range(N//4-1)] + [-N//4] + ["" for _ in range(N//4-1)] + [0] + ["" for _ in range(N//4-1)] + [N//4] + ["" for _ in range(N//4-1)]+ [N//2]
        ax[i, 1].set_xticks(np.arange(-N/2, N/2+1), labels=labels)
    ax[0, 1].set_yticks([0, 1, 2, 3, 4])
    ax[1, 1].set_yticks([0, 1, 2, 3, 4])
    ax[0, 1].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    ax[1, 1].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    ax[2, 1].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # load energy data
    pkl_path_05 = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_dispersion_{N}_0.5_{D_max}.pkl"
    es_05 = None
    try:
        with open(pkl_path_05, "rb") as pkl_file_05:
            ps_05, es_05, _ = pickle.load(pkl_file_05)
            tfi_model_finite = TFIModelFinite(N, g=0.5)
            _, _, ps_con_05, es_con_05 = tfi_model_finite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 0.5")
    pkl_path_10 = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_dispersion_{N}_1.0_{D_max}.pkl"
    es_10 = None
    try:
        with open(pkl_path_10, "rb") as pkl_file_10:
            ps_10, es_10, _ = pickle.load(pkl_file_10)
            tfi_model_finite = TFIModelFinite(N, g=1.0)
            _, _, ps_con_10, es_con_10 = tfi_model_finite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 1.0")
    pkl_path_15 = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_dispersion_{N}_1.5_{D_max}.pkl"
    es_15 = None
    try:
        with open(pkl_path_15, "rb") as pkl_file_15:
            ps_15, es_15, _ = pickle.load(pkl_file_15)
            tfi_model_finite = TFIModelFinite(N, g=1.5)
            _, _, ps_con_15, es_con_15 = tfi_model_finite.get_exact_excitation_dispersion()
    except FileNotFoundError:
        print("No data available for g = 1.5")
    # plot ground state energy
    for i in range(3):
        ax[i, 1].axhline(y=0., color="black")
    # plot energies against momenta
    if es_05 is not None:
        plot_con_05, = ax[0, 1].plot(ps_con_05, es_con_05, "-", color="lightgreen", label=r"$\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
        plot_t_05, = ax[0, 1].plot(ps_05, es_05, ".", color="green", label=r"$\left[H_{\text{eff}} - \alpha \left( e^{-i\frac{2\pi}{N}k}T_{\text{eff}} + e^{i\frac{2\pi}{N}k}T_{\text{eff}}^{\dagger} \right) \right] \vert X ) = \epsilon_k \vert X )$")
    if es_10 is not None:
        plot_con_10, = ax[1, 1].plot(ps_con_10, es_con_10, "-", color="lightcoral", label=r"$\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
        plot_t_10, = ax[1, 1].plot(ps_10, es_10, ".", color="darkred", label=r"$\left[H_{\text{eff}} - \alpha \left( e^{-i\frac{2\pi}{N}k}T_{\text{eff}} + e^{i\frac{2\pi}{N}k}T_{\text{eff}}^{\dagger} \right) \right] \vert X ) = \epsilon_k \vert X )$")
    if es_15 is not None:
        plot_con_15, = ax[2, 1].plot(ps_con_15, es_con_15, "-", color="lightskyblue", label=r"$\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
        plot_t_15, = ax[2, 1].plot(ps_15, es_15, ".", color="blue", label=r"$\left[H_{\text{eff}} - \alpha \left( e^{-i\frac{2\pi}{N}k}T_{\text{eff}} + e^{i\frac{2\pi}{N}k}T_{\text{eff}}^{\dagger} \right) \right] \vert X ) = \epsilon_k \vert X )$")
    # legend
    fig.legend(handles=[plot_05, plot_10, plot_15], \
               labels=[r"$g = 0.5$", r"$g = 1.0$", r"$g = 1.5$"], \
               loc="center left", \
               bbox_to_anchor=(0.11, -0.01), \
               title=r"(a) DMRG + VQPE ($D_{\text{max}} = 64$)")
    fig.legend(handles=[plot_ed], \
               labels=[r"Exact diagonalization"], \
               loc="center left", \
               bbox_to_anchor=(0.11, -0.08), )
    fig.legend(handles=[plot_con_05, plot_con_10, plot_con_15], \
               labels=[r"$g = 0.5$", r"$g = 1.0$", r"$g = 1.5$"], \
               loc="center left", \
               bbox_to_anchor=(0.34, -0.01), \
               title=r"(a)(b) $\epsilon_k = 2 \sqrt{g^2 - 2g\cos \left( \frac{2\pi}{N}k \right) + 1}$")
    fig.legend(handles=[plot_t_05, plot_t_10, plot_t_15], \
               labels=[r"$g = 0.5$", r"$g = 1.0$", r"$g = 1.5$"], \
               loc="center left", \
               bbox_to_anchor=(0.6, -0.01), \
               title=r"(b) $\left[H_{\text{eff}} - \alpha \left( e^{-i\frac{2\pi}{N}k}T_{\text{eff}} + e^{i\frac{2\pi}{N}k}T_{\text{eff}}^{\dagger} \right) \right] \vert X ) = \epsilon_k \vert X )$")
    # save
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return

def run_excitations_obc(N, g, D_max, k, guess_mps0="product", num_runs=10, eps=1.e-14):
    """For the TFI model on a finite chain of length N with open boundary conditions and transverse 
    field g, find the ground state with DMRG and k variational quasiparticle excitations on top."""
    script_path = Path(__file__).resolve().parent
    file_base = f"excitations_obc_{N}_{g}_{D_max}"
    log_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.log"
    pkl_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"{file_base}.pkl"
    with open(log_path, "w", buffering=1) as log_file:
        sys.stdout = log_file
        sys.stderr = log_file
        tfi_model_finite = TFIModelFinite(N, g)
        # ground state with dmrg
        _, mps0, _= run_dmrg(N, "open", g, D_max, guess_mps0, num_runs, eps)
        sigma_x = TFIModelFinite(N, g).sigma_x
        mx0 = np.mean(mps0.get_site_expectation_values([sigma_x] * N))
        # variational quasiparticle excitations
        mpo = tfi_model_finite.get_mpo(bc="open")
        # paramagnetic phase (single spin flip excitations) and critical point (gapless excitation)
        if g >= 1.:  
            excitation_engine = VariationalQuasiparticleExcitationEngine(mps0, mpo)
            es, empss = excitation_engine.run(k)
        elif 0 <= g < 1:  # ferromagnetic phase (topological domain wall excitations)
            for i in range(100):
                _, mps0_tilde, _ = run_dmrg(N, "open", g, D_max, guess_mps0, num_runs, eps)
                mx0_tilde = np.mean(mps0_tilde.get_site_expectation_values([sigma_x] * N))
                if np.abs(mx0 + mx0_tilde) < 1.e-5:
                    print(f"Found mx0={mx0} and mx0_tilde={mx0_tilde} after {i+2} ground state searchs.")
                    excitation_engine = VariationalQuasiparticleExcitationEngine(mps0, mpo, p=None, \
                                                                                 mps0_tilde=mps0_tilde)
                    es, empss = excitation_engine.run(k+1)
                    es, empss = es[1:], empss[1:]
                    break
        print("")
        print(f"Found {k} variational quasiparticle excitations on top of ground state.")
        print(f"Energies: {np.array(es)}.")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump((es, empss), pkl_file)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return

def plot_excitations_local_energies(N, g, D_max):
    """Both for PBC and OBC, plot the local energies of the four lowest lying excitations. Also show
    that the total energies lie on the dispersion curve, for respective momenta (PBC) and wave 
    number (OBC)."""
    script_path = Path(__file__).resolve().parent
    # figure 
    png_path = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_local_energies.png"
    fig, ax = plt.subplots(2, 2, figsize=(15, 7))
    ax[0, 0].set_title("PBC")
    ax[0, 0].set_ylabel(r"$\langle h_{n, n+1} \rangle$")
    ax[1, 0].set_title("OBC")
    ax[1, 0].set_xlabel(r"bond $n$")
    ax[1, 0].set_ylabel(r"$\langle h_{n, n+1} \rangle$")
    ax[0, 1].set_xlabel(r"wave number $p$")
    ax[0, 1].set_xticks(np.arange(-np.pi, 3*np.pi/2, np.pi/2), [r"$-\pi$",r"$-\pi /2$", r"$0$", r"$\pi /2$",r"$\pi$"])
    ax[0, 1].set_ylabel(r"$\epsilon_p$")
    ax[0, 1].grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.7)
    # model and local Hamiltonians
    tfi_model_finite = TFIModelFinite(N, g)
    h_bonds = tfi_model_finite.get_h_bonds()
    # PBC
    pkl_path_pbc = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_pbc_{N}_{g}_{D_max}.pkl"
    es_pbc = None
    try:
        with open(pkl_path_pbc, "rb") as pkl_file_pbc:
            empss_sup_p, _, es_pbc, empss = pickle.load(pkl_file_pbc)
    except FileNotFoundError:
        print(f"No data available for PBC")
    _, mps0, _ = run_dmrg(N, "periodic", g, D_max, guess_mps0="product")
    E_bonds_gs = mps0.get_bond_expectation_values(h_bonds)
    E_bonds_1p = empss_sup_p[1].get_bond_expectation_values(h_bonds)
    e_bonds_1p = [E_bonds_1p[n] - E_bonds_gs[n] for n in range(1, N-2)]
    E_bonds_1m = empss_sup_p[2].get_bond_expectation_values(h_bonds)
    e_bonds_1m = [E_bonds_1m[n] - E_bonds_gs[n] for n in range(1, N-2)]
    E_bonds_2p = empss_sup_p[3].get_bond_expectation_values(h_bonds)
    e_bonds_2p = [E_bonds_2p[n] - E_bonds_gs[n] for n in range(1, N-2)]
    E_bonds_2m = empss_sup_p[4].get_bond_expectation_values(h_bonds)
    e_bonds_2m = [E_bonds_2m[n] - E_bonds_gs[n] for n in range(1, N-2)]
    colors_pbc = ["cyan", "lightskyblue", "blue", "darkblue"]
    ax[0, 0].plot(range(2, N-1), e_bonds_1p, "-", color=colors_pbc[0], label=rf"PBC $\vert k = 1 \rangle + \vert k = -1 \rangle$")
    ax[0, 0].plot(range(2, N-1), e_bonds_1m, "-", color=colors_pbc[1], label=rf"$\vert 1 \rangle - \vert -1 \rangle$")
    ax[0, 0].plot(range(2, N-1), e_bonds_2p, "-", color=colors_pbc[2], label=rf"$\vert 2 \rangle + \vert -2 \rangle$")
    ax[0, 0].plot(range(2, N-1), e_bonds_2m, "-", color=colors_pbc[3], label=rf"$\vert 2 \rangle - \vert -2 \rangle$")
    # OBC
    pkl_path_obc = script_path.parent / "data" / "mps_ground_states_excitations" / f"excitations_obc_{N}_{g}_{D_max}.pkl"
    es_obc = None
    try:
        with open(pkl_path_obc, "rb") as pkl_file_obc:
            es_obc, empss = pickle.load(pkl_file_obc)
    except FileNotFoundError:
        print(f"No data available for OBC")
    # local energies
    _, mps0, _= run_dmrg(N, "open", g, D_max, guess_mps0="product")
    E_bonds_gs = mps0.get_bond_expectation_values(h_bonds)
    E_bonds_1 = empss[0].get_bond_expectation_values(h_bonds)
    e_bonds_1 = [E_bonds_1[n] - E_bonds_gs[n] for n in range(1, N-2)]
    E_bonds_2 = empss[1].get_bond_expectation_values(h_bonds)
    e_bonds_2 = [E_bonds_2[n] - E_bonds_gs[n] for n in range(1, N-2)]
    E_bonds_3 = empss[2].get_bond_expectation_values(h_bonds)
    e_bonds_3 = [E_bonds_3[n] - E_bonds_gs[n] for n in range(1, N-2)]
    E_bonds_4 = empss[3].get_bond_expectation_values(h_bonds)
    e_bonds_4 = [E_bonds_4[n] - E_bonds_gs[n] for n in range(1, N-2)]
    colors_obc = ["gold", "olivedrab", "green", "darkslategray"]
    ax[1, 0].plot(range(2, N-1), e_bonds_1, "-", color=colors_obc[0], label=rf"OBC $\vert k = 1 \rangle$")
    ax[1, 0].plot(range(2, N-1), e_bonds_2, "-", color=colors_obc[1], label=rf"$\vert 2 \rangle$")
    ax[1, 0].plot(range(2, N-1), e_bonds_3, "-", color=colors_obc[2], label=rf"$\vert 3 \rangle$")
    ax[1, 0].plot(range(2, N-1), e_bonds_4, "-", color=colors_obc[3], label=rf"$\vert 4 \rangle$")
    # (quasi) dispersion relation
    ps_pbc, es_pbc_exact, ps_con, es_con = tfi_model_finite.get_exact_excitation_dispersion()
    ps_pbc = [2* np.pi * p / N for p in ps_pbc]
    ps_con = [2* np.pi * p / N for p in ps_con]
    ps_obc = [np.pi * p / (N+1) for p in range(1, N+1)]
    es_obc_exact = 2 * np.sqrt(g**2 - 2 * g * np.cos(ps_obc) + 1)
    ax[0, 1].plot(ps_con, es_con, "-", color="lightgray", zorder=1)
    ax[0, 1].plot(ps_pbc, es_pbc_exact, "x", color="lightgray", markersize=3, zorder=2)
    ax[0, 1].plot(ps_obc, es_obc_exact, ".", color="lightgray", markersize=3, zorder=3)
    ax[0, 1].scatter([2*np.pi/N, -2*np.pi/N, 4*np.pi/N, -4*np.pi/N], es_pbc[1:5], c=["royalblue"]*4, marker="x", s=15, zorder=4)
    ax[0, 1].scatter(ps_obc[:4], es_obc[:4], c=colors_obc, s=10, zorder=5)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(ax[0, 1], width="60%", height="60%", loc="upper center")
    axins.plot(ps_con, es_con, "-", color="lightgray", zorder=1, label=r"$\epsilon_p = 2 \sqrt{g^2 - 2g\cos(p) + 1}$")
    axins.plot(ps_pbc, es_pbc_exact, "x", color="lightgray", markersize=6, zorder=2, label=r"PBC momenta $p = \frac{2\pi}{N}k$")
    axins.plot(ps_obc, es_obc_exact, ".", color="lightgray", markersize=7, zorder=3, label=r"OBC wave numbers $p = \frac{\pi}{N+1}k$")
    axins.scatter([2*np.pi/N, -2*np.pi/N, 4*np.pi/N, -4*np.pi/N], es_pbc[1:5], c=["royalblue"]*4, marker="x", s=70, zorder=4)
    axins.scatter(ps_obc[:4], es_obc[:4], c=colors_obc, s=35, zorder=5)
    axins.set_xlim(-11*np.pi/N, 11*np.pi/N)
    axins.set_ylim(0.98, 1.2)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax[0, 1], axins, loc1=3, loc2=4, fc="none", ec="0.5")
    # legends and save
    handles0, labels0 = axins.get_legend_handles_labels()
    handles1, labels1 = ax[0, 0].get_legend_handles_labels()
    handles2, labels2 = ax[1, 0].get_legend_handles_labels()
    ax[1, 1].legend(handles0 + handles1 + handles2, labels0 + labels1 + labels2, loc='lower left', bbox_to_anchor=(0, -0.15))
    ax[1, 1].axis('off')
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    return