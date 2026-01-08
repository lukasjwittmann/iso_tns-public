"""Toy code implementing variational quasiparticle excitations on top of a diagonal isometric PEPS
ground state."""

import numpy as np
import opt_einsum as oe
from scipy.linalg import null_space
from scipy.sparse.linalg import LinearOperator, eigsh
from functools import reduce
from copy import deepcopy

from ..matrix_decompositions import qr_positive, svd_truncation
from .a_iso_peps.src.isoTPS.square.isoTPS import isoTPS_Square as DiagonalIsometricPEPS
from .b_model import DiagonalSquareLattice, TFIModelDiagonalSquare
from .c_mps import MPS, TwoSiteSweep
from .d_expectation_values import get_flipped_As, get_flipped_hs, get_flipped_Cs, get_flipped_mps, \
                                  subtract_energy_offset_mpos, get_expectation_value_boundary

from ..mps.b_model_finite import TFIModelFinite


class VariationalQuasiparticleExcitationsEngine:
    "..."
    def __init__(self, D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, h_mpos, bc, chi_max_b, eps_b=1.e-15):
        # lattice parameters
        self.Nx = len(ALs)
        self.Lx = self.Nx // 2
        self.Ly = len(ALs[0])
        self.Ny = 2 * self.Ly - 1
        self.N = 2 * self.Lx * self.Ly
        # ground state
        self.d = np.shape(ALs[0][0])[0]
        self.D_max = D_max
        self.chi_max_c = chi_max_c
        self.ALs = ALs
        self.ARs = ARs
        self.CDs = CDs
        self.CCs = CCs
        self.CUs = CUs
        # shapes of excitation tensors
        self.VLs = get_VLs(self.ALs)
        self.VDs = get_VDs(self.CDs[-1])
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(self.ALs, self.CDs, self.CCs, self.CUs)
        self.shape_Xs_column, self.shape_vecX_column = get_shape_Xs_vecX_column(self.CDs[-1])
        self.ADs, self.AUs = get_ADs_AUs(self.ALs, self.CDs, self.CUs)
        # Hamiltonian (and ground state energy)
        self.h_mpos = h_mpos
        """
        es0 = iso_peps0.copy().get_column_expectation_values(h_mpos)
        if zeroE:
            subtract_energy_offset_mpos(h_mpos, es0)
            es0 = iso_peps0.copy().get_column_expectation_values(h_mpos)
        self.E0 = np.sum(es0)
        print(f"Initialize excitation engine from ground state isoPEPS with E0 = {self.E0}.")
        """
        # boundary compression parameters
        self.bc = bc
        self.chi_max_b = chi_max_b
        self.eps_b = eps_b
        print(f"Initialize excitation engine with boundary compression = {bc}, chi_max_b = {chi_max_b}.")
        # boundaries only containing the Hamiltonian
        #self.bc = "column"
        self.Lhs = self.get_Lhs()
        self.Rhs = self.get_Rhs()
        print("Compressed boundaries Lhs and Rhs only containing the Hamiltonian.")
        print(f"-> (Lh|C) = {get_expectation_value_boundary(self.CDs[-1], self.Lhs[-1], "left")}. \n")
        #self.bc = bc

    def run(self, k, N=None):
        """
        if vecX_guess == "random":
            np.random.seed(0)
            vecX_guess = np.random.randn(self.shape_vecX + self.shape_vecX_column) \
                         + 1.j * np.random.randn(self.shape_vecX + self.shape_vecX_column)
            vecX_guess /= np.linalg.norm(vecX_guess)
        elif vecX_guess == "spin_flip":
            Bs_guess = Bs_from_spin_flip(g=3.5, k=1, ALs=self.ALs, CCs=self.CCs, CDs=self.CDs, direction="x")
            vecX_guess = vecX_from_non_orthogonal_Bs(self.ALs, self.ARs, self.CDs, self.CCs, self.CUs, Bs_guess, self.chi_max_b)
        E_guess = np.inner(np.conj(vecX_guess), Heff(self)._matvec(vecX_guess))
        print(f"E_guess = {E_guess}.")
        vecXs = []
        Es = []
        if N is None:
            for i in range(k):
                Bs_guess = Bs_from_spin_flip(g=3.5, k=i+1, ALs=self.ALs, CCs=self.CCs, CDs=self.CDs, direction="x")
                vecX_guess = vecX_from_non_orthogonal_Bs(self.ALs, self.ARs, self.CDs, self.CCs, self.CUs, Bs_guess, self.chi_max_b)
                H_eff = Heff(self, deflation_vecXs=vecXs)
                E_guess = np.inner(np.conj(vecX_guess), H_eff._matvec(vecX_guess))
                print(f"E_guess = {E_guess}.")
                Es_gs, vecXs_gs = eigsh(H_eff, k=1, which="SA", maxiter=50, tol=1.e-4, v0=vecX_guess)
                print(f"=> E{i} = {Es_gs[0]}. \n")
                Es.append(Es_gs[0])
                vecXs.append(vecXs_gs[:, 0])
        print("Spin flips with coefficients from perturbation theory:")
        for i in range(k):
            Bs_guess = Bs_from_spin_flip(g=3.5, k=i+1, ALs=self.ALs, CCs=self.CCs, CDs=self.CDs, direction="x")
            vecX_guess = vecX_from_non_orthogonal_Bs(self.ALs, self.ARs, self.CDs, self.CCs, self.CUs, Bs_guess, self.chi_max_b)
            E_guess = np.inner(np.conj(vecX_guess), Heff(self)._matvec(vecX_guess))
            print(f"=> E{i+1}_guess = {E_guess}.")
        """
        print("")
        print(f"Optimize {k} excitation(s) above the ground state from effective Hamiltonian.")
        if N is None:
            Es, vecXs = eigsh(Heff(self), k=k, which="SA", maxiter=50, tol=1.e-4)
        else:
            T, phis = lanczos(vecX_guess, Heff(self), N, stabilize=True)
            V = np.array(phis).transpose()
            Es, us = np.linalg.eigh(T)
            vecXs = np.dot(V, us)
        vecXs = [vecXs[:, i] for i in range(k)]
        print(f"Optimization of {np.shape(vecXs[0])[0]} excitation parameters done.")
        """
        for i in range(k):
            print(f"- Excitation {i+1}:")
            vecX = vecXs[i][:self.shape_vecX]
            vecX_column = vecXs[i][self.shape_vecX:]
            self.print_all_excitation_norms(vecX, vecX_column)
            print(f"=> E_{i+1} = {Es[i]}.")
            print(f"=> E_{i+1} = {Es[i]} (e{i+1} = {Es[i] - self.E0}).")
        """
        return Es, vecXs
    
    def print_all_excitation_norms(self, vecX, vecX_column):
        Xs = vec_to_tensors(vecX, self.shape_Xs)
        Xs_column = vec_to_tensors_column(vecX_column, self.shape_Xs_column)
        print("excitations AL-VL-X-AR:")
        for nx in range(self.Nx):
            for y in range(self.Ly):
                if Xs[nx][y] is not None:
                    X = Xs[nx][y].copy()
                    print(f"> {np.shape(X)} excitation parameters at site {nx,y} " \
                          + f"with ||X_{nx,y}||^2 = {np.linalg.norm(X)**2}.")
        print("excitations AL-AL-AL-X_column:")
        for ny in range(self.Ny):
            if Xs_column[ny] is not None:
                X_column = Xs_column[ny].copy()
                print(f"> {np.shape(X_column)} excitation parameters on bond {self.Nx-1,ny} " \
                      + f"with ||X_column_{ny}||^2 = {np.linalg.norm(X_column)**2}.")
        X2 = np.linalg.norm(vecX)**2 + np.linalg.norm(vecX_column)**2
        print(f"-> {self.shape_vecX} + {self.shape_vecX_column} = {self.shape_vecX + self.shape_vecX_column} " \
              + f"excitation parameters with ||X||^2 + ||X_column||^2 = {X2}.")
        return

    # conversions vecX <-> Bs
    def vecX_to_Bs(self, vecX):
        assert np.shape(vecX) == (self.shape_vecX + self.shape_vecX_column,)
        Xs = vec_to_tensors(vecX[:self.shape_vecX], self.shape_Xs)
        Bs = Xs_to_Bs(Xs, self.VLs)
        Xs_column = vec_to_tensors_column(vecX[self.shape_vecX:], self.shape_Xs_column)
        Bs_column = Xs_column_to_Bs_column(Xs_column, self.VDs)
        Bs_double = Bs_column_to_Bs(Bs_column, self.ALs[-1], self.CDs[-1], self.CUs[-1])
        for y in range(self.Ly):
            if Bs_double[y] is not None:
                if Bs[-1][y] is not None:
                    Bs[-1][y] += Bs_double[y]
                else:
                    Bs[-1][y] = Bs_double[y]
        return Bs
        
    def Bs_to_vecX(self, Bs):
        assert len(Bs) == self.Nx and len(Bs[0]) == self.Ly
        Xs = Bs_to_Xs(Bs, self.VLs)
        vecX = tensors_to_vec(Xs, self.shape_vecX)
        Xs_column = Bs_to_Xs_column(Bs[-1], self.ALs[-1], self.CDs[-1], self.CUs[-1], self.VDs)
        vecX_column = tensors_to_vec_column(Xs_column, self.shape_vecX_column)
        return np.hstack([vecX, vecX_column])
    
    def test_Bs(self):
        print("Test conversions vecX -> Bs -> vecX_new.")
        vecX = np.random.normal(size=(self.shape_vecX + self.shape_vecX_column,)) \
               + 1.j * np.random.normal(size=(self.shape_vecX + self.shape_vecX_column,))
        vecX /= np.linalg.norm(vecX)
        # vecX -> Bs -> vecX_new =? vecX
        Bs = self.vecX_to_Bs(vecX)
        vecX_new = self.Bs_to_vecX(Bs)
        print(f"- ||vecX - vecX_new|| = {np.linalg.norm(vecX - vecX_new)}.")
        # (vecX*|vecX) =? sum_nx (Bs_sum[nx]*|Bs_sum[nx]) for Bs_sum[nx] flipped
        overlap_vecX = np.inner(np.conj(vecX), vecX)
        Bs_sum = Bs_to_Bs_sum(Bs, self.ADs, self.AUs)
        Bs_sum = [get_flipped_Bs_sum(B_sum) for B_sum in Bs_sum]
        overlaps = []
        for nx in range(self.Nx):
            if Bs_sum[nx] is not None:
                overlap = np.ones((1, 1))
                for y in reversed(range(self.Ly)):
                    overlap = oe.contract("ab,cadefgh,ibdefgh->ci", \
                                          overlap, Bs_sum[nx][y], np.conj(Bs_sum[nx][y]))
                assert np.shape(overlap) == (1, 1)
                overlaps.append(np.real_if_close(overlap[0, 0]))
        overlap = sum(overlaps)
        print(f"- |(vecX*|vecX) - sum_nx (Bs_sum[nx]*|Bs_sum[nx])| = {np.abs(overlap_vecX - overlap)}.")
    
    # boundary compressions
    def perform_boundary_compression(self, Ls_list, Cs=None):
        if np.any([Ls is not None for Ls in Ls_list]):
            if self.bc == "variational":
                boundary_compression = BoundaryCompression(Ls_list, self.chi_max_b, self.eps_b)
                boundary_compression.run(N_sweeps=3)
                L = boundary_compression.psi
                #print(f"(maximal truncation error: {np.max(boundary_compression.trunc_errors)})")
                return L
            elif self.bc == "column":
                assert Cs is not None
                boundary_compression = BoundaryColumnCompression(Ls_list, Cs, self.chi_max_b, self.eps_b)
                L = boundary_compression.run()
                return L
        else:
            return None

    # boundaries only containing the Hamiltonian (saved as class attributes)
    def get_Lhs(self):
        Lhs = [None] * self.Nx
        for nx in range(1, self.Nx):
            # extract all needed tensors
            A1s, A2s = deepcopy(self.ALs[nx-1]), deepcopy(self.ALs[nx])
            hs = deepcopy(self.h_mpos[nx-1])
            Lh = Lhs[nx-1].copy() if Lhs[nx-1] is not None else None
            Cs = deepcopy(self.CDs[nx])
            # flip tensors for even nx
            if nx%2 == 0:
                A1s, A2s = get_flipped_As(A1s), get_flipped_As(A2s)
                hs = get_flipped_hs(hs)
                Lh = get_flipped_mps(Lh)
                Cs = get_flipped_Cs(Cs)
            # perform boundary compression
            if self.bc == "variational":
                if hs is None and Lh is None:
                    Lh = None
                else:
                    try:
                        vbc = VBC_h(A1s, A2s, hs, Lh, self.chi_max_b)
                        vbc.run(N_sweeps=3)
                        Lh = vbc.psi
                    except AssertionError:
                        Lh = None
            elif self.bc == "column":
                CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side="left")
                Ls_list = [get_Ls_h(hs, A1s, A2s, A1s, A2s), \
                        get_Ls_transfer(Lh, A2s, A2s)]
                Lh = self.perform_boundary_compression(Ls_list, CCdaggers)
            """
            CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side="left")
            Ls_list = [get_Ls_h(hs, A1s, A2s, A1s, A2s), \
                    get_Ls_transfer(Lh, A2s, A2s)]
            Lh = self.perform_boundary_compression(Ls_list, CCdaggers)
            """
            # flip boundary mps back for even nx
            if nx%2 == 0:
                Lh = get_flipped_mps(Lh)
            # save boundary mps
            Lhs[nx] = Lh
        return Lhs

    def get_Rhs(self):
        Rhs = [None] * self.Nx
        for nx in reversed(range(1, self.Nx-1)):
            # extract all needed tensors
            A1s, A2s = deepcopy(self.ARs[nx+1]), deepcopy(self.ARs[nx])
            hs = deepcopy(self.h_mpos[nx])
            Rh = Rhs[nx+1].copy() if Rhs[nx+1] is not None else None
            Cs = deepcopy(self.CDs[nx-1])
            # flip tensors for even nx
            if nx%2 == 0:
                A1s, A2s = get_flipped_As(A1s), get_flipped_As(A2s)
                hs = get_flipped_hs(hs)
                Rh = get_flipped_mps(Rh)
                Cs = get_flipped_Cs(Cs)
            # perform boundary compression
            if self.bc == "variational":
                if hs is None and Rh is None:
                    Rh = None
                else:
                    try:
                        vbc = VBC_h(A1s, A2s, hs, Rh, self.chi_max_b)
                        vbc.run(N_sweeps=3)
                        Rh = vbc.psi
                    except AssertionError:
                        Rh = None
            elif self.bc == "column":
                CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side="right")
                Rs_list = [get_Ls_h(hs, A1s, A2s, A1s, A2s), \
                        get_Ls_transfer(Rh, A2s, A2s)]
                Rh = self.perform_boundary_compression(Rs_list, CCdaggers)
            """
            CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side="right")
            Rs_list = [get_Ls_h(hs, A1s, A2s, A1s, A2s), \
                    get_Ls_transfer(Rh, A2s, A2s)]
            Rh = self.perform_boundary_compression(Rs_list, CCdaggers)
            """
            # flip boundary mps back for even nx
            if nx%2 == 0:
                Rh = get_flipped_mps(Rh)
            # save boundary mps
            Rhs[nx] = Rh
        return Rhs
    
    # boundaries containing Bs_sum
    def get_RBs(self, Bs_sum):
        RBs = [None] * self.Nx
        for nx in reversed(range(1, self.Nx)):
            # extract all needed tensors
            Bs_ket = [np.transpose(B, (0, 1, 2, 5, 6, 3, 4)) for B in deepcopy(Bs_sum[nx])] if Bs_sum[nx] is not None else None
            As_ket = [np.transpose(AL, (0, 3, 4, 1, 2)) for AL in deepcopy(self.ALs[nx])]
            As_bra = deepcopy(self.ARs[nx])
            RB = RBs[nx+1].copy() if nx < self.Nx-1 and RBs[nx+1] is not None else None
            if nx%2 == 1:
                Cs = [np.conj(CD) for CD in deepcopy(self.CDs[nx-1])]
            elif nx%2 == 0:
                Cs = [np.conj(CU) for CU in deepcopy(self.CUs[nx-1])]
            # flip tensors for even nx
            if nx%2 == 0:
                Bs_ket, As_bra = get_flipped_Bs_sum(Bs_ket), get_flipped_As(As_bra)
                As_ket = get_flipped_As(As_ket)
                RB = get_flipped_mps(RB)
                Cs = get_flipped_Cs(Cs)
            # perform boundary compression
            if self.bc == "variational":
                if Bs_ket is None and RB is None:
                    RB = None
                else:
                    try:
                        vbc = VBC_B(Bs_ket, As_bra, As_ket, RB, self.chi_max_b)
                        vbc.run(N_sweeps=3)
                        RB = vbc.psi
                    except AssertionError:
                        RB = None
            elif self.bc == "column":
                Rs_list = [get_Ls_B(Bs_ket, As_bra), \
                           get_Ls_transfer(RB, As_ket, As_bra)]
                RB = self.perform_boundary_compression(Rs_list, Cs)
            """
            Rs_list = [get_Ls_B(Bs_ket, As_bra), \
                        get_Ls_transfer(RB, As_ket, As_bra)]
            RB = self.perform_boundary_compression(Rs_list, Cs)
            """
            # flip boundary mps back for even nx
            if nx%2 == 0:
                RB = get_flipped_mps(RB)
            # save boundary mps
            RBs[nx] = RB
        return RBs
    
    def get_LhBs(self, Bs_sum):
        LhBs = [None] * self.Nx
        for nx in range(1, self.Nx):
            # extract all needed tensors
            A1s_bra, A2s_bra = deepcopy(self.ALs[nx-1]), deepcopy(self.ALs[nx])
            hs = deepcopy(self.h_mpos[nx-1])
            B1s_ket = deepcopy(Bs_sum[nx-1]) if Bs_sum[nx-1] is not None else None
            A2s_ket = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx])]
            A1s_ket = deepcopy(self.ALs[nx-1])
            B2s_ket = deepcopy(Bs_sum[nx]) if Bs_sum[nx] is not None else None
            Lh = self.Lhs[nx-1].copy() if self.Lhs[nx-1] is not None else None
            LhB = LhBs[nx-1].copy() if LhBs[nx-1] is not None else None
            if nx%2 == 1:
                Cs = [np.transpose(np.conj(CD), (0, 2, 1, 3)) for CD in deepcopy(self.CDs[nx])]
            elif nx%2 == 0:
                Cs = [np.transpose(np.conj(CU), (0, 2, 1, 3)) for CU in deepcopy(self.CUs[nx])]
            # flip tensors for even nx
            if nx%2 == 0:
                A1s_bra, A2s_bra = get_flipped_As(A1s_bra), get_flipped_As(A2s_bra)
                hs = get_flipped_hs(hs)
                B1s_ket, A2s_ket = get_flipped_Bs_sum(B1s_ket), get_flipped_As(A2s_ket)
                A1s_ket, B2s_ket = get_flipped_As(A1s_ket), get_flipped_Bs_sum(B2s_ket)
                Lh = get_flipped_mps(Lh)
                LhB = get_flipped_mps(LhB)
                Cs = get_flipped_Cs(Cs)
            # perform boundary compression
            if self.bc == "variational":
                if B1s_ket is None and B2s_ket is None and LhB is None:
                    LhB = None
                else:
                    try:
                        vbc = VBC_Bh(hs, B1s_ket, B2s_ket, A1s_bra, A2s_bra, A2s_ket, Lh, LhB, self.chi_max_b)
                        vbc.run(N_sweeps=3)
                        LhB = vbc.psi
                    except AssertionError:
                        LhB = None
            elif self.bc == "column":
                Ls_list = [get_Ls_Bh(hs, B1s_ket, A2s_ket, A1s_bra, A2s_bra), \
                        get_Ls_hB(hs, A1s_ket, B2s_ket, A1s_bra, A2s_bra), \
                        get_Ls_LhB(Lh, B2s_ket, A2s_bra), \
                        get_Ls_transfer(LhB, A2s_ket, A2s_bra)]
                LhB = self.perform_boundary_compression(Ls_list, Cs)
            """
            Ls_list = [get_Ls_Bh(hs, B1s_ket, A2s_ket, A1s_bra, A2s_bra), \
                    get_Ls_hB(hs, A1s_ket, B2s_ket, A1s_bra, A2s_bra), \
                    get_Ls_LhB(Lh, B2s_ket, A2s_bra), \
                    get_Ls_transfer(LhB, A2s_ket, A2s_bra)]
            LhB = self.perform_boundary_compression(Ls_list, Cs)
            """
            # flip boundary mps back for even nx
            if nx%2 == 0:
                LhB = get_flipped_mps(LhB)
            # save boundary mps
            LhBs[nx] = LhB
        return LhBs
            
    # All nonzero contributions to Heff|B)
    def get_Bs_new_1(self, Bs, Bs_sum):
        # nx = 0, ..., Nx-2
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(self.Nx-1):
            if np.any([B is not None for B in Bs[nx]]):
                # extract all needed tensors
                B1s_ket, A2s_ket = deepcopy(Bs_sum[nx]), [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])]
                AD1s_bra, AU1s_bra, A2s_bra = deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx]), deepcopy(A2s_ket)
                hs = deepcopy(self.h_mpos[nx])
                # flip tensors for odd nx
                if nx%2 == 1:
                    B1s_ket, A2s_ket = get_flipped_Bs_sum(B1s_ket), get_flipped_As(A2s_ket)
                    AD1s_bra, AU1s_bra = get_flipped_Bs(AU1s_bra), get_flipped_Bs(AD1s_bra)
                    A2s_bra = get_flipped_As(A2s_bra)
                    hs = get_flipped_hs(hs)
                # compute up and down environments
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abcde,afghicj,dklg,bmlhien,ojpqr,ksto,tnuqr->fmpsu", \
                                           DPs[y], \
                                           B1s_ket[y], hs[2*y], np.conj(AD1s_bra[y]), \
                                           A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = oe.contract("abcde,fgha,hicde->gbfi", \
                                      A2s_ket[-1], hs[-1], np.conj(A2s_bra[-1]))[:, np.newaxis, :, :, :]
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abcde,faghijc,kdlg,mblhine,opjqr,skto,tunqr->fmpsu", \
                                           UPs[y], \
                                           B1s_ket[y], hs[2*y], np.conj(AU1s_bra[y]), \
                                           A2s_ket[y-1], hs[2*y-1], np.conj(A2s_bra[y-1]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abcde,afghicj,dklg,fmjkn->bmlhien", \
                                                    DPs[y], B1s_ket[y], hs[2*y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])
        return Bs_new

    def get_Bs_new_2(self, Bs, Bs_sum):
        # nx = 0, ..., Nx-2
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(self.Nx-1):
            if np.any([B is not None for B in Bs[nx]]) and Bs_sum[nx+1] is not None:
                # extract all needed tensors
                A1s_ket, B2s_ket = deepcopy(self.ALs[nx]), deepcopy(Bs_sum[nx+1])
                AD1s_bra, AU1s_bra = deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                A2s_bra = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])]
                hs = deepcopy(self.h_mpos[nx])
                # flip tensors for odd nx
                if nx%2 == 1:
                    A1s_ket, B2s_ket = get_flipped_As(A1s_ket), get_flipped_Bs_sum(B2s_ket)
                    AD1s_bra, AU1s_bra = get_flipped_Bs(AU1s_bra), get_flipped_Bs(AD1s_bra)
                    A2s_bra = get_flipped_As(A2s_bra)
                    hs = get_flipped_hs(hs)
                # compute up and down environments
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abcde,fghci,djkf,blkghem,anoipqr,jsto,tmuqr->nlpsu", \
                                           DPs[y], \
                                           A1s_ket[y], hs[2*y], np.conj(AD1s_bra[y]), \
                                           B2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = oe.contract("abcdefg,hijc,jkefg->iabdhk", \
                                      B2s_ket[-1], hs[-1], np.conj(A2s_bra[-1]))[0, :, :, :, :, :]
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abcde,fghic,jdkf,lbkghme,naopiqr,sjto,tumqr->nlpsu", \
                                           UPs[y], \
                                           A1s_ket[y], hs[2*y], np.conj(AU1s_bra[y]), \
                                           B2s_ket[y-1], hs[2*y-1], np.conj(A2s_bra[y-1]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abcde,fghci,djkf,alijm->blkghem", \
                                                    DPs[y], A1s_ket[y], hs[2*y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])
        return Bs_new

    def get_Bs_new_3(self, Bs, RBs):
        # nx = 0, ..., Nx-3
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(self.Nx-2):
            if np.any([B is not None for B in Bs[nx]]) and RBs[nx+2] is not None:
                # extract all needed tensors
                A1s_ket, A2s_ket = deepcopy(self.ALs[nx]), deepcopy(self.ALs[nx+1])
                AD1s_bra, AU1s_bra = deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                A2s_bra = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])]
                hs = deepcopy(self.h_mpos[nx])
                RB = mps_to_tensors(RBs[nx+2])
                # flip tensors for odd nx
                if nx%2 == 1:
                    A1s_ket, A2s_ket = get_flipped_As(A1s_ket), get_flipped_As(A2s_ket)
                    AD1s_bra, AU1s_bra = get_flipped_Bs(AU1s_bra), get_flipped_Bs(AD1s_bra)
                    A2s_bra = get_flipped_As(A2s_bra)
                    hs = get_flipped_hs(hs)
                    RB = get_flipped_Cs(RB)
                # compute up and down environments
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abcde,fghci,djkf,blkghem,niopq,jrsn,smtuv,apuw,wqvx->xlort", \
                                           DPs[y], \
                                           A1s_ket[y], hs[2*y], np.conj(AD1s_bra[y]), \
                                           A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]), \
                                           RB[2*y], RB[2*y+1])
                UPs = [None] * self.Ly
                UPs[-1] = oe.contract("abcde,fgha,hicje,kdjl->gklbfi", \
                                      A2s_ket[-1], hs[-1], np.conj(A2s_bra[-1]), RB[-1])[0, :, :, :, :, :]
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abcde,fghic,jdkf,lbkghme,noipq,rjsn,stmuv,wpux,xqva->wlort", \
                                           UPs[y], A1s_ket[y], hs[2*y], np.conj(AU1s_bra[y]), \
                                           A2s_ket[y-1], hs[2*y-1], np.conj(A2s_bra[y-1]), \
                                           RB[2*(y-1)], RB[2*y-1])
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abcde,fghci,djkf,alijm->blkghem", \
                                                    DPs[y], A1s_ket[y], hs[2*y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])
        return Bs_new
    
    def get_Bs_new_4(self, Bs, Bs_sum):
        # nx = 0, ..., Nx-3
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(self.Nx-2):
            if np.any([B is not None for B in Bs[nx]]) and self.Rhs[nx+1] is not None:
                # extract all needed tensors
                Bs_ket, ADs_bra, AUs_bra = deepcopy(Bs_sum[nx]), deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                Rh = mps_to_tensors(self.Rhs[nx+1])
                # flip tensors for odd nx
                if nx%2 == 1:
                    Bs_ket, ADs_bra, AUs_bra = get_flipped_Bs_sum(Bs_ket), get_flipped_Bs(AUs_bra), get_flipped_Bs(ADs_bra)
                    Rh = get_flipped_Cs(Rh)
                # compute up and down environments
                Rh = [np.ones((1, 1, 1, 1))] + Rh
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abc,adef,fghi,bjklmdg,cnklmeh->ijn", \
                                           DPs[y], Rh[2*y], Rh[2*y+1], Bs_ket[y], np.conj(ADs_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = np.ones((1, 1, 1))
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abc,defg,ghia,jbklmeh,ncklmfi->djn", \
                                           UPs[y], Rh[2*y], Rh[2*y+1], Bs_ket[y], np.conj(AUs_bra[y]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abc,adef,fghi,bjklmdg,ijn->cnklmeh", \
                                                    DPs[y], Rh[2*y], Rh[2*y+1], Bs_ket[y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                   
        return Bs_new

    def get_Bs_new_5(self, Bs, Bs_sum):
        # nx = 1, ..., Nx-1
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(1, self.Nx):
            if np.any([B is not None for B in Bs[nx]]) and Bs_sum[nx-1] is not None:
                # extract all needed tensors
                B1s_ket, A2s_ket = deepcopy(Bs_sum[nx-1]), [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx])]
                A1s_bra, AD2s_bra, AU2s_bra = deepcopy(self.ALs[nx-1]), deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                hs = deepcopy(self.h_mpos[nx-1])
                # flip tensors for odd nx
                if nx%2 == 1:
                    B1s_ket, A2s_ket = get_flipped_Bs_sum(B1s_ket), get_flipped_As(A2s_ket)
                    A1s_bra, AD2s_bra, AU2s_bra = get_flipped_As(A1s_bra), get_flipped_Bs(AU2s_bra), get_flipped_Bs(AD2s_bra)
                    hs = get_flipped_hs(hs)
                # compute up and down environments
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abcde,fcghi,djkf,blkemhi,anopqgr,jsto,tpqmu->nlrsu", \
                                           DPs[y], A2s_ket[y], hs[2*y], np.conj(AD2s_bra[y]), \
                                           B1s_ket[y], hs[2*y+1], np.conj(A1s_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = oe.contract("abcdefg,hijc,jdekg->iabfhk", \
                                      B1s_ket[-1], hs[-1], np.conj(A1s_bra[-1]))[0, :, :, :, :, :]
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abcde,fgchi,jdkf,lbkmehi,naopqrg,sjto,tpqum->nlrsu", \
                                           UPs[y], A2s_ket[y], hs[2*y], np.conj(AU2s_bra[y]), \
                                           B1s_ket[y-1], hs[2*y-1], np.conj(A1s_bra[y-1]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abcde,fcghi,djkf,algjm->blkemhi", \
                                                    DPs[y], A2s_ket[y], hs[2*y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                    
        return Bs_new

    def get_Bs_new_6(self, Bs, Bs_sum):
        # nx = 1, ..., Nx-1
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(1, self.Nx):
            if np.any([B is not None for B in Bs[nx]]):
                # extract all needed tensors
                A1s_ket, B2s_ket = deepcopy(self.ALs[nx-1]), deepcopy(Bs_sum[nx])
                A1s_bra, AD2s_bra, AU2s_bra = deepcopy(self.ALs[nx-1]), deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                hs = deepcopy(self.h_mpos[nx-1])
                # flip tensors for odd nx
                if nx%2 == 1:
                    A1s_ket, B2s_ket = get_flipped_As(A1s_ket), get_flipped_Bs_sum(B2s_ket)
                    A1s_bra, AD2s_bra, AU2s_bra = get_flipped_As(A1s_bra), get_flipped_Bs(AU2s_bra), get_flipped_Bs(AD2s_bra)
                    hs = get_flipped_hs(hs)
                # compute up and down environments
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abcde,afgchij,dklg,bmlenij,opqhr,ksto,tpqnu->fmrsu", \
                                           DPs[y], B2s_ket[y], hs[2*y], np.conj(AD2s_bra[y]), \
                                           A1s_ket[y], hs[2*y+1], np.conj(A1s_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = oe.contract("abcde,fgha,hbcie->gdfi", \
                                      A1s_ket[-1], hs[-1], np.conj(A1s_bra[-1]))[np.newaxis, :, :, :, :]
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abcde,faghcij,kdlg,mblneij,opqrh,skto,tpqun->fmrsu", \
                                           UPs[y], B2s_ket[y], hs[2*y], np.conj(AU2s_bra[y]), \
                                           A1s_ket[y-1], hs[2*y-1], np.conj(A1s_bra[y-1]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abcde,afgchij,dklg,fmhkn->bmlenij", \
                                                    DPs[y], B2s_ket[y], hs[2*y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                   
        return Bs_new

    def get_Bs_new_7(self, Bs, RBs):
        # nx = 1, ..., Nx-2
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(1, self.Nx-1):
            if np.any([B is not None for B in Bs[nx]]) and RBs[nx+1] is not None:
                # extract all needed tensors
                A1s_ket, A2s_ket = deepcopy(self.ALs[nx-1]), deepcopy(self.ALs[nx])
                A1s_bra , AD2s_bra, AU2s_bra = deepcopy(self.ALs[nx-1]), deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                hs = deepcopy(self.h_mpos[nx-1])
                RB = mps_to_tensors(RBs[nx+1])
                # flip tensors for odd nx
                if nx%2 == 1:
                    A1s_ket, A2s_ket = get_flipped_As(A1s_ket), get_flipped_As(A2s_ket)
                    A1s_bra, AD2s_bra, AU2s_bra = get_flipped_As(A1s_bra), get_flipped_Bs(AU2s_bra), get_flipped_Bs(AD2s_bra)
                    hs = get_flipped_hs(hs)
                    RB = get_flipped_Cs(RB)
                # compute up and down environments
                RB = [np.ones((1, 1, 1, 1))] + RB
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abcde,fcghi,djkf,blkemno,ahnp,pioq,rstgu,jvwr,wstmx->qluvx", \
                                           DPs[y], A2s_ket[y], hs[2*y], np.conj(AD2s_bra[y]), \
                                           RB[2*y], RB[2*y+1], \
                                           A1s_ket[y], hs[2*y+1], np.conj(A1s_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = oe.contract("abcde,fgha,hbcie->gdfi", \
                                      A1s_ket[-1], hs[-1], np.conj(A1s_bra[-1]))[np.newaxis, :, :, :, :]
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abcde,fgchi,jdkf,lbkmeno,phnq,qioa,rstug,vjwr,wstxm->pluvx", \
                                           UPs[y], A2s_ket[y], hs[2*y], np.conj(AU2s_bra[y]), \
                                           RB[2*y], RB[2*y+1], \
                                           A1s_ket[y-1], hs[2*y-1], np.conj(A1s_bra[y-1]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abcde,fcghi,djkf,ahlm,mino,opgjq->bpkeqln", \
                                                    DPs[y], A2s_ket[y], hs[2*y], RB[2*y], RB[2*y+1], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                 
        return Bs_new

    def get_Bs_new_8(self, Bs, LhBs):
        # nx = 2, ..., Nx-1
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(2, self.Nx):
            if np.any([B is not None for B in Bs[nx]]) and LhBs[nx-1] is not None:
                # extract all needed tensors
                LhB = mps_to_tensors(LhBs[nx-1])
                As_ket = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx])]
                ADs_bra, AUs_bra = deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                # flip tensors for odd nx
                if nx%2 == 1:
                    LhB = get_flipped_Cs(LhB)
                    As_ket, ADs_bra, AUs_bra = get_flipped_As(As_ket), get_flipped_Bs(AUs_bra), get_flipped_Bs(ADs_bra)
                # compute up and down environments
                LhB = [np.ones((1, 1, 1, 1))] + LhB
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("ab,acde,efgh,icfjk,blidgjk->hl", \
                                           DPs[y], LhB[2*y], LhB[2*y+1], As_ket[y], np.conj(ADs_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = np.ones((1, 1))
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("ab,cdef,fgha,idgjk,lbiehjk->cl", \
                                           UPs[y], LhB[2*y], LhB[2*y+1], As_ket[y], np.conj(AUs_bra[y]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("ab,acde,efgh,icfjk,hl->blidgjk", \
                                                    DPs[y], LhB[2*y], LhB[2*y+1], As_ket[y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                    
        return Bs_new

    def get_Bs_new_9(self, Bs, Bs_sum):
        # nx = 2, ..., Nx-1
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(2, self.Nx):
            if np.any([B is not None for B in Bs[nx]]) and self.Lhs[nx-1] is not None:
                # extract all needed tensors
                Lh = mps_to_tensors(self.Lhs[nx-1])
                Bs_ket, ADs_bra, AUs_bra = deepcopy(Bs_sum[nx]), deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                # flip tensors for odd nx
                if nx%2 == 1:
                    Lh = get_flipped_Cs(Lh)
                    Bs_ket, ADs_bra, AUs_bra = get_flipped_Bs_sum(Bs_ket), get_flipped_Bs(AUs_bra), get_flipped_Bs(ADs_bra)
                # compute up and down environments
                Lh = [np.ones((1, 1, 1, 1))] + Lh
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abc,adef,fghi,bjkdglm,cnkehlm->ijn", \
                                           DPs[y], Lh[2*y], Lh[2*y+1], Bs_ket[y], np.conj(ADs_bra[y]))
                UPs = [None] * self.Ly
                UPs[-1] = np.ones((1, 1, 1))
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abc,defg,ghia,jbkehlm,nckfilm->djn", \
                                           UPs[y], Lh[2*y], Lh[2*y+1], Bs_ket[y], np.conj(AUs_bra[y]))
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abc,adef,fghi,bjkdglm,ijn->cnkehlm", \
                                                    DPs[y], Lh[2*y], Lh[2*y+1], Bs_ket[y], UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                   
        return Bs_new

    def get_Bs_new_10(self, Bs, RBs):
        # nx = 2, ..., Nx-2
        Bs_new = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(2, self.Nx-1):
            if np.any([B is not None for B in Bs[nx]]) and self.Lhs[nx-1] is not None and RBs[nx+1] is not None:
                # extract all needed tensors
                Lh = mps_to_tensors(self.Lhs[nx-1])
                As_ket, ADs_bra, AUs_bra = deepcopy(self.ALs[nx]), deepcopy(self.ADs[nx]), deepcopy(self.AUs[nx])
                RB = mps_to_tensors(RBs[nx+1])
                # flip tensors for odd nx
                if nx%2 == 1:
                    Lh = get_flipped_Cs(Lh)
                    As_ket, ADs_bra, AUs_bra = get_flipped_As(As_ket), get_flipped_Bs(AUs_bra), get_flipped_Bs(ADs_bra)
                    RB = get_flipped_Cs(RB)
                # compute up and down environments
                Lh = [np.ones((1, 1, 1, 1))] + Lh
                RB = [np.ones((1, 1, 1, 1))] + RB
                DPs = [None] * self.Ly
                DPs[0] = np.ones((1, 1, 1))
                for y in range(self.Ly-1):
                    DPs[y+1] = oe.contract("abc,adef,fghi,jdgkl,cmjehno,bknp,ploq->iqm", \
                                           DPs[y], \
                                           Lh[2*y], Lh[2*y+1], \
                                           As_ket[y], np.conj(ADs_bra[y]), \
                                           RB[2*y], RB[2*y+1])
                UPs = [None] * self.Ly
                UPs[-1] = np.ones((1, 1, 1))
                for y in range(self.Ly-1, 0, -1):
                    UPs[y-1] = oe.contract("abc,defg,ghia,jehkl,mcjfino,pknq,qlob->dpm", \
                                           UPs[y], \
                                           Lh[2*y], Lh[2*y+1], \
                                           As_ket[y], np.conj(AUs_bra[y]), \
                                           RB[2*y], RB[2*y+1])
                # compute new B tensors
                for y in range(self.Ly):
                    if nx%2 == 0:
                        Y = y
                    elif nx%2 == 1:
                        Y = self.Ly - 1 - y
                    if Bs[nx][Y] is not None:
                        Bs_new[nx][y] = oe.contract("abc,adef,fghi,jdgkl,bkmn,nlop,ipq->cqjehmo", \
                                                    DPs[y], \
                                                    Lh[2*y], Lh[2*y+1], \
                                                    As_ket[y], \
                                                    RB[2*y], RB[2*y+1], \
                                                    UPs[y])
                if nx%2 == 1:
                    Bs_new[nx] = get_flipped_Bs(Bs_new[nx])                 
        return Bs_new


class Heff(LinearOperator):
    def __init__(self, excitation_engine):
        self.excitation_engine = excitation_engine
        shape_vecX = excitation_engine.shape_vecX + excitation_engine.shape_vecX_column
        shape = (shape_vecX, shape_vecX)
        dtype = reduce(np.promote_types, [excitation_engine.ALs[0][0].dtype, \
                                          excitation_engine.CDs[0][0].dtype, \
                                          excitation_engine.h_mpos[0][0].dtype])
        super().__init__(dtype=dtype, shape=shape)
        self.matvec_counter = 0
        """
        if deflation_vecXs is not None:
            self.deflation_vecXs = [def_vecX/np.linalg.norm(def_vecX) for def_vecX in deflation_vecXs]
        else:
            self.deflation_vecXs = None
        """

    def _matvec(self, vecX):
        self.matvec_counter += 1
        print(f"matvec {self.matvec_counter}...")
        # vecX -> Bs -> Bs_sum
        Bs = self.excitation_engine.vecX_to_Bs(vecX)
        Bs_sum = Bs_to_Bs_sum(Bs, self.excitation_engine.ADs, self.excitation_engine.AUs)
        # compress boundaries containing Bs_sum
        RBs = self.excitation_engine.get_RBs(Bs_sum)
        print("compressed RBs.")
        LhBs = self.excitation_engine.get_LhBs(Bs_sum)
        print("compressed LhBs.")
        # compute all contributions to Bs_new
        Bs_new_1 = self.excitation_engine.get_Bs_new_1(Bs, Bs_sum)
        Bs_new_2 = self.excitation_engine.get_Bs_new_2(Bs, Bs_sum)
        Bs_new_3 = self.excitation_engine.get_Bs_new_3(Bs, RBs)
        Bs_new_4 = self.excitation_engine.get_Bs_new_4(Bs, Bs_sum)
        Bs_new_5 = self.excitation_engine.get_Bs_new_5(Bs, Bs_sum)
        Bs_new_6 = self.excitation_engine.get_Bs_new_6(Bs, Bs_sum)
        Bs_new_7 = self.excitation_engine.get_Bs_new_7(Bs, RBs)
        Bs_new_8 = self.excitation_engine.get_Bs_new_8(Bs, LhBs)
        Bs_new_9 = self.excitation_engine.get_Bs_new_9(Bs, Bs_sum)
        Bs_new_10 = self.excitation_engine.get_Bs_new_10(Bs, RBs)
        Bs_new = [[None] * self.excitation_engine.Ly for _ in range(self.excitation_engine.Nx)]
        for nx in range(self.excitation_engine.Nx):
            for y in range(self.excitation_engine.Ly):
                B_new_list = [Bs_new_1[nx][y], Bs_new_2[nx][y], Bs_new_3[nx][y], Bs_new_4[nx][y], \
                              Bs_new_5[nx][y], Bs_new_6[nx][y], Bs_new_7[nx][y], Bs_new_8[nx][y], \
                              Bs_new_9[nx][y], Bs_new_10[nx][y]]
                B_new_list = [B for B in B_new_list if B is not None]
                if B_new_list:
                    Bs_new[nx][y] = sum(B_new_list)
        # Bs_new -> vecX_new
        vecX_new = self.excitation_engine.Bs_to_vecX(Bs_new)
        """
        if self.deflation_vecXs is not None:
            for def_vecX in self.deflation_vecXs:
                vecX_new -= np.inner(np.conj(def_vecX), vecX_new) * def_vecX
        """
        return vecX_new
    

class ExcitedIsometricPEPS:
    def __init__(self, D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, vecX, bc, chi_max_b, eps_b=1.e-15):
        #print(f"Initialize ExcitedIsometricPEPS with {np.shape(vecX)[0]} excitations parameters.")
        # ground state iso_peps
        self.d = np.shape(ALs[0][0])[0]
        self.D_max = D_max
        self.chi_max_c = chi_max_c
        self.Nx = len(ALs)
        self.Lx = self.Nx // 2
        self.Ly = len(ALs[0])
        self.Ny = 2 * self.Ly - 1
        self.N = 2 * self.Lx * self.Ly
        self.ALs = ALs
        self.ARs = ARs
        self.CDs = CDs
        self.CCs = CCs
        self.CUs = CUs
        # excitations AL-VL-X-AR
        self.VLs = get_VLs(self.ALs)
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(self.ALs, self.CDs, self.CCs, self.CUs)
        self.vecX = vecX[:self.shape_vecX]
        self.Xs = vec_to_tensors(self.vecX, self.shape_Xs)
        Bs = Xs_to_Bs(self.Xs, self.VLs)
        # excitations AL-AL-AL-X_column
        self.VDs = get_VDs(self.CDs[-1])
        self.shape_Xs_column, self.shape_vecX_column = get_shape_Xs_vecX_column(self.CDs[-1])
        self.vecX_column = vecX[self.shape_vecX:]
        self.Xs_column = vec_to_tensors_column(self.vecX_column, self.shape_Xs_column)
        Bs_column = Xs_column_to_Bs_column(self.Xs_column, self.VDs)
        # combine excitations
        Bs_double = Bs_column_to_Bs(Bs_column, self.ALs[-1], self.CDs[-1], self.CUs[-1])
        for y in range(self.Ly):
            if Bs_double[y] is not None:
                if Bs[-1][y] is not None:
                    Bs[-1][y] += Bs_double[y]
                else:
                    Bs[-1][y] = Bs_double[y]
        self.ADs, self.AUs = get_ADs_AUs(self.ALs, self.CDs, self.CUs)
        self.Bs_sum = Bs_to_Bs_sum(Bs, self.ADs, self.AUs)
        # boundary compression parameters
        self.bc = bc
        self.chi_max_b = chi_max_b
        self.eps_b = eps_b

    @classmethod
    def from_ExcitedIsometricPEPSOverlap(cls, e_iso_peps_overlap, bc, chi_max_b, eps_b=1.e-15):
        D_max, chi_max_c = e_iso_peps_overlap.D_max, e_iso_peps_overlap.chi_max_c
        ALs, ARs = e_iso_peps_overlap.ALs, e_iso_peps_overlap.ARs
        CDs, CCs, CUs = e_iso_peps_overlap.CDs, e_iso_peps_overlap.CCs, e_iso_peps_overlap.CUs
        shape_Xs, shape_vecX = get_shape_Xs_vecX(ALs, CDs, CCs, CUs)
        shape_Xs_column, shape_vecX_column = get_shape_Xs_vecX_column(CDs[-1])
        Xs, Xs_column = e_iso_peps_overlap.extract_Xs_form2()
        vecX, vecX_column = tensors_to_vec(Xs, shape_vecX), tensors_to_vec_column(Xs_column, shape_vecX_column)
        return cls(D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, np.hstack([vecX, vecX_column]), \
                   bc, chi_max_b, eps_b)

    def print_all_excitation_norms(self):
        print("excitations AL-VL-X-AR:")
        for nx in range(self.Nx):
            for y in range(self.Ly):
                if self.Xs[nx][y] is not None:
                    X = self.Xs[nx][y].copy()
                    print(f"> {np.shape(X)} excitation parameters at site {nx,y} " \
                          + f"with ||X_{nx,y}||^2 = {np.linalg.norm(X)**2}.")
        print("excitations AL-AL-AL-X_column:")
        for ny in range(self.Ny):
            if self.Xs_column[ny] is not None:
                X_column = self.Xs_column[ny].copy()
                print(f"> {np.shape(X_column)} excitation parameters on bond {self.Nx-1,ny} " \
                      + f"with ||X_column_{ny}||^2 = {np.linalg.norm(X_column)**2}.")
        X2 = np.linalg.norm(self.vecX)**2 + np.linalg.norm(self.vecX_column)**2
        print(f"-> {self.shape_vecX} + {self.shape_vecX_column} = {self.shape_vecX + self.shape_vecX_column} " \
              + f"excitation parameters with ||X||^2 + ||X_column||^2 = {X2}.")
        return

    # boundary compressions
    def perform_boundary_compression(self, Ls_list, Cs=None):
        if np.any([Ls is not None for Ls in Ls_list]):
            if self.bc == "variational":
                boundary_compression = BoundaryCompression(Ls_list, self.chi_max_b, self.eps_b)
                boundary_compression.run(N_sweeps=3)
                L = boundary_compression.psi
                #print(f"(maximal truncation error: {np.max(boundary_compression.trunc_errors)})")
                return L
            elif self.bc == "column":
                assert Cs is not None
                boundary_compression = BoundaryColumnCompression(Ls_list, Cs, self.chi_max_b, self.eps_b)
                L = boundary_compression.run()
                return L
        else:
            return None
        
    def initialize_compressed_boundaries(self):
        self.LBBs = self.get_LBBs()
        print("Compressed LBBs.")
        self.RBkets = self.get_RBkets()
        print("Compressed RBkets.")
        self.RBbras = self.get_RBbras()
        print("Compressed RBbras.")
        self.RBBs = self.get_RBBs(self.RBkets, self.RBbras)
        print("Compressed RBBs.")
        return

    def get_LBBs(self):
        LBBs = [None] * self.Nx
        for nx in range(self.Nx-2):
            if self.Bs_sum[nx] is not None:
                # extract all needed tensors
                Bs_ket, Bs_bra = deepcopy(self.Bs_sum[nx]), deepcopy(self.Bs_sum[nx])
                LBB = LBBs[nx-1].copy() if nx > 0 and LBBs[nx-1] is not None else None
                As_ket = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx])]
                As_bra = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx])]
                # flip tensors for even nx
                if nx%2 == 0:
                    Bs_ket, Bs_bra = get_flipped_Bs_sum(Bs_ket), get_flipped_Bs_sum(Bs_bra)
                    LBB = get_flipped_mps(LBB)
                    As_ket, As_bra = get_flipped_As(As_ket), get_flipped_As(As_bra)
                # perform boundary compression
                if self.bc == "variational":
                    if Bs_ket is None and LBB is None:
                        LBB = None
                    else:
                        try:
                            vbc = VBC_BB(Bs_ket, As_ket, None, LBB, self.chi_max_b)
                            vbc.run(N_sweeps=3)
                            LBB = vbc.psi
                        except AssertionError:
                            LBB = None
                elif self.bc == "column":
                    Ls_list = [get_Ls_BB(Bs_ket, Bs_bra), \
                               get_Ls_transfer(LBB, As_ket, As_bra)]
                    Ds = []
                    for y in range(self.Ly):
                        D1 = np.shape(As_ket[y])[3]
                        Ds.append((D1, D1))
                        if y < self.Ly-1:
                            D2 = np.shape(As_ket[y])[4]
                            Ds.append((D2, D2))
                    Cs = mps_to_tensors(MPS.from_identity_product_state(Ds))
                    LBB = self.perform_boundary_compression(Ls_list, Cs)
                """
                Ls_list = [get_Ls_BB(Bs_ket, Bs_bra), \
                           get_Ls_transfer(LBB, As_ket, As_bra)]
                Ds = []
                for y in range(self.Ly):
                    D1 = np.shape(As_ket[y])[3]
                    Ds.append((D1, D1))
                    if y < self.Ly-1:
                        D2 = np.shape(As_ket[y])[4]
                        Ds.append((D2, D2))
                Cs = mps_to_tensors(MPS.from_identity_product_state(Ds))
                LBB = self.perform_boundary_compression(Ls_list, Cs)
                """
                # flip boundary mps back for even nx
                if nx%2 == 0:
                    LBB = get_flipped_mps(LBB)
                # save boundary mps
                LBBs[nx] = LBB
        return LBBs

    def get_RBkets(self):
        RBkets = [None] * self.Nx
        for nx in reversed(range(2, self.Nx)):
            # extract all needed tensors
            Bs_ket = [np.transpose(B, (0, 1, 2, 5, 6, 3, 4)) for B in deepcopy(self.Bs_sum[nx])] if self.Bs_sum[nx] is not None else None
            As_ket = [np.transpose(AL, (0, 3, 4, 1, 2)) for AL in deepcopy(self.ALs[nx])]
            As_bra = deepcopy(self.ARs[nx])
            RBket = RBkets[nx+1].copy() if nx < self.Nx-1 and RBkets[nx+1] is not None else None
            if nx%2 == 1:
                Cs = [np.conj(CD) for CD in deepcopy(self.CDs[nx-1])]
            elif nx%2 == 0:
                Cs = [np.conj(CU) for CU in deepcopy(self.CUs[nx-1])]
            # flip tensors for even nx
            if nx%2 == 0:
                Bs_ket, As_bra = get_flipped_Bs_sum(Bs_ket), get_flipped_As(As_bra)
                As_ket = get_flipped_As(As_ket)
                RBket = get_flipped_mps(RBket)
                Cs = get_flipped_Cs(Cs)
            # perform boundary compression
            if self.bc == "variational":
                if Bs_ket is None and RBket is None:
                    RBket = None
                else:
                    try:
                        vbc = VBC_B(Bs_ket, As_bra, As_ket, RBket, self.chi_max_b)
                        vbc.run(N_sweeps=3)
                        RBket = vbc.psi
                    except AssertionError:
                        RBket = None
            elif self.bc == "column":
                Rs_list = [get_Ls_B(Bs_ket, As_bra), \
                           get_Ls_transfer(RBket, As_ket, As_bra)]
                RBket = self.perform_boundary_compression(Rs_list, Cs)
            """
            Rs_list = [get_Ls_B(Bs_ket, As_bra), \
                       get_Ls_transfer(RBket, As_ket, As_bra)]
            RBket = self.perform_boundary_compression(Rs_list, Cs)
            """
            # flip boundary mps back for even nx
            if nx%2 == 0:
                RBket = get_flipped_mps(RBket)
            # save boundary mps
            RBkets[nx] = RBket
        return RBkets

    def get_RBbras(self):
        RBbras = [None] * self.Nx
        for nx in range(self.Nx):
            RBket = self.RBkets[nx]
            if RBket is not None:
                RBbra = RBket.copy()
                RBbra.Ms = [np.transpose(np.conj(M), (0, 2, 1, 3)) for M in RBbra.Ms]
                RBbras[nx] = RBbra
        """
        for nx in reversed(range(2, self.Nx)):
            # extract all needed tensors
            As_ket = deepcopy(self.ARs[nx])
            Bs_bra = [np.transpose(B, (0, 1, 2, 5, 6, 3, 4)) for B in deepcopy(self.Bs_sum[nx])] if self.Bs_sum[nx] is not None else None
            As_bra = [np.transpose(AL, (0, 3, 4, 1, 2)) for AL in deepcopy(self.ALs[nx])]
            RBbra = RBbras[nx+1].copy() if nx < self.Nx-1 and RBbras[nx+1] is not None else None
            if nx%2 == 1:
                Cs = [np.transpose(CD, (0, 2, 1, 3)) for CD in deepcopy(self.CDs[nx-1])]
            elif nx%2 == 0:
                Cs = [np.transpose(CU, (0, 2, 1, 3)) for CU in deepcopy(self.CUs[nx-1])]
            # flip tensors for even nx
            if nx%2 == 0:
                As_ket, Bs_bra = get_flipped_As(As_ket), get_flipped_Bs_sum(Bs_bra)
                As_bra = get_flipped_As(As_bra)
                RBbra = get_flipped_mps(RBbra)
                Cs = get_flipped_Cs(Cs)
            # perform boundary compression
            Rs_list = [get_Ls_Bbra(As_ket, Bs_bra), \
                       get_Ls_transfer(RBbra, As_ket, As_bra)]
            RBbra = self.perform_boundary_compression(Rs_list, Cs)
            # flip boundary mps back for even nx
            if nx%2 == 0:
                RBbra = get_flipped_mps(RBbra)
            # save boundary mps
            RBbras[nx] = RBbra
            """
        return RBbras

    def get_RBBs(self, RBkets, RBbras):
        RBBs = [None] * self.Nx
        for nx in reversed(range(2, self.Nx)):
            # extract all needed tensors
            Bs_ket = [np.transpose(B, (0, 1, 2, 5, 6, 3, 4)) for B in deepcopy(self.Bs_sum[nx])] if self.Bs_sum[nx] is not None else None
            Bs_bra = [np.transpose(B, (0, 1, 2, 5, 6, 3, 4)) for B in deepcopy(self.Bs_sum[nx])] if self.Bs_sum[nx] is not None else None
            As_ket = [np.transpose(AL, (0, 3, 4, 1, 2)) for AL in deepcopy(self.ALs[nx])]
            As_bra = [np.transpose(AL, (0, 3, 4, 1, 2)) for AL in deepcopy(self.ALs[nx])]
            RBket = RBkets[nx+1].copy() if nx < self.Nx-1 and RBkets[nx+1] is not None else None
            RBbra = RBbras[nx+1].copy() if nx < self.Nx-1 and RBbras[nx+1] is not None else None
            RBB = RBBs[nx+1].copy() if nx < self.Nx-1 and RBBs[nx+1] is not None else None
            # flip tensors for even nx
            if nx%2 == 0:
                Bs_ket, Bs_bra = get_flipped_Bs_sum(Bs_ket), get_flipped_Bs_sum(Bs_bra)
                As_ket, As_bra = get_flipped_As(As_ket), get_flipped_As(As_bra)
                RBket = get_flipped_mps(RBket)
                RBbra = get_flipped_mps(RBbra)
                RBB = get_flipped_mps(RBB)
            # perform boundary compression
            if self.bc == "variational":
                if Bs_ket is None and RBB is None:
                    RBB = None
                else:
                    try:
                        vbc = VBC_BB(Bs_ket, As_ket, RBbra, RBB, self.chi_max_b)
                        vbc.run(N_sweeps=3)
                        RBB = vbc.psi
                    except AssertionError:
                        RBB = None
            elif self.bc == "column":
                Rs_list = [get_Ls_BB(Bs_ket, Bs_bra), \
                           get_Ls_LhB(RBbra, Bs_ket, As_bra), \
                           get_Ls_LBketBbra(RBket, As_ket, Bs_bra), \
                           get_Ls_transfer(RBB, As_ket, As_bra)]
                Ds = []
                for y in range(self.Ly):
                    D1 = np.shape(As_ket[y])[3]
                    Ds.append((D1, D1))
                    if y < self.Ly-1:
                        D2 = np.shape(As_ket[y])[4]
                        Ds.append((D2, D2))
                Cs = mps_to_tensors(MPS.from_identity_product_state(Ds))
                RBB = self.perform_boundary_compression(Rs_list, Cs)
                if RBB is not None:
                    print([np.shape(M) for M in RBB.Ms])
            """
            Rs_list = [get_Ls_BB(Bs_ket, Bs_bra), \
                        get_Ls_LhB(RBbra, Bs_ket, As_bra), \
                        get_Ls_LBketBbra(RBket, As_ket, Bs_bra), \
                        get_Ls_transfer(RBB, As_ket, As_bra)]
            Ds = []
            for y in range(self.Ly):
                D1 = np.shape(As_ket[y])[3]
                Ds.append((D1, D1))
                if y < self.Ly-1:
                    D2 = np.shape(As_ket[y])[4]
                    Ds.append((D2, D2))
            Cs = mps_to_tensors(MPS.from_identity_product_state(Ds))
            RBB = self.perform_boundary_compression(Rs_list, Cs)
            """
            # flip boundary mps back for even nx
            if nx%2 == 0:
                RBB = get_flipped_mps(RBB)
            # save boundary mps
            RBBs[nx] = RBB
        return RBBs
 
    # column expectation values
    def get_column_expectation_values(self, h_mpos):
        assert len(h_mpos) == 2*self.Lx-1 and len(h_mpos[0]) == 2*self.Ly
        Nx = len(h_mpos)
        es = [0.] * Nx
        for nx in range(Nx):
            e = 0.
            # e1
            if nx > 0 and self.LBBs[nx-1] is not None:
                hs = deepcopy(h_mpos[nx])
                LBB = mps_to_tensors(self.LBBs[nx-1].copy())
                A1s = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx])]
                A2s = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])] 
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    LBB = get_flipped_Cs(LBB)
                    A1s, A2s = get_flipped_As(A1s), get_flipped_As(A2s)
                LBB = [np.ones((1, 1, 1, 1))] + LBB
                e1 = np.ones((1, 1, 1, 1))
                for y in range(self.Ly):
                    e1 = oe.contract("abcd,aefg,ghij,kehbl,cmnk,nfido,plqrs,mtup,uovrs->jqtv", \
                                     e1, LBB[2*y], LBB[2*y+1], \
                                     A1s[y], hs[2*y], np.conj(A1s[y]), \
                                     A2s[y], hs[2*y+1], np.conj(A2s[y]))
                assert np.shape(e1) == (1, 1, 1, 1)
                e += np.real_if_close(e1[0, 0, 0, 0])
            # e2
            if self.Bs_sum[nx] is not None:
                hs = deepcopy(h_mpos[nx])
                B1s_ket, B1s_bra = deepcopy(self.Bs_sum[nx]), deepcopy(self.Bs_sum[nx])
                A2s_ket = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])]
                A2s_bra = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])] 
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    B1s_ket, B1s_bra = get_flipped_Bs_sum(B1s_ket), get_flipped_Bs_sum(B1s_bra)
                    A2s_ket, A2s_bra = get_flipped_As(A2s_ket), get_flipped_As(A2s_bra)
                e2 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e2 = oe.contract("abcde,afghicj,dklg,bmlhien,ojpqr,ksto,tnuqr->fmpsu", \
                                     e2, B1s_ket[y], hs[2*y], np.conj(B1s_bra[y]), \
                                     A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
                assert np.shape(e2) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e2[0, 0, 0, 0, 0])
            # e3
            if self.Bs_sum[nx] is not None:
                hs = deepcopy(h_mpos[nx])
                B1s_ket, A1s_bra = deepcopy(self.Bs_sum[nx]), deepcopy(self.ALs[nx])
                A2s_ket, B2s_bra = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])], deepcopy(self.Bs_sum[nx+1])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    B1s_ket, A1s_bra = get_flipped_Bs_sum(B1s_ket), get_flipped_As(A1s_bra)
                    A2s_ket, B2s_bra = get_flipped_As(A2s_ket), get_flipped_Bs_sum(B2s_bra)
                e3 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e3 = oe.contract("abcde,afghicj,dklg,lhiem,njopq,krsn,btsmupq->ftoru", \
                                     e3, B1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                                     A2s_ket[y], hs[2*y+1], np.conj(B2s_bra[y]))
                assert np.shape(e3) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e3[0, 0, 0, 0, 0])
            # e4
            if self.Bs_sum[nx] is not None and nx < Nx-1 and self.RBbras[nx+2] is not None:
                hs = deepcopy(h_mpos[nx])
                B1s_ket, A1s_bra = deepcopy(self.Bs_sum[nx]), deepcopy(self.ALs[nx])
                A2s_ket, A2s_bra = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])], deepcopy(self.ALs[nx+1])
                RBbra = mps_to_tensors(self.RBbras[nx+2])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    B1s_ket, A1s_bra = get_flipped_Bs_sum(B1s_ket), get_flipped_As(A1s_bra)
                    A2s_ket, A2s_bra = get_flipped_As(A2s_ket), get_flipped_As(A2s_bra)
                    RBbra = get_flipped_Cs(RBbra)
                RBbra = RBbra + [np.ones((1, 1, 1, 1))]
                e4 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e4 = oe.contract("abcde,afghicj,dklg,lhiem,njopq,krsn,smtuv,bpuw,wqvx->fxort", \
                                     e4, B1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                                     A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]), \
                                     RBbra[2*y], RBbra[2*y+1])
                assert np.shape(e4) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e4[0, 0, 0, 0, 0])
            # e5
            if self.Bs_sum[nx] is not None:
                hs = deepcopy(h_mpos[nx])
                A1s_ket, B1s_bra = deepcopy(self.ALs[nx]), deepcopy(self.Bs_sum[nx])
                B2s_ket, A2s_bra = deepcopy(self.Bs_sum[nx+1]), [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])] 
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    A1s_ket, B1s_bra = get_flipped_As(A1s_ket), get_flipped_Bs_sum(B1s_bra)
                    B2s_ket, A2s_bra = get_flipped_Bs_sum(B2s_ket), get_flipped_As(A2s_bra)
                e5 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e5 = oe.contract("abcde,fghci,djkf,alkghem,bnoipqr,jsto,tmuqr->lnpsu", \
                                     e5, A1s_ket[y], hs[2*y], np.conj(B1s_bra[y]), \
                                     B2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
                assert np.shape(e5) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e5[0, 0, 0, 0, 0])
            # e6
            if self.Bs_sum[nx+1] is not None:
                hs = deepcopy(h_mpos[nx])
                A1s_ket, A1s_bra = deepcopy(self.ALs[nx]), deepcopy(self.ALs[nx])
                B2s_ket, B2s_bra = deepcopy(self.Bs_sum[nx+1]), deepcopy(self.Bs_sum[nx+1])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    A1s_ket, A1s_bra = get_flipped_As(A1s_ket), get_flipped_As(A1s_bra)
                    B2s_ket, B2s_bra = get_flipped_Bs_sum(B2s_ket), get_flipped_Bs_sum(B2s_bra)
                e6 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e6 = oe.contract("abcde,fghci,djkf,kghel,amniopq,jrsn,btslupq->mtoru", \
                                     e6, A1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                                     B2s_ket[y], hs[2*y+1], np.conj(B2s_bra[y]))
                assert np.shape(e6) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e6[0, 0, 0, 0, 0])
            # e7
            if self.Bs_sum[nx+1] is not None and nx < Nx-1 and self.RBbras[nx+2] is not None:
                hs = deepcopy(h_mpos[nx])
                A1s_ket, A1s_bra = deepcopy(self.ALs[nx]), deepcopy(self.ALs[nx])
                B2s_ket, A2s_bra = self.Bs_sum[nx+1], deepcopy(self.ALs[nx+1])
                RBbra = mps_to_tensors(self.RBbras[nx+2])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    A1s_ket, A1s_bra = get_flipped_As(A1s_ket), get_flipped_As(A1s_bra)
                    B2s_ket, A2s_bra = get_flipped_Bs_sum(B2s_ket), get_flipped_As(A2s_bra)
                    RBbra = get_flipped_Cs(RBbra)
                RBbra = RBbra + [np.ones((1, 1, 1, 1))]
                e7 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e7 = oe.contract("abcde,fghci,djkf,kghel,amniopq,jrsn,sltuv,bpuw,wqvx->mxort", \
                                     e7, A1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                                     B2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]), \
                                     RBbra[2*y], RBbra[2*y+1])
                assert np.shape(e7) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e7[0, 0, 0, 0, 0])
            # e8
            if self.Bs_sum[nx] is not None and nx < Nx-1 and self.RBkets[nx+2] is not None:
                hs = deepcopy(h_mpos[nx])
                A1s_ket, B1s_bra = deepcopy(self.ALs[nx]), deepcopy(self.Bs_sum[nx])
                A2s_ket, A2s_bra = deepcopy(self.ALs[nx+1]), [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(self.ARs[nx+1])] 
                RBket = mps_to_tensors(self.RBkets[nx+2])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    A1s_ket, B1s_bra = get_flipped_As(A1s_ket), get_flipped_Bs_sum(B1s_bra)
                    A2s_ket, A2s_bra = get_flipped_As(A2s_ket), get_flipped_As(A2s_bra)
                    RBket = get_flipped_Cs(RBket)
                RBket = RBket + [np.ones((1, 1, 1, 1))]
                e8 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e8 = oe.contract("abcde,fghci,djkf,alkghem,niopq,jrsn,smtuv,bpuw,wqvx->lxort", \
                                     e8, A1s_ket[y], hs[2*y], np.conj(B1s_bra[y]), \
                                     A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]), \
                                     RBket[2*y], RBket[2*y+1])
                assert np.shape(e8) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e8[0, 0, 0, 0, 0])
            # e9
            if self.Bs_sum[nx+1] is not None and nx < Nx-1 and self.RBkets[nx+2] is not None:
                hs = deepcopy(h_mpos[nx])
                A1s_ket, A1s_bra = deepcopy(self.ALs[nx]), deepcopy(self.ALs[nx])
                A2s_ket, B2s_bra = deepcopy(self.ALs[nx+1]), deepcopy(self.Bs_sum[nx+1])
                RBket = mps_to_tensors(self.RBkets[nx+2])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    A1s_ket, A1s_bra = get_flipped_As(A1s_ket), get_flipped_As(A1s_bra)
                    A2s_ket, B2s_bra = get_flipped_As(A2s_ket), get_flipped_Bs_sum(B2s_bra)
                    RBket = get_flipped_Cs(RBket)
                RBket = RBket + [np.ones((1, 1, 1, 1))]
                e9 = np.ones((1, 1, 1, 1, 1))
                for y in range(self.Ly):
                    e9 = oe.contract("abcde,fghci,djkf,kghel,minop,jqrm,asrltuv,bouw,wpvx->sxnqt", \
                                     e9, A1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                                     A2s_ket[y], hs[2*y+1], np.conj(B2s_bra[y]), \
                                     RBket[2*y], RBket[2*y+1])
                assert np.shape(e9) == (1, 1, 1, 1, 1)
                e += np.real_if_close(e9[0, 0, 0, 0, 0])
            # e10
            if nx < Nx-1 and self.RBBs[nx+2] is not None:
                hs = deepcopy(h_mpos[nx])
                A1s_ket, A1s_bra = deepcopy(self.ALs[nx]), deepcopy(self.ALs[nx])
                A2s_ket, A2s_bra = deepcopy(self.ALs[nx+1]), deepcopy(self.ALs[nx+1])
                RBB = mps_to_tensors(self.RBBs[nx+2])
                if nx%2 == 1:
                    hs = get_flipped_hs(hs)
                    A1s_ket, A1s_bra = get_flipped_As(A1s_ket), get_flipped_As(A1s_bra)
                    A2s_ket, A2s_bra = get_flipped_As(A2s_ket), get_flipped_As(A2s_bra)
                    RBB = get_flipped_Cs(RBB)
                RBB = RBB + [np.ones((1, 1, 1, 1))]
                e10 = np.ones((1, 1, 1, 1))
                for y in range(self.Ly):
                    e10 = oe.contract("abcd,efgbh,cije,jfgdk,lhmno,ipql,qkrst,ansu,uotv->vmpr", \
                                      e10, A1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                                      A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]), \
                                      RBB[2*y], RBB[2*y+1])
                assert np.shape(e10) == (1, 1, 1, 1)
                e += np.real_if_close(e10[0, 0, 0, 0])
            es[nx] = e
        return es
    
    def get_uniform_bond_energies(self, g):
        tfi_model = TFIModelFinite(2*self.Ly, g)
        es_bond = [[None] * (2*self.Ly-1) for _ in range(2*self.Lx-1)]
        for by in range(2*self.Ly-1):
            h_mpos = [tfi_model.get_bond_mpo(by, by+1) for _ in range(2*self.Lx-1)]
            es = self.get_column_expectation_values(h_mpos)
            for bx in range(2*self.Lx-1):
                es_bond[bx][by] = es[bx]
        return es_bond

    def get_bond_expectation_values(self, h_bonds):
        h_mpos_array = h_bonds_to_mpos(h_bonds, self.Lx, self.Ly)
        es_bond = [[None] * (2*self.Ly-1) for _ in range(2*self.Lx-1)]
        for by in range(2*self.Ly-1):
            h_mpos = [h_mpos_array[bx][by] for bx in range(2*self.Lx-1)]
            es = self.get_column_expectation_values(h_mpos)
            for bx in range(2*self.Lx-1):
                es_bond[bx][by] = es[bx]
        print("Computed bond expectation values.")
        return es_bond


# ground state 
def extract_all_isometric_configurations(iso_peps, min_dims=True):
    """Extract all (ALs|CDs,CCs,CUs|ARs) configurations of iso_peps by moving the orthogonality 
    column from left to right with YB-moves and the orthogonality center from down to up with 
    QR-decompositions. If min_dims=True, always take the minimum of the outer dimensions for the 
    inner dimension in the orthogonal matrix decompositions."""
    Lx = iso_peps.Lx
    Ly = iso_peps.Ly
    Nx = 2 * Lx
    Ny = 2 * Ly - 1
    ALs = [[None] * Ly for _ in range(Nx)]
    ARs = [[None] * Ly for _ in range(Nx)]
    CDs = [[None] * Ny for _ in range(Nx)]
    CCs = [[None] * Ny for _ in range(Nx)]
    CUs = [[None] * Ny for _ in range(Nx)]
    for nx in range(Nx+1):
        iso_peps_copy = DiagonalIsometricPEPS(iso_peps.Lx, iso_peps.Ly, D_max=iso_peps.D_max, \
                                              chi_factor=iso_peps.chi_factor, chi_max=iso_peps.chi_max, \
                                              d=iso_peps.d, shifting_options=iso_peps.shifting_options, \
                                              yb_options=iso_peps.yb_options, \
                                              tebd_options=iso_peps.tebd_options)
        iso_peps_copy._init_as_copy(iso_peps)
        iso_peps_copy.move_orthogonality_column_to(nx, min_dims)
        if nx < Nx:
            ARs[nx] = iso_peps_copy.get_ARs(nx)
        if nx > 0:
            ALs[nx-1] = iso_peps_copy.get_ALs(nx)
            Cs = iso_peps_copy.get_Cs(nx)
            C = MPS(Cs, norm=1.)
            Us, Vs, Ss, _ = C.get_canonical_form()
            CDs[nx-1] = Us
            CCs[nx-1] = [np.tensordot(Ss[ny], Vs[ny], axes=(1, 0)) for ny in range(Ny)]
            CUs[nx-1] = Vs
    return ALs, ARs, CDs, CCs, CUs


def get_ADs_AUs_ACs(ALs, CDs, CUs, CCs):
    assert len(ALs) == len(CDs) == len(CUs)
    assert len(ALs[0]) == (len(CDs[0])+1)//2 == (len(CUs[0])+1)//2
    Nx = len(ALs)
    Ly = len(ALs[0])
    ADs = [[None] * Ly for _ in range(Nx)]
    AUs = [[None] * Ly for _ in range(Nx)]
    ACs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            AL = ALs[nx][y].copy()
            if nx%2 == 0:
                if y == 0:
                    CD1, CU1 = np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1))
                elif y > 0:
                    CD1, CU1 = CDs[nx][2*y-1].copy(), CUs[nx][2*y-1].copy()
                CD2, CU2, CC2 = CDs[nx][2*y].copy(), CUs[nx][2*y].copy(), CCs[nx][2*y].copy()
            elif nx%2 == 1:
                CD1, CU1 = CDs[nx][2*y].copy(), CUs[nx][2*y].copy()
                if y < Ly-1:
                    CD2, CU2, CC2 = CDs[nx][2*y+1].copy(), CUs[nx][2*y+1].copy(), CCs[nx][2*y+1].copy()
                elif y == Ly-1:
                    CD2, CU2, CC2 = np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1))
            ADs[nx][y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                     AL, CD1, CD2)
            AUs[nx][y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                     AL, CU1, CU2)
            ACs[nx][y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                     AL, CD1, CC2)
    return ADs, AUs, ACs


# excitations AL-VL-X-AR

def get_VLs(ALs):
    """For left isometric tensor AL[nx][y], compute tensor VL[nx][y] such that
    .
    |\
    | \
    .  (VL)=== d*Dld*Dlu-Drd*Dru
     \/ |  Dru                   
     /\ |  /    = 0               
    .  (AL*)                     
    | /    \                    
    |/     Drd                   
    .         

    If d*Dld*Dru-Drd*Dru <= 0, set the corresponding VL to None.                         
    """
    Nx = len(ALs)
    Ly = len(ALs[0])
    VLs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            AL = ALs[nx][y].copy()
            d, Dld, Dlu, Drd, Dru = np.shape(AL)
            if (d * Dld * Dlu) - (Drd * Dru) > 0:
                AL = np.reshape(AL, (d * Dld * Dlu, Drd * Dru))
                VL = null_space(np.conj(AL).T)
                Dr = (d * Dld * Dlu) - (Drd * Dru)
                VL = np.reshape(VL, (d, Dld, Dlu, Dr))
                VLs[nx][y] = VL
    return VLs

def get_shape_Xs_vecX(ALs, CDs, CCs, CUs):
    """For left isometric tensor AL[nx][y] and orthogonality column tensors CC[nx][2*y] and
    CD[nx][2*y-1]/CU[nx][2*y+1], compute the following shapes of perturbation tensors X[nx][y]:

    for even nx:
    
    Dlu     chi_u Druu      Dlu                         .
      \         |/            \                         |\
       \       (CC)            \                        | \
        \  d   /|               \  d  chi_u Druu        .  (VL)=== d*Dld*Dlu-Drd*Dru
         \ |  / |                \ |      |/             \/ |  Dru
          (AL)  |       ->        (VL)===(X)      with   /\ |  /    = 0
         /    \ |                /        |\            .  (AL*)
        /      \|               /     chi_d Drdd        | /    \
       /       (CD)            /                        |/     Drd
      /         |\            /                         .
    Dld     chi_d Drdd      Dld     

    for odd nx:
    
    Dlu     chi_u Druu 
      \         |/              ^
       \       (CU)             | 
        \  d   /|               |
         \ |  / |               |
          (AL)  |       ________|
         /    \ |            
        /      \|          
       /       (CC)        
      /         |\        
    Dld     chi_d Drdd    

    If d*Dld*Dlu-Drd*Dru <= 0, set the corresponding shapes to None. Also return the length of the 
    vector vecX containing all X[nx][y]s that are not None.
    """
    Nx = len(ALs)
    Ly = len(ALs[0])
    shape_Xs = [[None] * Ly for _ in range(Nx)]
    shape_vecX = 0
    for nx in range(Nx):
        for y in range(Ly):
            d, Dld, Dlu, Drd, Dru = np.shape(ALs[nx][y])
            if (d * Dld * Dlu) - (Drd * Dru) > 0:
                Dr = (d * Dld * Dlu) - (Drd * Dru)
                if nx%2 == 0:
                    _, _, Druu, chi_u = np.shape(CCs[nx][2*y])
                    if y == 0:
                        chi_d, Drdd = 1, 1
                    elif y > 0:
                        chi_d, _, Drdd, _ = np.shape(CDs[nx][2*y-1])
                elif nx%2 == 1:
                    chi_d, _, Drdd, _ = np.shape(CCs[nx][2*y])
                    if y < Ly-1:
                        _, _, Druu, chi_u = np.shape(CUs[nx][2*y+1])
                    elif y == Ly-1:
                        Druu, chi_u = 1, 1
                shape_Xs[nx][y] = (chi_d, chi_u, Dr, Drdd, Druu)
                shape_vecX += chi_d * chi_u * Dr * Drdd * Druu
    return shape_Xs, shape_vecX

def vec_to_tensors(vecX, shape_Xs):
    """Reshape a vector vecX into tensors of shapes shape_Xs."""
    Nx = len(shape_Xs)
    Ly = len(shape_Xs[0])
    Xs = [[None] * Ly for _ in range(Nx)]
    vec_ind = 0
    for nx in range(Nx):
        for y in range(Ly):
            shape_X = shape_Xs[nx][y]
            if shape_X is not None:
                X = vecX[vec_ind : vec_ind + np.prod(shape_X)]
                X = np.reshape(X, shape_X)
                Xs[nx][y] = X
                vec_ind += np.prod(shape_X)
    assert vec_ind == len(vecX)
    return Xs

def tensors_to_vec(Xs, shape_vecX):
    """Reshape all tensors in Xs into one vector of length shape_vecX."""
    Nx = len(Xs)
    Ly = len(Xs[0])
    vecX = np.zeros(shape_vecX, dtype=complex)
    vec_ind = 0
    for nx in range(Nx):
        for y in range(Ly):
            if Xs[nx][y] is not None:
                X = Xs[nx][y].copy().flatten()
                vecX[vec_ind : vec_ind + np.size(X)] = X
                vec_ind += np.size(X)
    assert vec_ind == shape_vecX
    return vecX

def Xs_to_Bs(Xs, VLs):
    assert len(Xs) == len(VLs) and len(Xs[0]) == len(VLs[0])
    Nx = len(Xs)
    Ly = len(Xs[0])
    Bs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            if VLs[nx][y] is not None:
                assert Xs[nx][y] is not None
                Bs[nx][y] = oe.contract("abcd,efdgh->efabcgh", \
                                        VLs[nx][y], Xs[nx][y])
    return Bs

def Bs_to_Xs(Bs, VLs):
    assert len(Bs) == len(VLs) and len(Bs[0]) == len(VLs[0])
    Nx = len(Bs)
    Ly = len(Bs[0])
    Xs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            if Bs[nx][y] is not None:
                if VLs[nx][y] is not None:
                    X = oe.contract("abcd,efabcgh->efdgh", \
                                    np.conj(VLs[nx][y]), Bs[nx][y])
                    Xs[nx][y] = X
    return Xs


# excitations AL-AL-AL-X_column

def get_VDs(CDs):
    """For down isometric orthogonality column tensor CD[ny], compute tensor VD[ny] such that

            chi_u
              |
              |
         .--(CD*)---.
        /     |    /
       /chi   |   /     =   0   with chi = chi_d * Dl * Dr - chi_u
      /  |    .  /
     /   |   /  /
    .---(VD)---.
         | /
         |/
         .

    If chi_d * Dl * Dr - chi_u <= 0, set the corresponding VD to None.
    """
    Ny = len(CDs)
    VDs = [None] * Ny
    for ny in range(Ny):
        CD = CDs[ny].copy()
        chi_d, Dl, Dr, chi_u = np.shape(CD)
        if chi_d * Dl * Dr - chi_u > 0:
            chi = chi_d * Dl * Dr - chi_u
            CD = np.reshape(CD, (chi_d * Dl * Dr, chi_u))
            VD = null_space(np.conj(CD).T)
            VD = np.reshape(VD, (chi_d, Dl, Dr, chi))
            VDs[ny] = VD
    return VDs

def get_shape_Xs_vecX_column(CDs):
    """For down isometric orthogonality column tensor CD[ny], compute the shape 
    (chi_d * Dl * Dr - chi_u, chi_u) of the down-gauge excitation parametrization X[ny]. If 
    chi_d * Dl * Dr - chi_u <= 0, set the corresponding shape to None. Also return the length of the
    vector vecX containing all X[ny]s that are not None."""
    Ny = len(CDs)
    shape_Xs = [None] * Ny
    shape_vecX = 0
    for ny in range(Ny):
        CD = CDs[ny].copy()
        chi_d, Dl, Dr, chi_u = np.shape(CD)
        if chi_d * Dl * Dr - chi_u > 0:
            shape_Xs[ny] = (chi_d * Dl * Dr - chi_u, chi_u)
            shape_vecX += (chi_d * Dl * Dr - chi_u) * chi_u
    return shape_Xs, shape_vecX
    
def vec_to_tensors_column(vecX_column, shape_Xs_column):
    """Reshape a vector vecX_column into tensors of shapes shape_Xs_column."""
    Ny = len(shape_Xs_column)
    Xs_column = [None] * Ny
    vec_ind = 0
    for ny in range(Ny):
        shape_X_column = shape_Xs_column[ny]
        if shape_X_column is not None:
            X_column = vecX_column[vec_ind : vec_ind + np.prod(shape_X_column)]
            X_column = np.reshape(X_column, shape_X_column)
            Xs_column[ny] = X_column
            vec_ind += np.prod(shape_X_column)
    assert vec_ind == len(vecX_column)
    return Xs_column

def tensors_to_vec_column(Xs_column, shape_vecX_column):
    """Reshape all tensors in Xs_column into one vector of length shape_vecX_column."""
    Ny = len(Xs_column)
    vecX_column = np.zeros(shape_vecX_column, dtype=complex)
    vec_ind = 0
    for ny in range(Ny):
        if Xs_column[ny] is not None:
            X_column = Xs_column[ny].copy().flatten()
            vecX_column[vec_ind : vec_ind + np.size(X_column)] = X_column
            vec_ind += len(X_column)
    assert vec_ind == shape_vecX_column
    return vecX_column

def Xs_column_to_Bs_column(Xs_column, VDs):
    """For X[ny] the down-gauge parametrization, compute the perturbation tensor B[ny] given by

                        |
                       (X)
        |               |
    ---(B)---   =   ---(VD)--- 
        |               |
    
    """
    Ny = len(Xs_column)
    Bs_column = [None] * Ny
    for ny in range(Ny):
        if VDs[ny] is not None:
            assert Xs_column[ny] is not None
            Bs_column[ny] = np.tensordot(VDs[ny], Xs_column[ny], axes=(3, 0))
    return Bs_column

def Bs_column_to_Xs_column(Bs_column, VDs):
    """For B[ny] the perturbation tensor, compute the down-gauge parametrization given by

              |
              |
         .--(VD*)---.
        /     |    /         |
       /      |   /     =   (X) 
      /  |    .  /           |
     /   |   /  /
    .---(B)----.
         | /
         |/
         .
    """
    Ny = len(Bs_column)
    Xs_column = [None] * Ny
    for ny in range(Ny):
        if Bs_column[ny] is not None:
            Xs_column[ny] = np.tensordot(np.conj(VDs[ny]), Bs_column[ny], axes=((0, 1, 2), (0, 1, 2)))
    return Xs_column


# bring excitations AL-AL-AL-X_column into form AL-AL-B

def Bs_column_to_Bs(Bs_column, ALs, CDs, CUs):
    assert len(Bs_column) == len(CDs) == len(CUs) == 2*len(ALs)-1
    Ly = len(ALs)
    Bs = [None] * Ly
    for y in range(Ly-1):
        if Bs_column[2*y] is not None:
            Bs[y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                ALs[y], Bs_column[2*y], CUs[2*y+1])
        if Bs_column[2*y+1] is not None:
            if Bs[y] is not None:
                Bs[y] += oe.contract("abcde,fdgh,heij->fjabcgi", \
                                     ALs[y], CDs[2*y], Bs_column[2*y+1])
            else:
                Bs[y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                    ALs[y], CDs[2*y], Bs_column[2*y+1])
    if Bs_column[-1] is not None:
        Bs[-1] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                             ALs[-1], Bs_column[-1], np.ones((1, 1, 1, 1)))
    return Bs

def Bs_to_Xs_column(Bs, ALs, CDs, CUs, VDs):
    assert len(Bs) == len(ALs) == (len(CDs)+1)//2 == (len(CUs)+1)//2 == (len(VDs)+1)//2
    Ly = len(Bs)
    Xs_column = [None] * (2*Ly-1)
    for y in range(Ly-1):
        if Bs[y] is not None:
            B_double = oe.contract("abcdefg,cdehi->abhifg", \
                                   Bs[y], np.conj(ALs[y]))
            if VDs[2*y] is not None:
                X1 = oe.contract("abcdef,gdfb,aceh->hg", \
                                 B_double, np.conj(CUs[2*y+1]), np.conj(VDs[2*y]))
                Xs_column[2*y] = X1
            if VDs[2*y+1] is not None:
                X2 = oe.contract("abcdef,aceg,gdfh->hb", \
                                    B_double, np.conj(CDs[2*y]), np.conj(VDs[2*y+1]))
                Xs_column[2*y+1] = X2
    if Bs[-1] is not None:
        B = oe.contract("abcdefg,cdehi->abhifg", \
                        Bs[-1], np.conj(ALs[-1]))
        assert np.shape(B)[1] == np.shape(B)[3] == np.shape(B)[5] == 1
        if VDs[-1] is not None:
            Xs_column[-1] = oe.contract("abcd,acde->eb", \
                                        B[:, :, :, 0, :, 0], np.conj(VDs[-1]))
    return Xs_column


# summarize the sum of multiple excitations on one column into "mpo" with doubled bond dimensions

def Bs_to_Bs_sum(Bs, ADs, AUs):
    assert len(Bs) == len(ADs) == len(AUs)
    assert len(Bs[0]) == len(ADs[0]) == len(AUs[0])
    Nx = len(Bs)
    Ly = len(Bs[0])
    Bs_sum  = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        # if the whole column does not contain any B, return None
        if np.all([B is None for B in Bs[nx]]):
            Bs_sum[nx] = None
        else:
            # bottom site
            AD, B = ADs[nx][0].copy(), Bs[nx][0].copy() if Bs[nx][0] is not None else None
            chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(AD)
            assert chi_d == 1
            B_sum = np.zeros(shape=(1, 2*chi_u, d, Dld, Dlu, Drd, Dru), dtype=AD.dtype)
            B_sum[:, :chi_u, :, :, :, :, :] = AD
            if B is not None:
                assert np.shape(B) == np.shape(AD)
                B_sum[:, chi_u:, :, :, :, :, :] = B
            Bs_sum[nx][0] = B_sum
            # middle sites
            for y in range(1, Ly-1):
                AD, AU, B = ADs[nx][y].copy(), AUs[nx][y].copy(), Bs[nx][y].copy() if Bs[nx][y] is not None else None
                assert np.shape(AD) == np.shape(AU)
                chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(AD)
                B_sum = np.zeros(shape=(2*chi_d, 2*chi_u, d, Dld, Dlu, Drd, Dru), dtype=AD.dtype)
                B_sum[:chi_d, :chi_u, :, :, :, :, :] = AD
                B_sum[chi_d:, chi_u:, :, :, :, :, :] = AU
                if B is not None:
                    assert np.shape(B) == np.shape(AD)
                    B_sum[:chi_d, chi_u:, :, :, :, :, :] = B
                Bs_sum[nx][y] = B_sum
            # top site
            AU, B = AUs[nx][Ly-1].copy(), Bs[nx][Ly-1].copy() if Bs[nx][Ly-1] is not None else None
            chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(AU)
            assert chi_u == 1
            B_sum = np.zeros(shape=(2*chi_d, 1, d, Dld, Dlu, Drd, Dru), dtype=AU.dtype)
            B_sum[chi_d:, :, :, :, :, :, :] = AU
            if B is not None:
                assert np.shape(B) == np.shape(AU)
                B_sum[:chi_d, :, :, :, :, :, :] = B
            Bs_sum[nx][Ly-1] = B_sum
    return Bs_sum

def get_flipped_Bs_sum(Bs_sum):
    if Bs_sum is None:
        return None
    Ly = len(Bs_sum)
    Bs_sum_flipped = [None] * Ly
    # first tensor -> last tensor
    chi_d, two_chi_u, d, Dld, Dlu, Drd, Dru = np.shape(Bs_sum[0])
    assert chi_d == 1
    chi_u = two_chi_u // 2
    AD = Bs_sum[0][:, :chi_u, :, :, :, :, :].copy()
    B = Bs_sum[0][:, chi_u:, :, :, :, :, :].copy()
    B_sum_flipped = np.zeros(shape=(two_chi_u, 1, d, Dlu, Dld, Dru, Drd), dtype=AD.dtype)
    B_sum_flipped[:chi_u, :, :, :, :, :, :] = np.transpose(B, (1, 0, 2, 4, 3, 6, 5))
    B_sum_flipped[chi_u:, :, :, :, :, :, :] = np.transpose(AD, (1, 0, 2, 4, 3, 6, 5))
    Bs_sum_flipped[Ly-1] = B_sum_flipped
    # middle tensors -> reversed middle tensors
    for y in range(1, Ly-1):
        two_chi_d, two_chi_u, d, Dld, Dlu, Drd, Dru = np.shape(Bs_sum[y])
        chi_d = two_chi_d // 2
        chi_u = two_chi_u // 2
        AD = Bs_sum[y][:chi_d, :chi_u, :, :, :, :, :].copy()
        B = Bs_sum[y][:chi_d, chi_u:, :, :, :, :, :].copy()
        AU = Bs_sum[y][chi_d:, chi_u:, :, :, :, :, :].copy()
        B_sum_flipped = np.zeros(shape=(two_chi_u, two_chi_d, d, Dlu, Dld, Dru, Drd), dtype=AD.dtype)
        B_sum_flipped[:chi_u, :chi_d, :, :, :, :, :] = np.transpose(AU, (1, 0, 2, 4, 3, 6, 5))
        B_sum_flipped[:chi_u, chi_d:, :, :, :, :, :] = np.transpose(B, (1, 0, 2, 4, 3, 6, 5))
        B_sum_flipped[chi_u:, chi_d:, :, :, :, :, :] = np.transpose(AD, (1, 0, 2, 4, 3, 6, 5))
        Bs_sum_flipped[Ly-1-y] = B_sum_flipped
    # last tensor -> first tensor
    two_chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(Bs_sum[Ly-1])
    assert chi_u == 1
    chi_d = two_chi_d // 2
    B = Bs_sum[Ly-1][:chi_d, :, :, :, :, :, :].copy()
    AU = Bs_sum[Ly-1][chi_d:, :, :, :, :, :, :].copy()
    B_sum_flipped = np.zeros(shape=(1, two_chi_d, d, Dlu, Dld, Dru, Drd), dtype=AU.dtype)
    B_sum_flipped[:, :chi_d, :, :, :, :, :] = np.transpose(AU, (1, 0, 2, 4, 3, 6, 5))
    B_sum_flipped[:, chi_d:, :, :, :, :, :] = np.transpose(B, (1, 0, 2, 4, 3, 6, 5))
    Bs_sum_flipped[0] = B_sum_flipped
    return Bs_sum_flipped

def get_flipped_Bs(Bs):
    if Bs is None:
        return None
    Bs_flipped = []
    for B in Bs[::-1]:
        if B is not None:
            Bs_flipped.append(np.transpose(B, (1, 0, 2, 4, 3, 6, 5)))
        else:
            Bs_flipped.append(None)
    return Bs_flipped
    

# boundary compressions

class BoundaryCompression(TwoSiteSweep):
    def __init__(self, Ls_list, chi_max, eps):
        # boundary tensors
        Ls_list = [Ls for Ls in Ls_list if Ls is not None]
        assert Ls_list
        self.N_terms = len(Ls_list)
        assert np.all([len(Ls) == len(Ls_list[0]) for Ls in Ls_list[1:]])
        self.Ls_list = Ls_list
        # random initial MPS
        Ds = [(np.shape(L)[1], np.shape(L)[2]) for L in self.Ls_list[0]]
        mps_guess = MPS.from_random_up_isometries(Ds, chi_max, norm=1.)
        super().__init__(mps_guess, chi_max, eps)
        # environments
        self.DPs_list = [[None] * self.N_centers for _ in range(self.N_terms)]
        for i in range(self.N_terms):
            self.DPs_list[i][0] = np.ones((1, 1))
        self.UPs_list = [[None] * self.N_centers for _ in range(self.N_terms)]
        for i in range(self.N_terms):
            self.UPs_list[i][-1] = np.ones((1, 1))
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")

    def get_theta_updated(self, n, theta_guess):
        theta_updated_list = []
        for i in range(self.N_terms):
            theta_updated = oe.contract("ab,acde,efgh,hi->bcdfgi", \
                                        self.DPs_list[i][n], self.Ls_list[i][n], self.Ls_list[i][n+1], self.UPs_list[i][n])
            theta_updated_list.append(theta_updated)
        return sum(theta_updated_list)
        
    def update_Env(self, n, sweep_dir):
        if sweep_dir == "forth":
            U_updated = self.Us[n]
            for i in range(self.N_terms):
                self.DPs_list[i][n+1] = oe.contract("ab,acde,bcdf->ef", \
                                                    self.DPs_list[i][n], self.Ls_list[i][n], np.conj(U_updated))
        if sweep_dir == "back":
            V_updated = self.psi.Ms[n+1]
            for i in range(self.N_terms):
                self.UPs_list[i][n-1] = oe.contract("ab,cdea,fdeb->cf", \
                                                    self.UPs_list[i][n], self.Ls_list[i][n+1], np.conj(V_updated))


class BoundaryColumnCompression:
    def __init__(self, Ls_list, Cs, chi_max, eps):
        # boundary tensors
        Ls_list = [Ls for Ls in Ls_list if Ls is not None]
        assert Ls_list
        self.N_terms = len(Ls_list)
        assert np.all([len(Ls) == len(Ls_list[0]) for Ls in Ls_list[1:]])
        self.N = len(Ls_list[0])
        self.Ls_list = Ls_list
        # column tensors
        assert MPS(Cs, norm=1.).is_down_isometries()
        # boundary_column tensors
        self.LCs_list = [get_LCs(Ls, Cs) for Ls in Ls_list]
        # compressed boundary tensors
        self.Ls = [None] * self.N
        self.norm = None
        # truncation parameters
        self.trunc_errors = [None] * (self.N-1)
        self.chi_max = chi_max
        self.eps = eps

    def run(self):
        for n in range(self.N):
            self.truncate_L(n)
        """
        print(f"BoundaryColumnCompression truncated (B|C) to chi_max_b = {self.chi_max} for B " \
              + f"(maximal truncation error: {np.max(self.trunc_errors)}).")
        """
        L = MPS(self.Ls, self.norm) 
        return L    

    def truncate_L(self, n):
        Ls = [self.Ls_list[i][n] for i in range(self.N_terms)]
        assert np.all([np.shape(L)[:3] == np.shape(Ls[0])[:3] for L in Ls[1:]])
        chi_d, Db, Dt, _ = np.shape(Ls[0])
        Ls_matrices = [np.reshape(L, (chi_d * Db * Dt, np.shape(L)[-1])) for L in Ls]
        Q, Rs = qr_positive_stacked(Ls_matrices)
        if n < self.N-1:
            LCs = [self.LCs_list[i][n+1] for i in range(self.N_terms)]
            LCs = [np.tensordot(R, LC, axes=(1, 0)) for R, LC in zip(Rs, LCs)]
            LC = sum(LCs)
            U, _, _, trunc_error = svd_truncation(LC, self.chi_max, self.eps)
            self.trunc_errors[n] = trunc_error
            L_matrix = Q @ U
            L = np.reshape(L_matrix, (chi_d, Db, Dt, np.shape(U)[1]))
            self.Ls[n] = L
            for i in range(self.N_terms):
                self.Ls_list[i][n+1] = oe.contract("ab,bc,cdef->adef", \
                                                   np.conj(U).T, Rs[i], self.Ls_list[i][n+1])
        elif n == self.N-1:
            assert np.all([np.shape(R)[1] == 1 for R in Rs])
            U, R_sum = qr_positive(sum(Rs))
            L_matrix = Q @ U
            L = np.reshape(L_matrix, (chi_d, Db, Dt, 1))
            self.Ls[n] = L
            self.norm = np.real_if_close(R_sum[0, 0])

def Ls_double_to_Ls(Ls_double):
    L = len(Ls_double)
    N = 2 * L - 1
    Ls = [None] * N
    for y in range(L-1):
        chi_d, Ddb, Ddt, Dub, Dut, chi_u = np.shape(Ls_double[y])
        chi = min(chi_d * Ddb * Ddt, Dub * Dut * chi_u)
        L_double = np.reshape(Ls_double[y], (chi_d * Ddb * Ddt, Dub * Dut * chi_u))
        Ld, Lu = qr_positive(L_double)
        Ld = np.reshape(Ld, (chi_d, Ddb, Ddt, chi))
        Lu = np.reshape(Lu, (chi, Dub, Dut, chi_u))
        Ls[2*y] = Ld
        Ls[2*y+1] = Lu
    chi_d, Db, Dt, _, _, _ = np.shape(Ls_double[-1])
    Ls[-1] = np.reshape(Ls_double[-1], (chi_d, Db, Dt, 1))
    assert np.shape(Ls[0])[0] == np.shape(Ls[-1])[3] == 1
    return Ls

def get_Ls_h(hs, A1s_ket, A2s_ket, A1s_bra, A2s_bra):
    assert len(A1s_ket) == len(A2s_ket) == len(A1s_bra) == len(A2s_bra) == len(hs)//2
    L = len(A1s_ket) 
    Ls_double = [None] * L
    for y in range(L):
        L_double = oe.contract("abcde,fgha,hbcij,kelmn,gopk,pjqrs->dfimrnsloq", \
                               A1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                               A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
        shape = np.shape(L_double)
        L_double = np.reshape(L_double, (np.prod(shape[:3]), \
                                         shape[3], shape[4], shape[5], shape[6], \
                                         np.prod(shape[7:10])))
        Ls_double[y] = L_double
    return Ls_double_to_Ls(Ls_double)

def get_Ls_Bh(hs, B1s_ket, A2s_ket, A1s_bra, A2s_bra):
    if B1s_ket is None:
        return None
    assert len(B1s_ket) == len(A2s_ket) == len(A1s_bra) == len(A2s_bra) == len(hs)//2
    L = len(B1s_ket)
    Ls_double = [None] * L
    for y in range(L):
        L_double = oe.contract("abcdefg,hijc,jdekl,mgnop,iqrm,rlstu->afhkotpubnqs", \
                               B1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                               A2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
        shape = np.shape(L_double)
        L_double = np.reshape(L_double, (np.prod(shape[:4]), \
                                         shape[4], shape[5], shape[6], shape[7], \
                                         np.prod(shape[8:])))
        Ls_double[y] = L_double
    return Ls_double_to_Ls(Ls_double)
        
def get_Ls_hB(hs, A1s_ket, B2s_ket, A1s_bra, A2s_bra):
    if B2s_ket is None:
        return None
    assert len(A1s_ket) == len(B2s_ket) == len(A1s_bra) == len(A2s_bra) == len(hs)//2
    L = len(A1s_ket)
    Ls_double = [None] * L
    for y in range(L):
        L_double = oe.contract("abcde,fgha,hbcij,klmenop,gqrm,rjstu->kdfiotpulnqs", \
                               A1s_ket[y], hs[2*y], np.conj(A1s_bra[y]), \
                               B2s_ket[y], hs[2*y+1], np.conj(A2s_bra[y]))
        shape = np.shape(L_double)
        L_double = np.reshape(L_double, (np.prod(shape[:4]), \
                                         shape[4], shape[5], shape[6], shape[7], \
                                         np.prod(shape[8:])))
        Ls_double[y] = L_double
    return Ls_double_to_Ls(Ls_double)

def get_Ls_B(Bs_ket, As_bra):
    if Bs_ket is None:
        return None
    assert len(Bs_ket) == len(As_bra)
    L = len(Bs_ket)
    Ls_double = [None] * L
    for y in range(L):
        Ls_double[y] = oe.contract("abcdefg,cdehi->afhgib", \
                                   Bs_ket[y], np.conj(As_bra[y]))
    return Ls_double_to_Ls(Ls_double)

def get_Ls_LhB(Lh, Bs_ket, As_bra):
    if Lh is None or Bs_ket is None:
        return None
    assert len(Bs_ket) == len(As_bra) == (Lh.N+1)//2
    L = len(Bs_ket)
    Lhs = mps_to_tensors(Lh)
    Ls_double = [None] * L
    for y in range(L):
        if y < L-1:
            L_double = oe.contract("abcd,defg,hijbekl,jcfmn->ahkmlngi", \
                                   Lhs[2*y], Lhs[2*y+1], Bs_ket[y], np.conj(As_bra[y]))
        elif y == L-1:
            L_double = oe.contract("abcd,efgbhij,gchkl->aeikjldf", \
                                   Lhs[-1], Bs_ket[-1], np.conj(As_bra[-1]))
        shape = np.shape(L_double)
        L_double = np.reshape(L_double, (np.prod(shape[:2]), \
                                         shape[2], shape[3], shape[4], shape[5], \
                                         np.prod(shape[6:])))
        Ls_double[y] = L_double
    return Ls_double_to_Ls(Ls_double)

def get_Ls_transfer(L, As_ket, As_bra):
    if L is None:
        return None
    assert len(As_ket) == len(As_bra) == (L.N+1)//2
    Ls = mps_to_tensors(L)
    L = len(As_ket)
    Ls_double = [None] * L
    for y in range(L-1):
        Ls_double[y] = oe.contract("abcd,defg,hbeij,hcfkl->aikjlg", \
                                   Ls[2*y], Ls[2*y+1], As_ket[y], np.conj(As_bra[y]))
    Ls_double[-1] = oe.contract("abcd,ebfgh,ecfij->agihjd", \
                                Ls[-1], As_ket[-1], np.conj(As_bra[-1]))
    return Ls_double_to_Ls(Ls_double)

# needed for local energies

def get_Ls_Bbra(As_ket, Bs_bra):
    if Bs_bra is None:
        return None
    assert len(As_ket) == len(Bs_bra)
    L = len(As_ket)
    Ls_double = [None] * L
    for y in range(L):
        Ls_double[y] = oe.contract("abcde,fgabchi->fdheig", \
                                   As_ket[y], np.conj(Bs_bra[y]))
    return Ls_double_to_Ls(Ls_double)

def get_Ls_BB(Bs_ket, Bs_bra):
    if Bs_ket is None or Bs_bra is None:
        return None
    assert len(Bs_ket) == len(Bs_bra)
    L = len(Bs_ket)
    Ls_double = [None] * L
    for y in range(L):
        L_double = oe.contract("abcdefg,hicdejk->ahfjgkbi", \
                               Bs_ket[y], np.conj(Bs_bra[y]))
        shape = np.shape(L_double)
        L_double = np.reshape(L_double, (np.prod(shape[:2]), \
                                         shape[2], shape[3], shape[4], shape[5], \
                                         np.prod(shape[6:])))
        Ls_double[y] = L_double
    return Ls_double_to_Ls(Ls_double)

def get_Ls_LBketBbra(LBket, As_ket, Bs_bra):
    if LBket is None or Bs_bra is None:
        return None
    assert len(As_ket) == len(Bs_bra) == (LBket.N+1)//2
    L = len(As_ket)
    LBkets = mps_to_tensors(LBket)
    Ls_double = [None] * L
    for y in range(L):
        if y < L-1:
            L_double = oe.contract("abcd,defg,hbeij,klhcfmn->akimjngl", \
                                   LBkets[2*y], LBkets[2*y+1], As_ket[y], np.conj(Bs_bra[y]))
        elif y == L-1:
            L_double = oe.contract("abcd,ebfgh,ijecfkl->aigkhldj", \
                                   LBkets[-1], As_ket[-1], np.conj(Bs_bra[-1]))
        shape = np.shape(L_double)
        L_double = np.reshape(L_double, (np.prod(shape[:2]), \
                                         shape[2], shape[3], shape[4], shape[5], \
                                         np.prod(shape[6:])))
        Ls_double[y] = L_double
    return Ls_double_to_Ls(Ls_double)


def qr_positive_stacked(Ms):
    assert np.all([np.shape(M)[0] == np.shape(Ms[0])[0] for M in Ms[1:]]), \
           "All matrices must have the same number of lines"
    M_stacked = np.hstack(Ms)
    Q, R_stacked = qr_positive(M_stacked)
    col_starts = np.cumsum([0] + [np.shape(M)[1] for M in Ms[:-1]])
    col_ends = col_starts + np.array([np.shape(M)[1] for M in Ms])
    Rs = [R_stacked[:, start:end] for start, end in zip(col_starts, col_ends)]
    return Q, Rs

def get_LCs(Ls, Cs):
    assert len(Ls) == len(Cs)
    N = len(Ls)
    assert np.all([np.shape(Ls[n])[1] == np.shape(Cs[n])[1] \
                   and np.shape(Ls[n])[2] == np.shape(Cs[n])[2] for n in range(N)])
    LCs = [None] * N
    LC = np.ones((1, 1))
    for n in reversed(range(N)):
        LC = oe.contract("ab,cdea,fdeb->cf", \
                         LC, Ls[n], Cs[n])
        LCs[n] = LC
    return LCs

def CCdaggers_to_down_isometric_form(Cs, side):
    if side == "right":
        Cs = [np.transpose(C.copy(), (0, 2, 1, 3)) for C in Cs]
    CCdaggers = []
    for C in Cs:
        chi_d, Dl, _, chi_u = np.shape(C)
        CCdagger = np.tensordot(C, np.conj(C), axes=(2, 2))
        CCdagger = np.transpose(CCdagger, (0, 3, 1, 4, 2, 5))
        CCdagger = np.reshape(CCdagger, (chi_d**2, Dl, Dl, chi_u**2))
        CCdaggers.append(CCdagger)
    CCdaggers_mps = MPS(CCdaggers, norm=1.)
    CCdaggers_mps.to_down_isometries()
    CCdaggers = CCdaggers_mps.Ms
    norm = CCdaggers_mps.norm
    return CCdaggers, norm

def mps_to_tensors(mps):
    Ms = [M.copy() for M in mps.Ms]
    Ms[0] *= mps.norm
    return Ms


def h_bonds_to_mpos(h_bonds, Lx, Ly):
    Nx = 2 * Lx - 1
    Ny = 2 * Ly - 1
    N_bonds = Nx * Ny
    assert len(h_bonds) == N_bonds
    lattice = DiagonalSquareLattice(Lx, Ly)
    h_mpos_array = [[None] * Ny for _ in range(Nx)]
    for n in range(N_bonds):
        bx, by = lattice.get_bond_vector(n)
        h_mpo = [None] * (Ny+1)
        h_id = np.zeros(shape=(1, 1, 2, 2))
        h_id[0, 0, :, :] = np.eye(2)
        for ny in list(range(by)) + list(range(by+2, Ny+1)):
            h_mpo[ny] = h_id.copy()
        h_bond = h_bonds[n].copy()
        h_bond = np.transpose(h_bond, (0, 2, 1, 3))
        h_bond = np.reshape(h_bond, (4, 4))
        h_d, h_u = qr_positive(h_bond)
        h_d = np.reshape(h_d, (1, 2, 2, 4))
        h_d = np.transpose(h_d, (0, 3, 1, 2))
        h_u = np.reshape(h_u, (4, 2, 2, 1))
        h_u = np.transpose(h_u, (0, 3, 1, 2))
        h_mpo[by], h_mpo[by+1] = h_d, h_u
        h_mpos_array[bx][by] = h_mpo
    return h_mpos_array


def get_Ds(As_bottom, As_top):
    assert len(As_bottom) == len(As_top)
    L = len(As_bottom)
    Ds = []
    for y in range(L):
        D1b = np.shape(As_bottom[y])[3]
        D1t = np.shape(As_top[y])[3]
        Ds.append((D1b, D1t))
        if y < L-1:
            D2b = np.shape(As_bottom[y])[4]
            D2t = np.shape(As_top[y])[4]
            Ds.append((D2b, D2t))
    return Ds


class VBC_h(TwoSiteSweep):
    def __init__(self, A1s, A2s, hs, Bh1, chi_max, eps=1.e-15):
        assert hs is not None or Bh1 is not None
        mps_guess = MPS.from_random_up_isometries(get_Ds(A2s, A2s), chi_max, norm=1.)
        super().__init__(mps_guess, chi_max, eps)
        self.A1s = A1s
        self.A2s = A2s
        self.hs = hs
        self.Bh1 = Bh1
        if hs is not None:
            self.DPs = [None] * self.N_centers
            self.UPs = [None] * self.N_centers
        if Bh1 is not None:
            self.DPTs = [None] * self.N_centers
            self.UPTs = [None] * self.N_centers
        self.init_Env()

    def get_theta_updated(self, n, theta2_guess):  
        theta2_updated = np.zeros(shape=theta2_guess.shape, dtype=np.complex128)                    
        if n%2 == 0:
            if self.hs is not None:
                theta2_updated += oe.contract("abcd,eafgh,bije,jcklm,fikn->dglhmn", \
                                              self.DPs[n], \
                                              self.A2s[n//2], self.hs[n+1], np.conj(self.A2s[n//2]), \
                                              self.UPs[n])
            if self.Bh1 is not None:
                theta2_updated += self.Bh1.norm * oe.contract("ab,acde,efgh,icfjk,idglm,hn->bjlkmn", \
                                                              self.DPTs[n], \
                                                              self.Bh1.Ms[n], self.Bh1.Ms[n+1], \
                                                              self.A2s[n//2], np.conj(self.A2s[n//2]), \
                                                              self.UPTs[n])
        elif n%2 == 1:
            if self.hs is not None:
                theta2_updated += oe.contract("abcdef,abcghi->fdeghi", \
                                              self.DPs[n], \
                                              self.UPs[n])
            if self.Bh1 is not None:
                theta2_updated += self.Bh1.norm * oe.contract("abcd,aefg->dbcefg", \
                                                              self.DPTs[n], \
                                                              self.UPTs[n])
        return theta2_updated

    def init_Env(self):                         
        # Down parts for center 0
        if self.hs is not None:
            A1 = self.A1s[0][:, 0, :, 0, :]
            h = self.hs[0][0, :, :, :]
            DP = oe.contract("abc,dea,ebg->cdg", \
                             A1, h, np.conj(A1))  
            self.DPs[0] = DP[:, :, :, np.newaxis]
        if self.Bh1 is not None:
            self.DPTs[0] = np.ones((1, 1)) 
        # Up parts for all centers
        if self.hs is not None:
            A2 = self.A2s[-1][:, :, 0, :, 0] 
            h2 = self.hs[-1][:, 0, :, :]  
            A1 = self.A1s[-1] 
            h1 = self.hs[-2] 
            UP = oe.contract("abc,dea,efg,hijkb,ldmh,mijnf->klncg", \
                             A2, h2, np.conj(A2), \
                             A1, h1, np.conj(A1))
            self.UPs[-1] = UP[:, :, :, :, :, np.newaxis] 
        if self.Bh1 is not None:
            A2 = self.A2s[-1][:, :, 0, :, 0] 
            Bh1 = self.Bh1.Ms[-1][:, :, :, 0] 
            UPT = oe.contract("abc,dbe,dcf->aef", \
                              Bh1, A2, np.conj(A2))  
            self.UPTs[-1] = UPT[:, :, :, np.newaxis] 
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return
            
    def update_Env(self, n, sweep_dir):                                                   
        if sweep_dir == "forth":
            U_updated = self.Us[n] 
            if n%2 == 0:
                if self.hs is not None:
                    DP_updated = oe.contract("abcd,eafgh,bije,jcklm,dgln->fikhmn", \
                                             self.DPs[n], \
                                             self.A2s[n//2], self.hs[n+1], np.conj(self.A2s[n//2]), \
                                             np.conj(U_updated))
                    self.DPs[n+1] = DP_updated 
                if self.Bh1 is not None:
                    DPT_updated = oe.contract("ab,acde,efgh,icfjk,idglm,bjln->hkmn", \
                                              self.DPTs[n], \
                                              self.Bh1.Ms[n], self.Bh1.Ms[n+1], \
                                              self.A2s[n//2], np.conj(self.A2s[n//2]), \
                                              np.conj(U_updated))
                    self.DPTs[n+1] = DPT_updated 
            elif n%2 == 1:
                if self.hs is not None:
                    DP_updated = oe.contract("abcdef,fdeg,hijak,blmh,mijcn->klng", \
                                             self.DPs[n], \
                                             np.conj(U_updated), \
                                             self.A1s[(n+1)//2], self.hs[n+1], np.conj(self.A1s[(n+1)//2]))
                    self.DPs[n+1] = DP_updated 
                if self.Bh1 is not None:
                    DPT_updated = oe.contract("abcd,dbce->ae", \
                                              self.DPTs[n], \
                                              np.conj(U_updated))  
                    self.DPTs[n+1] = DPT_updated 
            if n >= 2:
                if self.hs is not None:
                    self.UPs[n-2] = None
                if self.Bh1 is not None:
                    self.UPTs[n-2] = None
            return
        elif sweep_dir == "back" and n > 0:
            V_updated = self.psi.Ms[n+1] 
            if n%2 == 1:
                if self.hs is not None:
                    UP_updated = oe.contract("abcdef,gdef->abcg", \
                                             self.UPs[n], \
                                             np.conj(V_updated))
                    self.UPs[n-1] = UP_updated 
                if self.Bh1 is not None:
                    UPT_updated = oe.contract("abcd,ebcd->ae", \
                                              self.UPTs[n], \
                                              np.conj(V_updated))  
                    self.UPTs[n-1] = UPT_updated 
            elif n%2 == 0:
                if self.hs is not None:
                    UP_updated = oe.contract("abcd,efagh,ibje,jkclm,nopqf,risn,soptk,uhmd->qrtglu", \
                                             self.UPs[n], \
                                             self.A2s[n//2], self.hs[n+1], np.conj(self.A2s[n//2]), \
                                             self.A1s[n//2], self.hs[n] , np.conj(self.A1s[n//2]), \
                                             np.conj(V_updated))
                    self.UPs[n-1] = UP_updated 
                if self.Bh1 is not None:
                    UPT_updated = oe.contract("ab,cdea,fghc,igdjk,ihelm,nkmb->fjln", \
                                              self.UPTs[n], \
                                              self.Bh1.Ms[n+1], self.Bh1.Ms[n], \
                                              self.A2s[n//2], np.conj(self.A2s[n//2]), \
                                              np.conj(V_updated))
                    self.UPTs[n-1] = UPT_updated 
            if n <= self.N_centers-3:
                if self.hs is not None:
                    self.DPs[n+2] = None
                if self.Bh1 is not None:
                    self.DPTs[n+2] = None
            return

class VBC_B(TwoSiteSweep):
    def __init__(self, Bs, ARs, ALs, RB, chi_max, eps=1.e-15):
        assert Bs is not None or RB is not None
        mps_guess = MPS.from_random_up_isometries(get_Ds(ALs, ARs), chi_max, norm=1.)
        super().__init__(mps_guess, chi_max, eps)
        self.Bs = Bs
        self.ARs = ARs
        self.ALs = ALs
        self.RB = RB
        if Bs is not None:
            self.DPs = [None] * self.N_centers
            self.UPs = [None] * self.N_centers
        if RB is not None:
            self.DPTs = [None] * self.N_centers
            self.UPTs = [None] * self.N_centers
        self.init_Env()

    def get_theta_updated(self, n, theta2_guess):  
        theta2_updated = np.zeros(shape=theta2_guess.shape, dtype=np.complex128)                    
        if n%2 == 0:
            if self.Bs is not None:
                theta2_updated += oe.contract("ab,bcdefgh,defij,kc->agihjk", \
                                              self.DPs[n], \
                                              self.Bs[n//2], np.conj(self.ARs[n//2]), \
                                              self.UPs[n])
            if self.RB is not None:
                theta2_updated += self.RB.norm * oe.contract("ab,acde,efgh,icfjk,idglm,hn->bjlkmn", \
                                                             self.DPTs[n], \
                                                             self.RB.Ms[n], self.RB.Ms[n+1], \
                                                             self.ALs[n//2], np.conj(self.ARs[n//2]), \
                                                             self.UPTs[n])
        elif n%2 == 1:
            if self.Bs is not None:
                theta2_updated += oe.contract("abcd,ebfg->acdfge", \
                                              self.DPs[n], \
                                              self.UPs[n])
            if self.RB is not None:
                theta2_updated += self.RB.norm * oe.contract("abcd,aefg->dbcefg", \
                                                             self.DPTs[n], \
                                                             self.UPTs[n])
        return theta2_updated
    
    def init_Env(self):                         
        # Down parts for center 0
        if self.Bs is not None: 
            self.DPs[0] = np.ones((1, 1))
        if self.RB is not None:
            self.DPTs[0] = np.ones((1, 1)) 
        # Up parts for all centers
        if self.Bs is not None:
            AR = self.ARs[-1][:, :, 0, :, 0] 
            B = self.Bs[-1][:, :, :, :, 0, :, 0]  
            self.UPs[-1] = oe.contract("abcde,cdf->baef", \
                                       B, np.conj(AR))
        if self.RB is not None:
            AR = self.ARs[-1][:, :, 0, :, 0] 
            AL = self.ALs[-1][:, :, 0, :, 0]
            RB = self.RB.Ms[-1][:, :, :, 0] 
            UPT = oe.contract("abc,dbe,dcf->aef", \
                              RB, AL, np.conj(AR))  
            self.UPTs[-1] = UPT[:, :, :, np.newaxis] 
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return
    
    def update_Env(self, n, sweep_dir):                                                   
        if sweep_dir == "forth":
            U_updated = self.Us[n] 
            if n%2 == 0:
                if self.Bs is not None:
                    DP_updated = oe.contract("ab,bcdefgh,defij,agik->kchj", \
                                             self.DPs[n], \
                                             self.Bs[n//2], np.conj(self.ARs[n//2]), \
                                             np.conj(U_updated))
                    self.DPs[n+1] = DP_updated 
                if self.RB is not None:
                    DPT_updated = oe.contract("ab,acde,efgh,icfjk,idglm,bjln->hkmn", \
                                              self.DPTs[n], \
                                              self.RB.Ms[n], self.RB.Ms[n+1], \
                                              self.ALs[n//2], np.conj(self.ARs[n//2]), \
                                              np.conj(U_updated))
                    self.DPTs[n+1] = DPT_updated 
            elif n%2 == 1:
                if self.Bs is not None:
                    DP_updated = oe.contract("abcd,acde->eb", \
                                             self.DPs[n], \
                                             np.conj(U_updated))
                    self.DPs[n+1] = DP_updated 
                if self.RB is not None:
                    DPT_updated = oe.contract("abcd,dbce->ae", \
                                              self.DPTs[n], \
                                              np.conj(U_updated))  
                    self.DPTs[n+1] = DPT_updated 
            if n >= 2:
                if self.Bs is not None:
                    self.UPs[n-2] = None
                if self.RB is not None:
                    self.UPTs[n-2] = None
            return
        elif sweep_dir == "back" and n > 0:
            V_updated = self.psi.Ms[n+1] 
            if n%2 == 1:
                if self.Bs is not None:
                    UP_updated = oe.contract("abcd,ecda->eb", \
                                             self.UPs[n], \
                                             np.conj(V_updated))
                    self.UPs[n-1] = UP_updated 
                if self.RB is not None:
                    UPT_updated = oe.contract("abcd,ebcd->ae", \
                                              self.UPTs[n], \
                                              np.conj(V_updated))  
                    self.UPTs[n-1] = UPT_updated 
            elif n%2 == 0:
                if self.Bs is not None:
                    UP_updated = oe.contract("ab,cbdefgh,defij,khja->kcgi", \
                                             self.UPs[n], \
                                             self.Bs[n//2], np.conj(self.ARs[n//2]), \
                                             np.conj(V_updated))
                    self.UPs[n-1] = UP_updated 
                if self.RB is not None:
                    UPT_updated = oe.contract("ab,cdea,fghc,igdjk,ihelm,nkmb->fjln", \
                                              self.UPTs[n], \
                                              self.RB.Ms[n+1], self.RB.Ms[n], \
                                              self.ALs[n//2], np.conj(self.ARs[n//2]), \
                                              np.conj(V_updated))
                    self.UPTs[n-1] = UPT_updated 
            if n <= self.N_centers-3:
                if self.Bs is not None:
                    self.DPs[n+2] = None
                if self.RB is not None:
                    self.DPTs[n+2] = None
            return

class VBC_Bh(TwoSiteSweep):
    def __init__(self, hs, B1s, B2s, AL1s, AL2s, AR2s, Lh, LBh, chi_max, eps=1.e-15):
        assert B1s is not None or B2s is not None or LBh is not None
        mps_guess = MPS.from_random_up_isometries(get_Ds(AR2s, AL2s), chi_max, norm=1.)
        super().__init__(mps_guess, chi_max, eps)
        self.hs = hs
        self.B1s = B1s
        self.B2s = B2s
        self.AL1s = AL1s
        self.AL2s = AL2s
        self.AR2s = AR2s
        self.Lh = Lh
        self.LBh = LBh
        if B1s is not None:
            self.DP1s = [None] * self.N_centers
            self.UP1s = [None] * self.N_centers 
        if B2s is not None:
            self.DP2s = [None] * self.N_centers
            self.UP2s = [None] * self.N_centers 
            if Lh is not None:
                self.DP3s = [None] * self.N_centers
                self.UP3s = [None] * self.N_centers            
        if LBh is not None:
            self.DPTs = [None] * self.N_centers
            self.UPTs = [None] * self.N_centers
        self.init_Env()

    def get_theta_updated(self, n, theta2_guess):  
        theta2_updated = np.zeros(shape=theta2_guess.shape, dtype=np.complex128)                    
        if n%2 == 0:
            if self.B1s is not None:
                theta2_updated += oe.contract("oabcd,eafgh,bije,jcklm,ofikn->dglhmn", \
                                              self.DP1s[n], \
                                              self.AR2s[n//2], self.hs[n+1], np.conj(self.AL2s[n//2]), \
                                              self.UP1s[n])
            if self.B2s is not None:
                theta2_updated += oe.contract("oabcd,opeafgh,bije,jcklm,pfikn->dglhmn", \
                                              self.DP2s[n], \
                                              self.B2s[n//2], self.hs[n+1], np.conj(self.AL2s[n//2]), \
                                              self.UP2s[n])
                if self.Lh is not None:
                    theta2_updated += self.Lh.norm * oe.contract("oab,acde,efgh,opicfjk,idglm,phn->bjlkmn", \
                                                                 self.DP3s[n], \
                                                                 self.Lh.Ms[n], self.Lh.Ms[n+1], \
                                                                 self.B2s[n//2], np.conj(self.AL2s[n//2]), \
                                                                 self.UP3s[n])
            if self.LBh is not None:
                theta2_updated += self.LBh.norm * oe.contract("ab,acde,efgh,icfjk,idglm,hn->bjlkmn", \
                                                              self.DPTs[n], \
                                                              self.LBh.Ms[n], self.LBh.Ms[n+1], \
                                                              self.AR2s[n//2], np.conj(self.AL2s[n//2]), \
                                                              self.UPTs[n])
        elif n%2 == 1:
            if self.B1s is not None:
                theta2_updated += oe.contract("jabcdef,jabcghi->fdeghi", \
                                              self.DP1s[n], \
                                              self.UP1s[n])
            if self.B2s is not None:
                theta2_updated += oe.contract("jabcdef,jabcghi->fdeghi", \
                                              self.DP2s[n], \
                                              self.UP2s[n])
                if self.Lh is not None:
                    theta2_updated += self.Lh.norm * oe.contract("habcd,haefg->dbcefg", \
                                                                 self.DP3s[n], \
                                                                 self.UP3s[n])
            if self.LBh is not None:
                theta2_updated += self.LBh.norm * oe.contract("abcd,aefg->dbcefg", \
                                                              self.DPTs[n], \
                                                              self.UPTs[n])
        return theta2_updated
    
    def init_Env(self):                         
        # Down parts for center 0
        if self.B1s is not None:
            B1 = self.B1s[0][0, :, :, 0, :, 0, :]
            AL1 = self.AL1s[0][:, 0, :, 0, :]
            h = self.hs[0][0, :, :, :]
            DP = oe.contract("habc,dea,ebg->hcdg", \
                             B1, h, np.conj(AL1))  
            self.DP1s[0] = DP[:, :, :, :, np.newaxis]
        if self.B2s is not None:
            AL1 = self.AL1s[0][:, 0, :, 0, :]
            h = self.hs[0][0, :, :, :]
            DP = oe.contract("abc,dea,ebg->cdg", \
                             AL1, h, np.conj(AL1))  
            self.DP2s[0] = DP[np.newaxis, :, :, :, np.newaxis]
            if self.Lh is not None:
                self.DP3s[0] = np.ones((1, 1, 1)) 
        if self.LBh is not None:
            self.DPTs[0] = np.ones((1, 1)) 
        # Up parts for all centers
        if self.B1s is not None:
            AR2 = self.AR2s[-1][:, :, 0, :, 0] 
            AL2 = self.AL2s[-1][:, :, 0, :, 0]
            h2 = self.hs[-1][:, 0, :, :]  
            B1 = self.B1s[-1][:, 0, :, :, :, :, :]
            h1 = self.hs[-2] 
            AL1 = self.AL1s[-1]
            UP = oe.contract("abc,dea,efg,ohijkb,ldmh,mijnf->oklncg", \
                             AR2, h2, np.conj(AL2), \
                             B1, h1, np.conj(AL1))
            self.UP1s[-1] = UP[:, :, :, :, :, :, np.newaxis] 
        if self.B2s is not None:
            B2 = self.B2s[-1][:, 0, :, :, 0, :, 0] 
            AL2 = self.AL2s[-1][:, :, 0, :, 0]
            h2 = self.hs[-1][:, 0, :, :]  
            AL1 = self.AL1s[-1]
            h1 = self.hs[-2] 
            UP = oe.contract("oabc,dea,efg,hijkb,ldmh,mijnf->oklncg", \
                             B2, h2, np.conj(AL2), \
                             AL1, h1, np.conj(AL1))
            self.UP2s[-1] = UP[:, :, :, :, :, :, np.newaxis] 
            if self.Lh is not None:
                B = self.B2s[-1][:, 0, :, :, 0, :, 0] 
                AL = self.AL2s[-1][:, :, 0, :, 0]
                Lh = self.Lh.Ms[-1][:, :, :, 0] 
                UP = oe.contract("abc,gdbe,dcf->gaef", \
                                 Lh, B, np.conj(AL))  
                self.UP3s[-1] = UP[:, :, :, :, np.newaxis]
        if self.LBh is not None:
            AR = self.AR2s[-1][:, :, 0, :, 0] 
            AL = self.AL2s[-1][:, :, 0, :, 0]
            LBh = self.LBh.Ms[-1][:, :, :, 0] 
            UPT = oe.contract("abc,dbe,dcf->aef", \
                              LBh, AR, np.conj(AL))  
            self.UPTs[-1] = UPT[:, :, :, np.newaxis] 
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return
    
    def update_Env(self, n, sweep_dir):                                                   
        if sweep_dir == "forth":
            U_updated = self.Us[n] 
            if n%2 == 0:
                if self.B1s is not None:
                    DP_updated = oe.contract("oabcd,eafgh,bije,jcklm,dgln->ofikhmn", \
                                             self.DP1s[n], \
                                             self.AR2s[n//2], self.hs[n+1], np.conj(self.AL2s[n//2]), \
                                             np.conj(U_updated))
                    self.DP1s[n+1] = DP_updated 
                if self.B2s is not None:
                    DP_updated = oe.contract("oabcd,opeafgh,bije,jcklm,dgln->pfikhmn", \
                                             self.DP2s[n], \
                                             self.B2s[n//2], self.hs[n+1], np.conj(self.AL2s[n//2]), \
                                             np.conj(U_updated))
                    self.DP2s[n+1] = DP_updated 
                    if self.Lh is not None:
                        DP_updated = oe.contract("oab,acde,efgh,opicfjk,idglm,bjln->phkmn", \
                                                 self.DP3s[n], \
                                                 self.Lh.Ms[n], self.Lh.Ms[n+1], \
                                                 self.B2s[n//2], np.conj(self.AL2s[n//2]), \
                                                 np.conj(U_updated))
                        self.DP3s[n+1] = DP_updated 
                if self.LBh is not None:
                    DP_updated = oe.contract("ab,acde,efgh,icfjk,idglm,bjln->hkmn", \
                                             self.DPTs[n], \
                                             self.LBh.Ms[n], self.LBh.Ms[n+1], \
                                             self.AR2s[n//2], np.conj(self.AL2s[n//2]), \
                                             np.conj(U_updated))
                    self.DPTs[n+1] = DP_updated 
            elif n%2 == 1:
                if self.B1s is not None:
                    DP_updated = oe.contract("oabcdef,fdeg,ophijak,blmh,mijcn->pklng", \
                                             self.DP1s[n], \
                                             np.conj(U_updated), \
                                             self.B1s[(n+1)//2], self.hs[n+1], np.conj(self.AL1s[(n+1)//2]))
                    self.DP1s[n+1] = DP_updated 
                if self.B2s is not None:
                    DP_updated = oe.contract("oabcdef,fdeg,hijak,blmh,mijcn->oklng", \
                                             self.DP2s[n], \
                                             np.conj(U_updated), \
                                             self.AL1s[(n+1)//2], self.hs[n+1], np.conj(self.AL1s[(n+1)//2]))
                    self.DP2s[n+1] = DP_updated
                    if self.Lh is not None:
                        DP_updated = oe.contract("fabcd,dbce->fae", \
                                                 self.DP3s[n], \
                                                 np.conj(U_updated))  
                        self.DP3s[n+1] = DP_updated 
                if self.LBh is not None:
                    DP_updated = oe.contract("abcd,dbce->ae", \
                                             self.DPTs[n], \
                                             np.conj(U_updated))  
                    self.DPTs[n+1] = DP_updated 
            if n >= 2:
                if self.B1s is not None:
                    self.UP1s[n-2] = None
                if self.B2s is not None:
                    self.UP2s[n-2] = None
                    if self.Lh is not None:
                        self.UP3s[n-2] = None
                if self.LBh is not None:
                    self.UPTs[n-2] = None
            return
        elif sweep_dir == "back" and n > 0:
            V_updated = self.psi.Ms[n+1] 
            if n%2 == 1:
                if self.B1s is not None:
                    UP_updated = oe.contract("habcdef,gdef->habcg", \
                                             self.UP1s[n], \
                                             np.conj(V_updated))
                    self.UP1s[n-1] = UP_updated 
                if self.B2s is not None:
                    UP_updated = oe.contract("habcdef,gdef->habcg", \
                                             self.UP2s[n], \
                                             np.conj(V_updated))
                    self.UP2s[n-1] = UP_updated 
                    if self.Lh is not None:
                        UP_updated = oe.contract("fabcd,ebcd->fae", \
                                                 self.UP3s[n], \
                                                 np.conj(V_updated))  
                        self.UP3s[n-1] = UP_updated 
                if self.LBh is not None:
                    UP_updated = oe.contract("abcd,ebcd->ae", \
                                             self.UPTs[n], \
                                             np.conj(V_updated))  
                    self.UPTs[n-1] = UP_updated 
            elif n%2 == 0:
                if self.B1s is not None:
                    UP_updated = oe.contract("vabcd,efagh,ibje,jkclm,wvnopqf,risn,soptk,uhmd->wqrtglu", \
                                             self.UP1s[n], \
                                             self.AR2s[n//2], self.hs[n+1], np.conj(self.AL2s[n//2]), \
                                             self.B1s[n//2], self.hs[n] , np.conj(self.AL1s[n//2]), \
                                             np.conj(V_updated))
                    self.UP1s[n-1] = UP_updated
                if self.B2s is not None:
                    UP_updated = oe.contract("vabcd,wvefagh,ibje,jkclm,nopqf,risn,soptk,uhmd->wqrtglu", \
                                             self.UP2s[n], \
                                             self.B2s[n//2], self.hs[n+1], np.conj(self.AL2s[n//2]), \
                                             self.AL1s[n//2], self.hs[n] , np.conj(self.AL1s[n//2]), \
                                             np.conj(V_updated))
                    self.UP2s[n-1] = UP_updated
                    if self.Lh is not None:
                        UP_updated = oe.contract("oab,cdea,fghc,poigdjk,ihelm,nkmb->pfjln", \
                                                 self.UP3s[n], \
                                                 self.Lh.Ms[n+1], self.Lh.Ms[n], \
                                                 self.B2s[n//2], np.conj(self.AL2s[n//2]), \
                                                 np.conj(V_updated))
                        self.UP3s[n-1] = UP_updated
                if self.LBh is not None:
                    UP_updated = oe.contract("ab,cdea,fghc,igdjk,ihelm,nkmb->fjln", \
                                              self.UPTs[n], \
                                              self.LBh.Ms[n+1], self.LBh.Ms[n], \
                                              self.AR2s[n//2], np.conj(self.AL2s[n//2]), \
                                              np.conj(V_updated))
                    self.UPTs[n-1] = UP_updated 
            if n <= self.N_centers-3:
                if self.B1s is not None:
                    self.DP1s[n+2] = None
                if self.B2s is not None:
                    self.DP2s[n+2] = None
                    if self.Lh is not None:
                        self.DP3s[n+2] = None
                if self.LBh is not None:
                    self.DPTs[n+2] = None
            return


class VBC_BB(TwoSiteSweep):
    def __init__(self, Bs, As, BB, BBB, chi_max, eps=1.e-15):
        assert Bs is not None or BBB is not None
        mps_guess = MPS.from_random_up_isometries(get_Ds(As, As), chi_max, norm=1.)
        super().__init__(mps_guess, chi_max, eps)
        self.Bs = Bs
        self.As = As
        self.BB = BB
        self.BBB = BBB
        if Bs is not None:
            self.DP1s = [None] * self.N_centers
            self.UP1s = [None] * self.N_centers
            if BB is not None:
                self.DP2s = [None] * self.N_centers
                self.UP2s = [None] * self.N_centers
                BBdagger = BB.copy()
                BBdagger.Ms = [np.transpose(np.conj(M), (0, 2, 1, 3)) for M in BBdagger.Ms]
                self.BBdagger = BBdagger
                self.DP3s = [None] * self.N_centers
                self.UP3s = [None] * self.N_centers
        if BBB is not None:
            self.DPTs = [None] * self.N_centers
            self.UPTs = [None] * self.N_centers
        self.init_Env()

    def get_theta_updated(self, n, theta2_guess):  
        theta2_updated = np.zeros(shape=theta2_guess.shape, dtype=np.complex128)                    
        if n%2 == 0:
            if self.Bs is not None:
                theta2_updated += oe.contract("abl,bcdefgh,lmdefij,kcm->agihjk", \
                                              self.DP1s[n], \
                                              self.Bs[n//2], np.conj(self.Bs[n//2]), \
                                              self.UP1s[n])
                if self.BB is not None:
                    theta2_updated += self.BB.norm * oe.contract("oab,acde,efgh,opicfjk,idglm,phn->bjlkmn", \
                                                                 self.DP2s[n], \
                                                                 self.BB.Ms[n], self.BB.Ms[n+1], \
                                                                 self.Bs[n//2], np.conj(self.As[n//2]), \
                                                                 self.UP2s[n])
                    theta2_updated += self.BBdagger.norm * oe.contract("oab,acde,efgh,icfjk,opidglm,phn->bjlkmn", \
                                                                      self.DP3s[n], \
                                                                      self.BBdagger.Ms[n], self.BBdagger.Ms[n+1], \
                                                                      self.As[n//2], np.conj(self.Bs[n//2]), \
                                                                      self.UP3s[n])
            if self.BBB is not None:
                theta2_updated += self.BBB.norm * oe.contract("ab,acde,efgh,icfjk,idglm,hn->bjlkmn", \
                                                              self.DPTs[n], \
                                                              self.BBB.Ms[n], self.BBB.Ms[n+1], \
                                                              self.As[n//2], np.conj(self.As[n//2]), \
                                                              self.UPTs[n])
        elif n%2 == 1:
            if self.Bs is not None:
                theta2_updated += oe.contract("abhcd,ebhfg->acdfge", \
                                              self.DP1s[n], \
                                              self.UP1s[n])
                if self.BB is not None:
                    theta2_updated += self.BB.norm * oe.contract("habcd,haefg->dbcefg", \
                                                                 self.DP2s[n], \
                                                                 self.UP2s[n])
                    theta2_updated += self.BBdagger.norm * oe.contract("habcd,haefg->dbcefg", \
                                                                       self.DP3s[n], \
                                                                       self.UP3s[n])
            if self.BBB is not None:
                theta2_updated += self.BBB.norm * oe.contract("abcd,aefg->dbcefg", \
                                                              self.DPTs[n], \
                                                              self.UPTs[n])
        return theta2_updated
    
    def init_Env(self):                         
        # Down parts for center 0
        if self.Bs is not None: 
            self.DP1s[0] = np.ones((1, 1, 1))
            if self.BB is not None:
                self.DP2s[0] = np.ones((1, 1, 1)) 
                self.DP3s[0] = np.ones((1, 1, 1))
        if self.BBB is not None:
            self.DPTs[0] = np.ones((1, 1)) 
        # Up parts for all centers
        if self.Bs is not None:
            B = self.Bs[-1][:, 0, :, :, 0, :, 0]  
            UP = oe.contract("abcd,ebcf->aedf", \
                             B, np.conj(B))
            UP = UP[np.newaxis, :, :, :, :]
            self.UP1s[-1] = UP
            if self.BB is not None:
                B = self.Bs[-1][:, 0, :, :, 0, :, 0] 
                A = self.As[-1][:, :, 0, :, 0]
                BB = self.BB.Ms[-1][:, :, :, 0] 
                UP = oe.contract("abc,gdbe,dcf->gaef", \
                                 BB, B, np.conj(A))  
                self.UP2s[-1] = UP[:, :, :, :, np.newaxis]
                BBdagger = self.BBdagger.Ms[-1][:, :, :, 0] 
                UP = oe.contract("abc,dbe,gdcf->gaef", \
                                 BBdagger, A, np.conj(B))  
                self.UP3s[-1] = UP[:, :, :, :, np.newaxis]
        if self.BBB is not None:
            A = self.As[-1][:, :, 0, :, 0] 
            BBB = self.BBB.Ms[-1][:, :, :, 0] 
            UP = oe.contract("abc,dbe,dcf->aef", \
                             BBB, A, np.conj(A))  
            self.UPTs[-1] = UP[:, :, :, np.newaxis] 
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return
    
    def update_Env(self, n, sweep_dir):                                                   
        if sweep_dir == "forth":
            U_updated = self.Us[n] 
            if n%2 == 0:
                if self.Bs is not None:
                    DP_updated = oe.contract("abl,bcdefgh,lmdefij,agik->kcmhj", \
                                             self.DP1s[n], \
                                             self.Bs[n//2], np.conj(self.Bs[n//2]), \
                                             np.conj(U_updated))
                    self.DP1s[n+1] = DP_updated 
                    if self.BB is not None:
                        DP_updated = oe.contract("oab,acde,efgh,opicfjk,idglm,bjln->phkmn", \
                                                 self.DP2s[n], \
                                                 self.BB.Ms[n], self.BB.Ms[n+1], \
                                                 self.Bs[n//2], np.conj(self.As[n//2]), \
                                                 np.conj(U_updated))
                        self.DP2s[n+1] = DP_updated 
                        DP_updated = oe.contract("oab,acde,efgh,icfjk,opidglm,bjln->phkmn", \
                                                 self.DP3s[n], \
                                                 self.BBdagger.Ms[n], self.BBdagger.Ms[n+1], \
                                                 self.As[n//2], np.conj(self.Bs[n//2]), \
                                                 np.conj(U_updated))
                        self.DP3s[n+1] = DP_updated 
                if self.BBB is not None:
                    DP_updated = oe.contract("ab,acde,efgh,icfjk,idglm,bjln->hkmn", \
                                             self.DPTs[n], \
                                             self.BBB.Ms[n], self.BBB.Ms[n+1], \
                                             self.As[n//2], np.conj(self.As[n//2]), \
                                             np.conj(U_updated))
                    self.DPTs[n+1] = DP_updated 
            elif n%2 == 1:
                if self.Bs is not None:
                    DP_updated = oe.contract("abfcd,acde->ebf", \
                                             self.DP1s[n], \
                                             np.conj(U_updated))
                    self.DP1s[n+1] = DP_updated 
                    if self.BB is not None:
                        DP_updated = oe.contract("fabcd,dbce->fae", \
                                                 self.DP2s[n], \
                                                 np.conj(U_updated))  
                        self.DP2s[n+1] = DP_updated 
                        DP_updated = oe.contract("fabcd,dbce->fae", \
                                                 self.DP3s[n], \
                                                 np.conj(U_updated))  
                        self.DP3s[n+1] = DP_updated 
                if self.BBB is not None:
                    DP_updated = oe.contract("abcd,dbce->ae", \
                                             self.DPTs[n], \
                                             np.conj(U_updated))  
                    self.DPTs[n+1] = DP_updated 
            if n >= 2:
                if self.Bs is not None:
                    self.UP1s[n-2] = None
                    if self.BB is not None:
                        self.UP2s[n-2] = None
                        self.UP3s[n-2] = None
                if self.BBB is not None:
                    self.UPTs[n-2] = None
            return
        elif sweep_dir == "back" and n > 0:
            V_updated = self.psi.Ms[n+1] 
            if n%2 == 1:
                if self.Bs is not None:
                    UP_updated = oe.contract("abfcd,ecda->ebf", \
                                             self.UP1s[n], \
                                             np.conj(V_updated))
                    self.UP1s[n-1] = UP_updated 
                    if self.BB is not None:
                        UP_updated = oe.contract("fabcd,ebcd->fae", \
                                                 self.UP2s[n], \
                                                 np.conj(V_updated))  
                        self.UP2s[n-1] = UP_updated 
                        UP_updated = oe.contract("fabcd,ebcd->fae", \
                                                 self.UP3s[n], \
                                                 np.conj(V_updated))  
                        self.UP3s[n-1] = UP_updated 
                if self.BBB is not None:
                    UP_updated = oe.contract("abcd,ebcd->ae", \
                                             self.UPTs[n], \
                                             np.conj(V_updated))  
                    self.UPTs[n-1] = UP_updated 
            elif n%2 == 0:
                if self.Bs is not None:
                    UP_updated = oe.contract("abl,cbdefgh,mldefij,khja->kcmgi", \
                                             self.UP1s[n], \
                                             self.Bs[n//2], np.conj(self.Bs[n//2]), \
                                             np.conj(V_updated))
                    self.UP1s[n-1] = UP_updated 
                    if self.BB is not None:
                        UP_updated = oe.contract("oab,cdea,fghc,poigdjk,ihelm,nkmb->pfjln", \
                                                 self.UP2s[n], \
                                                 self.BB.Ms[n+1], self.BB.Ms[n], \
                                                 self.Bs[n//2], np.conj(self.As[n//2]), \
                                                 np.conj(V_updated))
                        self.UP2s[n-1] = UP_updated
                        UP_updated = oe.contract("oab,cdea,fghc,igdjk,poihelm,nkmb->pfjln", \
                                                 self.UP3s[n], \
                                                 self.BBdagger.Ms[n+1], self.BBdagger.Ms[n], \
                                                 self.As[n//2], np.conj(self.Bs[n//2]), \
                                                 np.conj(V_updated))
                        self.UP3s[n-1] = UP_updated
                if self.BBB is not None:
                    UP_updated = oe.contract("ab,cdea,fghc,igdjk,ihelm,nkmb->fjln", \
                                              self.UPTs[n], \
                                              self.BBB.Ms[n+1], self.BBB.Ms[n], \
                                              self.As[n//2], np.conj(self.As[n//2]), \
                                              np.conj(V_updated))
                    self.UPTs[n-1] = UP_updated 
            if n <= self.N_centers-3:
                if self.Bs is not None:
                    self.DP1s[n+2] = None
                    if self.BB is not None:
                        self.DP2s[n+2] = None
                        self.DP3s[n+2] = None
                if self.BBB is not None:
                    self.DPTs[n+2] = None
            return


def Bs_from_spin_flip(g, k, ALs, CCs, CDs, direction="x"):
    # lattice parameters
    Nx = len(ALs)
    Lx = Nx // 2
    Ly = len(ALs[0])
    lattice = DiagonalSquareLattice(Lx, Ly)
    # compute coefficients
    """
    if 2*Lx*Ly <= 20:
        H = TFIModelDiagonalSquare(Lx, Ly, g).get_H()
        Es, psis = eigsh(H, k=k+1, which="SA")
        es = Es[1:] - Es[0]
        psis = psis[1:]
        print(f"es_exact = {np.array(es)}.")
    else:
        H = TFIModelDiagonalSquare(Lx, Ly, g).get_H_single_particle()
        es, psis = eigsh(H, k=k, which="SA")
        print(f"es_pert = {np.array(es)}.")
    """
    H = TFIModelDiagonalSquare(Lx, Ly, g).get_H_single_particle()
    print("hamiltonian done.")
    es, psis = eigsh(H, k=k, which="SA")
    psi = psis[:, -1]
    cs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            x, p = nx//2, nx%2
            site_scalar = lattice.get_site_scalar((x, y, p))
            cs[nx][y] = psi[site_scalar]
    # choose operator
    if direction == "x":
        op = np.array([[0., 1.], [1., 0.]])
    elif direction == "y":
        op = np.array([[0., -1.j], [1.j, 0.]])
    elif direction == "z":
        op = np.array([[1., 0.], [0., -1.]])
    # compute Bs
    Bs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            AL = ALs[nx][y].copy()
            if nx%2 == 0:
                if y == 0:
                    CD = np.ones((1, 1, 1, 1))
                elif y > 0:
                    CD = CDs[nx][2*y-1].copy()
                CC = CCs[nx][2*y].copy()
            elif nx%2 == 1:
                CD = CDs[nx][2*y].copy()
                if y < Ly-1:
                    CC = CCs[nx][2*y+1].copy()
                elif y == Ly-1:
                    CC = np.ones((1, 1, 1, 1))
            AC = oe.contract("abcde,fdgh,heij->fjabcgi", \
                             AL, CD, CC)
            B = cs[nx][y] * oe.contract("abcdefg,hc->abhdefg", \
                                        AC, op)
            Bs[nx][y] = B
    return Bs


def vecX_from_non_orthogonal_Bs(ALs, ARs, CDs, CCs, CUs, Bs, chi_max_b, eps=1.e-15):
    # lattice parameters
    Nx = len(ALs)
    Lx = Nx // 2
    Ly = len(ALs[0])
    lattice = DiagonalSquareLattice(Lx, Ly)
    # excitation dimensions
    ADs, AUs = get_ADs_AUs(ALs, CDs, CUs)
    VLs = get_VLs(ALs)
    VDs = get_VDs(CDs[-1])
    shape_Xs, shape_vecX = get_shape_Xs_vecX(ALs, CDs, CCs, CUs)
    shape_Xs_column, shape_vecX_column = get_shape_Xs_vecX_column(CDs[-1])
    # orthogonal random Bs
    vecX_random = np.random.randn(shape_vecX + shape_vecX_column) \
                  + 1.j * np.random.randn(shape_vecX + shape_vecX_column)
    vecX_random /= np.linalg.norm(vecX_random)
    Xs_random = vec_to_tensors(vecX_random[:shape_vecX], shape_Xs)
    Bs_random = Xs_to_Bs(Xs_random, VLs)
    Xs_random_column = vec_to_tensors_column(vecX_random[shape_vecX:], shape_Xs_column)
    Bs_random_column = Xs_column_to_Bs_column(Xs_random_column, VDs)
    Bs_random_double = Bs_column_to_Bs(Bs_random_column, ALs[-1], CDs[-1], CUs[-1])
    for y in range(Ly):
        if Bs_random_double[y] is not None:
            if Bs_random[-1][y] is not None:
                Bs_random[-1][y] += Bs_random_double[y]
            else:
                Bs_random[-1][y] = Bs_random_double[y]
    # non-orthogonal Bs
    Bs_sum = Bs_to_Bs_sum(Bs, ADs, AUs)
    # compute LBs 
    LBs = [None] * Nx
    for nx in range(Nx-1):
        if Bs_sum[nx] is not None:
            # extract all needed tensors
            Bs_ket = deepcopy(Bs_sum[nx])
            LB = LBs[nx-1].copy() if nx > 0 and LBs[nx-1] is not None else None
            As_ket = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(ARs[nx])]
            As_bra = deepcopy(ALs[nx])
            # flip tensors for even nx
            if nx%2 == 0:
                Bs_ket = get_flipped_Bs_sum(Bs_ket)
                LB = get_flipped_mps(LB)
                As_ket, As_bra = get_flipped_As(As_ket), get_flipped_As(As_bra)
            # perform boundary compression
            if Bs_ket is None and LB is None:
                LB = None
            else:
                try:
                    vbc = VBC_B(Bs_ket, As_bra, As_ket, LB, chi_max_b)
                    vbc.run(N_sweeps=3)
                    LB = vbc.psi
                except AssertionError:
                    LB = None
            # flip boundary mps back for even nx
            if nx%2 == 0:
                LB = get_flipped_mps(LB)
            # save boundary mps
            LBs[nx] = LB
    # compute Bs_new_L
    Bs_new_L = [[None] * Ly for _ in range(Nx)]
    for nx in range(1, Nx):
        if np.any([B is not None for B in Bs_random[nx]]) and LBs[nx-1] is not None:
            # extract all needed tensors
            LB = mps_to_tensors(LBs[nx-1])
            As_ket = [np.transpose(AR, (0, 3, 4, 1, 2)) for AR in deepcopy(ARs[nx])]
            ADs_bra, AUs_bra = deepcopy(ADs[nx]), deepcopy(AUs[nx])
            # flip tensors for odd nx
            if nx%2 == 1:
                LB = get_flipped_Cs(LB)
                As_ket, ADs_bra, AUs_bra = get_flipped_As(As_ket), get_flipped_Bs(AUs_bra), get_flipped_Bs(ADs_bra)
            # compute up and down environments
            LB = [np.ones((1, 1, 1, 1))] + LB
            DPs = [None] * Ly
            DPs[0] = np.ones((1, 1))
            for y in range(Ly-1):
                DPs[y+1] = oe.contract("ab,acde,efgh,icfjk,blidgjk->hl", \
                                        DPs[y], LB[2*y], LB[2*y+1], As_ket[y], np.conj(ADs_bra[y]))
            UPs = [None] * Ly
            UPs[-1] = np.ones((1, 1))
            for y in range(Ly-1, 0, -1):
                UPs[y-1] = oe.contract("ab,cdef,fgha,idgjk,lbiehjk->cl", \
                                        UPs[y], LB[2*y], LB[2*y+1], As_ket[y], np.conj(AUs_bra[y]))
            # compute new B tensors
            for y in range(Ly):
                if nx%2 == 0:
                    Y = y
                elif nx%2 == 1:
                    Y = Ly - 1 - y
                if Bs_random[nx][Y] is not None:
                    Bs_new_L[nx][y] = oe.contract("ab,acde,efgh,icfjk,hl->blidgjk", \
                                                  DPs[y], LB[2*y], LB[2*y+1], As_ket[y], UPs[y])
            if nx%2 == 1:
                Bs_new_L[nx] = get_flipped_Bs(Bs_new_L[nx])  
    # compute Bs_new_C
    Bs_new_C = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx): 
        if np.any([B is not None for B in Bs_random[nx]]):
            # extract all needed tensors
            Bs_ket = deepcopy(Bs_sum[nx])
            ADs_bra, AUs_bra = deepcopy(ADs[nx]), deepcopy(AUs[nx])
            # flip tensors for odd nx
            if nx%2 == 1:
                Bs_ket = get_flipped_Bs_sum(Bs_ket)
                ADs_bra, AUs_bra = get_flipped_Bs(AUs_bra), get_flipped_Bs(ADs_bra)
            # compute up and down environments
            DPs = [None] * Ly
            DPs[0] = np.ones((1, 1))
            for y in range(Ly-1):
                DPs[y+1] = oe.contract("ab,acdefgh,bidefgh->ci", \
                                       DPs[y], Bs_ket[y], np.conj(ADs_bra[y]))
            UPs = [None] * Ly
            UPs[-1] = np.ones((1, 1))
            for y in range(Ly-1, 0, -1):
                UPs[y-1] = oe.contract("ab,cadefgh,ibdefgh->ci", \
                                       UPs[y], Bs_ket[y], np.conj(AUs_bra[y]))
            # compute new B tensors
            for y in range(Ly):
                if nx%2 == 0:
                    Y = y
                elif nx%2 == 1:
                    Y = Ly - 1 - y
                if Bs_random[nx][Y] is not None:
                    Bs_new_C[nx][y] = oe.contract("ab,acdefgh,ci->bidefgh", \
                                                  DPs[y], Bs_ket[y], UPs[y])
            if nx%2 == 1:
                Bs_new_C[nx] = get_flipped_Bs(Bs_new_C[nx]) 
    # compute Bs_new
    Bs_new = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            B_new_list = [Bs_new_L[nx][y], Bs_new_C[nx][y]]
            B_new_list = [B for B in B_new_list if B is not None]
            if B_new_list:
                Bs_new[nx][y] = sum(B_new_list)
    # Bs_new -> vecX_new
    Xs_new = Bs_to_Xs(Bs_new, VLs)
    vecX_new = tensors_to_vec(Xs_new, shape_vecX)
    Xs_column_new = Bs_to_Xs_column(Bs_new[-1], ALs[-1], CDs[-1], CUs[-1], VDs)
    vecX_column_new = tensors_to_vec_column(Xs_column_new, shape_vecX_column)
    vecX = np.hstack([vecX_new, vecX_column_new])
    vecX /= np.linalg.norm(vecX)
    return vecX 


def lanczos(phi0, H, N=200, stabilize=True):
    if phi0 is None:
        phi0 = np.random.randn(H.shape[0]) + 1.j * np.random.randn(H.shape[0])
        phi0 /= np.linalg.norm(phi0)
    if phi0.ndim != 1:
        raise ValueError("phi0 should be a vector")
    if H.shape[1] != phi0.shape[0]:
        raise ValueError("shape of H does not match length of phi0")
    
    phis = []
    T = np.zeros((N, N))
    
    matvec_counter = 0
    phi0 = phi0 / np.linalg.norm(phi0)
    phis.append(phi0)
    phi = H._matvec(phi0)   #@ gives matrix product for both dense and sparse
    matvec_counter += 1
    print(f"{matvec_counter} matvecs done.")
    alpha = np.inner(phi0.conj(),phi).real
    T[0,0] = alpha
    
    phi = phi-alpha*phis[-1]
    
    for n in range(1,N):
        beta = np.linalg.norm(phi)
        if beta<1.e-13:
            print("lanczos terminated early after n={n:d} steps".format(n=n))
            T = T[:n,:n]
            break
        phi /= beta
        if stabilize:
            for vec in phis:
                phi -= vec*np.inner(vec.conj(),phi)
            phi /= np.linalg.norm(phi)
        phis.append(phi)
        phi = H._matvec(phi)-beta*phis[-2]
        matvec_counter += 1
        print(f"{matvec_counter} matvecs done.")
        alpha = np.inner(phis[-1].conj(),phi).real
        T[n,n] = alpha
        T[n-1,n] = T[n,n-1] = beta
        
        phi = phi-alpha*phis[-1]
    
    return T,phis   #phis has krylov basis states as rows