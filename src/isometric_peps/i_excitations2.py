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
from .b_model import DiagonalSquareLattice
from .c_mps import MPS, TwoSiteSweep
from .d_expectation_values import get_flipped_As, get_flipped_hs, get_flipped_Cs, get_flipped_mps, \
                                  subtract_energy_offset_mpos, get_expectation_value_boundary

from ..mps.b_model_finite import TFIModelFinite


class VariationalQuasiparticleExcitationsEngine:
    "..."
    def __init__(self, iso_peps0, h_mpos, bc, chi_max_b, eps_b=1.e-15, zeroE=False):
        # lattice parameters
        self.Lx = iso_peps0.Lx
        self.Ly = iso_peps0.Ly
        self.Nx = 2 * self.Lx
        self.Ny = 2 * self.Ly - 1
        # ground state
        self.iso_peps0 = iso_peps0
        self.D_max = iso_peps0.D_max
        self.chi_max_c = iso_peps0.chi_max
        # ground state tensors and (shapes of) excitation tensors
        self.ALs, self.ARs, self.CDs, self.CCs, self.CUs = extract_all_isometric_configurations(iso_peps0)
        self.VLs = get_VLs(self.ALs)
        self.VDs = get_VDs(self.CDs[-1])
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(self.ALs, self.CDs, self.CCs, self.CUs)
        self.shape_Xs_column, self.shape_vecX_column = get_shape_Xs_vecX_column(self.CDs[-1])
        self.ADs, self.AUs = get_ADs_AUs(self.ALs, self.CDs, self.CUs)
        # boundary compression parameters
        #self.bc = bc
        self.chi_max_b = chi_max_b
        self.eps_b = eps_b
        # Hamiltonian and ground state energy
        self.h_mpos = h_mpos
        es0 = iso_peps0.copy().get_column_expectation_values(h_mpos)
        if zeroE:
            subtract_energy_offset_mpos(h_mpos, es0)
            es0 = iso_peps0.copy().get_column_expectation_values(h_mpos)
        self.E0 = np.sum(es0)
        print(f"Initialize excitation engine from ground state isoPEPS with E0 = {self.E0}.")
        # boundaries only containing the Hamiltonian
        self.bc = "column"
        self.Lhs = self.get_Lhs()
        self.Rhs = self.get_Rhs()
        print("Compressed boundaries Lhs and Rhs only containing the Hamiltonian.")
        print(f"-> (Lh|C) = {get_expectation_value_boundary(self.CDs[-1], self.Lhs[-1], "left")}.")
        self.bc = bc

    def run(self, k):
        print(f"Optimize {k} excitation(s) above the ground state from effective Hamiltonian.")
        #Es, vecXs = eigsh(Heff(self), k, which="SA")
        Es, vecXs = eigsh(Heff(self), k, which="SA", maxiter=50, tol=1.e-4)
        vecXs = [vecXs[:, i] for i in range(k)]
        for i in range(k):
            print(f"- Excitation {i+1}:")
            vecX = vecXs[i][:self.shape_vecX]
            vecX_column = vecXs[i][self.shape_vecX:]
            self.print_all_excitation_norms(vecX, vecX_column)
            print(f"=> E_{i+1} = {Es[i]} (e{i+1} = {Es[i] - self.E0}).")
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
            CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side="left")
            Ls_list = [get_Ls_h(hs, A1s, A2s, A1s, A2s), \
                       get_Ls_transfer(Lh, A2s, A2s)]
            Lh = self.perform_boundary_compression(Ls_list, CCdaggers)
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
            CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side="right")
            Rs_list = [get_Ls_h(hs, A1s, A2s, A1s, A2s), \
                       get_Ls_transfer(Rh, A2s, A2s)]
            Rh = self.perform_boundary_compression(Rs_list, CCdaggers)
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
            Rs_list = [get_Ls_B(Bs_ket, As_bra), \
                       get_Ls_transfer(RB, As_ket, As_bra)]
            RB = self.perform_boundary_compression(Rs_list, Cs)
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
            Ls_list = [get_Ls_Bh(hs, B1s_ket, A2s_ket, A1s_bra, A2s_bra), \
                       get_Ls_hB(hs, A1s_ket, B2s_ket, A1s_bra, A2s_bra), \
                       get_Ls_LhB(Lh, B2s_ket, A2s_bra), \
                       get_Ls_transfer(LhB, A2s_ket, A2s_bra)]
            LhB = self.perform_boundary_compression(Ls_list, Cs)
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
        return vecX_new
    

class ExcitedIsometricPEPS:
    def __init__(self, D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, vecX, bc, chi_max_b, eps_b=1.e-15):
        print(f"Initialize ExcitedIsometricPEPS with {np.shape(vecX)[0]} excitations parameters.")
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
            Rs_list = [get_Ls_B(Bs_ket, As_bra), \
                       get_Ls_transfer(RBket, As_ket, As_bra)]
            RBket = self.perform_boundary_compression(Rs_list, Cs)
            # flip boundary mps back for even nx
            if nx%2 == 0:
                RBket = get_flipped_mps(RBket)
            # save boundary mps
            RBkets[nx] = RBket
        return RBkets

    def get_RBbras(self):
        RBbras = [None] * self.Nx
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
    iso_peps_copy = DiagonalIsometricPEPS(iso_peps.Lx, iso_peps.Ly, D_max=iso_peps.D_max, \
                                          chi_factor=iso_peps.chi_factor, chi_max=iso_peps.chi_max, \
                                          d=iso_peps.d, shifting_options=iso_peps.shifting_options, \
                                          yb_options=iso_peps.yb_options, \
                                          tebd_options=iso_peps.tebd_options)
    iso_peps_copy._init_as_copy(iso_peps)
    Lx = iso_peps_copy.Lx
    Ly = iso_peps_copy.Ly
    Nx = 2 * Lx
    Ny = 2 * Ly - 1
    ALs = [[None] * Ly for _ in range(Nx)]
    ARs = [[None] * Ly for _ in range(Nx)]
    CDs = [[None] * Ny for _ in range(Nx)]
    CCs = [[None] * Ny for _ in range(Nx)]
    CUs = [[None] * Ny for _ in range(Nx)]
    iso_peps_copy.move_orthogonality_column_to(0, min_dims)
    if min_dims:
        Cs = iso_peps_copy.get_Cs(0)
        C = MPS(Cs, norm=1.)
        Us, Vs, Ss, _ = C.get_canonical_form()
        iso_peps_copy.Ws = [np.transpose(U, (1, 3, 2, 0)) for U in Us]
    ARs[0] = iso_peps_copy.get_ARs(0)
    for nx in range(Nx):
        iso_peps_copy.move_orthogonality_column_to(nx+1, min_dims)
        ALs[nx] = iso_peps_copy.get_ALs(nx+1)
        if nx < Nx-1:
            ARs[nx+1] = iso_peps_copy.get_ARs(nx+1)
        Cs = iso_peps_copy.get_Cs(nx+1)
        C = MPS(Cs, norm=1.)
        Us, Vs, Ss, _ = C.get_canonical_form()
        CDs[nx] = Us
        CCs[nx] = [np.tensordot(Ss[ny], Vs[ny], axes=(1, 0)) for ny in range(Ny)]
        CUs[nx] = Vs
        if min_dims:
            if nx%2 == 0:
                iso_peps_copy.Ws = [np.transpose(V, (1, 3, 2, 0)) for V in Vs]
            elif nx%2 == 1:
                iso_peps_copy.Ws = [np.transpose(U, (1, 3, 2, 0)) for U in Us]
    return ALs, ARs, CDs, CCs, CUs

def get_ADs_AUs(ALs, CDs, CUs):
    assert len(ALs) == len(CDs) == len(CUs)
    assert len(ALs[0]) == (len(CDs[0])+1)//2 == (len(CUs[0])+1)//2
    Nx = len(ALs)
    Ly = len(ALs[0])
    ADs = [[None] * Ly for _ in range(Nx)]
    AUs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            AL = ALs[nx][y].copy()
            if nx%2 == 0:
                if y == 0:
                    CD1, CU1 = np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1))
                elif y > 0:
                    CD1, CU1 = CDs[nx][2*y-1].copy(), CUs[nx][2*y-1].copy()
                CD2, CU2 = CDs[nx][2*y].copy(), CUs[nx][2*y].copy()
            elif nx%2 == 1:
                CD1, CU1 = CDs[nx][2*y].copy(), CUs[nx][2*y].copy()
                if y < Ly-1:
                    CD2, CU2 = CDs[nx][2*y+1].copy(), CUs[nx][2*y+1].copy()
                elif y == Ly-1:
                    CD2, CU2 = np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1))
            ADs[nx][y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                     AL, CD1, CD2)
            AUs[nx][y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                     AL, CU1, CU2)
    return ADs, AUs


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