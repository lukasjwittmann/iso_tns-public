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
from .i_excitations2 import VariationalQuasiparticleExcitationsEngine, \
                            get_ADs_AUs_ACs, \
                            get_VLs, get_shape_Xs_vecX, vec_to_tensors, tensors_to_vec, Xs_to_Bs, Bs_to_Xs, \
                            get_VDs, get_shape_Xs_vecX_column, vec_to_tensors_column, tensors_to_vec_column, Xs_column_to_Bs_column, Bs_column_to_Xs_column, \
                            Bs_to_Bs_sum, get_flipped_Bs, get_flipped_Bs_sum, \
                            mps_to_tensors, h_bonds_to_mpos, get_Ds, VBC_h, VBC_B, VBC_Bh, VBC_BB

from ..mps.b_model_finite import TFIModelFinite
from ..mps.d_excitations import ExcitedMPS


class VariationalQuasiparticleExcitationsEngineMiddle(VariationalQuasiparticleExcitationsEngine):
    def __init__(self, D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, h_mpos, bc, chi_max_b, nx, eps_b=1.e-15):
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
        self.ALs = deepcopy(ALs)
        self.ARs = deepcopy(ARs)
        self.CDs = deepcopy(CDs)
        self.CCs = deepcopy(CCs)
        self.CUs = deepcopy(CUs)
        self.nx = nx
        # As
        As_left = deepcopy(ALs[:nx]) 
        As_right = deepcopy(ARs[nx:])
        As_column = deepcopy(ALs[nx-1])
        # CDs
        CDs_left = deepcopy(CDs[:nx]) 
        CDs_right = deepcopy([flip_Cs_left_right(CD) for CD in CDs[nx-1:-1]])
        CDs_column = deepcopy(CDs[nx-1])
        # CCs
        CCs_left = deepcopy(CCs[:nx]) 
        CCs_right = deepcopy([flip_Cs_left_right(CC) for CC in CCs[nx-1:-1]])
        CCs_column = deepcopy(CCs[nx-1])
        # CUs
        CUs_left = deepcopy(CUs[:nx]) 
        CUs_right = deepcopy([flip_Cs_left_right(CU) for CU in CUs[nx-1:-1]])
        CUs_column = deepcopy(CUs[nx-1])
        # shapes of excitation tensors
        ADs, AUs, ACs = get_ADs_AUs_ACs(As_left+As_right, CDs_left+CDs_right, CUs_left+CUs_right, CCs_left+CCs_right)
        self.ADs = ADs[:nx] + [flip_Bs_left_right(AD) for AD in ADs[nx:]]
        self.AUs = AUs[:nx] + [flip_Bs_left_right(AU) for AU in AUs[nx:]]
        self.ACs = ACs[:nx] + [flip_Bs_left_right(AC) for AC in ACs[nx:]]
        self.Vs_left = get_VLs(As_left)
        self.Vs_right = get_VLs(As_right)
        self.Vs_column = get_VDs(CDs_column)
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(As_left+As_right, CDs_left+CDs_right, CCs_left+CCs_right, CUs_left+CUs_right)
        self.shape_Xs_column, self.shape_vecX_column = get_shape_Xs_vecX_column(CDs_column)
        self.vecX = None
        self.Bs = None
        self.Bs_sum = None
        # Hamiltonian (and ground state energy)
        self.h_mpos = h_mpos
        # boundary compression parameters
        self.bc = bc
        self.chi_max_b = chi_max_b
        self.eps_b = eps_b
        self.Lhs = None
        self.Rhs = None
        print(f"Initialize excitation engine with boundary compression = {bc}, chi_max_b = {chi_max_b}.")

    def initialize_compressed_hamiltonian_boundaries(self):
        self.Lhs = self.get_Lhs()
        self.Rhs = self.get_Rhs()
        print("Compressed boundaries Lhs and Rhs only containing the Hamiltonian.")
        #print(f"-> (Lh|C) = {get_expectation_value_boundary(self.CDs[-1], self.Lhs[-1], "left")}. \n")

    def copy(self):
        engine_copy = VariationalQuasiparticleExcitationsEngineMiddle(self.D_max, self.chi_max_c, \
                                                                      deepcopy(self.ALs), deepcopy(self.ARs), deepcopy(self.CDs), deepcopy(self.CCs), deepcopy(self.CUs), \
                                                                      deepcopy(self.h_mpos), self.bc, self.chi_max_b, self.nx, self.eps_b)
        engine_copy.Lhs = deepcopy(self.Lhs)
        engine_copy.Rhs = deepcopy(self.Rhs)
        engine_copy.vecX = deepcopy(self.vecX)
        engine_copy.Bs = deepcopy(self.Bs)
        engine_copy.Bs_sum = deepcopy(self.Bs_sum)
        return engine_copy

    # conversions vecX <-> Bs
    def vecX_to_Bs(self, vecX):
        assert np.shape(vecX) == (self.shape_vecX + self.shape_vecX_column,)
        Xs = vec_to_tensors(vecX[:self.shape_vecX], self.shape_Xs)
        Xs_left = Xs[:self.nx]
        Xs_right = Xs[self.nx:]
        Bs_left = Xs_to_Bs(Xs_left, self.Vs_left)
        Bs_right = Xs_to_Bs(Xs_right, self.Vs_right)
        Bs = Bs_left + [flip_Bs_left_right(B_right) for B_right in Bs_right]
        Xs_column = vec_to_tensors_column(vecX[self.shape_vecX:], self.shape_Xs_column)
        Bs_column = Xs_column_to_Bs_column(Xs_column, self.Vs_column)
        Bs_double = Bs_column_to_Bs(Bs_column, self.ALs[self.nx-1], self.CDs[self.nx-1], self.CUs[self.nx-1], self.nx%2)
        for y in range(self.Ly):
            if Bs_double[y] is not None:
                if Bs[self.nx-1][y] is not None:
                    Bs[self.nx-1][y] += Bs_double[y]
                else:
                    Bs[self.nx-1][y] = Bs_double[y]
        return Bs
        
    def Bs_to_vecX(self, Bs):
        assert len(Bs) == self.Nx and len(Bs[0]) == self.Ly
        Bs_left = Bs[:self.nx]
        Bs_right = [flip_Bs_left_right(B) for B in Bs[self.nx:]]
        Xs_left = Bs_to_Xs(Bs_left, self.Vs_left)
        Xs_right = Bs_to_Xs(Bs_right, self.Vs_right)
        Xs = Xs_left + Xs_right
        vecX = tensors_to_vec(Xs, self.shape_vecX)
        Xs_column = Bs_to_Xs_column(Bs[self.nx-1], self.ALs[self.nx-1], self.CDs[self.nx-1], self.CUs[self.nx-1], self.Vs_column, self.nx%2)
        vecX_column = tensors_to_vec_column(Xs_column, self.shape_vecX_column)
        return np.hstack([vecX, vecX_column])
    
    def get_random_Bs(self):
        np.random.seed(0)
        vecX_random = np.random.randn(self.shape_vecX + self.shape_vecX_column) \
                                      + 1.j * np.random.randn(self.shape_vecX + self.shape_vecX_column)
        vecX_random /= np.linalg.norm(vecX_random)
        Bs_random = self.vecX_to_Bs(vecX_random)
        return Bs_random
    
    def initialize_excitations_from_emps_overlap(self, emps):
        print("Optimize excited isoPEPS from MPS.")
        Ms = ExcitedMPS(emps.ALs, emps.ARs, emps.vecX).get_single_mps_representation()
        Bs_nonorthogonal = get_Bs_from_mps_overlap(Ms, self.ALs, self.ARs, self.ADs, self.AUs)
        self.vecX = self.Bs_to_vecX(Bs_nonorthogonal)
        self.Bs = self.vecX_to_Bs(self.vecX)
        self.Bs_sum = Bs_to_Bs_sum(self.Bs, self.ADs, self.AUs)
        return

    def initialize_excitations(self, vecX):
        self.vecX = vecX
        self.Bs = self.vecX_to_Bs(self.vecX)
        self.Bs_sum = Bs_to_Bs_sum(self.Bs, self.ADs, self.AUs)
        return
    
    def print_all_excitation_norms(self):
        assert self.vecX is not None
        Xs = vec_to_tensors(self.vecX[:self.shape_vecX], self.shape_Xs)
        Xs_column = vec_to_tensors_column(self.vecX[self.shape_vecX:], self.shape_Xs_column)
        print("excitations AL-B-AR:")
        X2 = 0.
        for nx in range(self.Nx):
            for y in range(self.Ly):
                if Xs[nx][y] is not None:
                    X = Xs[nx][y].copy()
                    print(f"> {np.shape(X)} excitation parameters at site {nx,y} " \
                          + f"with ||X_{nx,y}||^2 = {np.linalg.norm(X)**2}.")
                    X2 += np.linalg.norm(X)**2
        print(f"-> {self.shape_vecX} excitation parameters with ||X||^2 = {X2}.")
        print("excitations AL-AL*D-AL:")
        X2_column = 0.
        for ny in range(self.Ny):
            if Xs_column[ny] is not None:
                X_column = Xs_column[ny].copy()
                print(f"> {np.shape(X_column)} excitation parameters on bond {self.nx,ny} " \
                      + f"with ||X_column_{ny}||^2 = {np.linalg.norm(X_column)**2}.")
                X2_column += np.linalg.norm(X_column)**2
        print(f"-> {self.shape_vecX_column} excitation parameters with ||X_column||^2 = {X2_column}.")
        print(f"=> {self.shape_vecX} + {self.shape_vecX_column} = {self.shape_vecX + self.shape_vecX_column} " \
              + f"excitation parameters with ||X||^2 + ||X_column||^2 = {X2+X2_column}.")
        return
        
    def initialize_compressed_excitation_boundaries(self):
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
            if self.Bs_sum[nx] is not None and self.Bs_sum[nx+1]:
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
            if self.Bs_sum[nx] is not None and self.Bs_sum[nx+1] is not None:
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
    

def flip_Cs_left_right(Cs):
    return [np.transpose(C.copy(), (0, 2, 1, 3)) for C in Cs]

def flip_As_left_right(As):
    return [np.transpose(A.copy(), (0, 3, 4, 1, 2)) for A in As]

def flip_Bs_left_right(Bs):
    if Bs is None:
        return None
    Bs_flipped = []
    for B in Bs:
        if B is not None:
            Bs_flipped.append(np.transpose(B.copy(), (0, 1, 2, 5, 6, 3, 4)))
        else:
            Bs_flipped.append(None)
    return Bs_flipped


def Bs_column_to_Bs(Bs_column, ALs, CDs, CUs, mod_nx):
    assert len(Bs_column) == len(CDs) == len(CUs) == 2*len(ALs)-1
    Ly = len(ALs)
    Bs = [None] * Ly
    if mod_nx == 1:
        if Bs_column[0] is not None:
            Bs[0] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                ALs[0], np.ones((1, 1, 1, 1)), Bs_column[0])
        for y in range(1, Ly):
            if Bs_column[2*y-1] is not None:
                Bs[y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                    ALs[y], Bs_column[2*y-1], CUs[2*y])
            if Bs_column[2*y] is not None:
                if Bs[y] is not None:
                    Bs[y] += oe.contract("abcde,fdgh,heij->fjabcgi", \
                                        ALs[y], CDs[2*y-1], Bs_column[2*y])
                else:
                    Bs[y] = oe.contract("abcde,fdgh,heij->fjabcgi", \
                                        ALs[y], CDs[2*y-1], Bs_column[2*y])
        return Bs
    elif mod_nx == 0:
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

def Bs_to_Xs_column(Bs, ALs, CDs, CUs, VDs, mod_nx):
    assert len(Bs) == len(ALs) == (len(CDs)+1)//2 == (len(CUs)+1)//2 == (len(VDs)+1)//2
    Ly = len(Bs)
    Xs_column = [None] * (2*Ly-1)
    if mod_nx == 1:
        if Bs[0] is not None:
            B = oe.contract("abcdefg,cdehi->abhifg", \
                            Bs[0], np.conj(ALs[0]))
            if VDs[0] is not None:
                X2 = oe.contract("abcdef,aceg,gdfh->hb", \
                                B, np.ones((1, 1, 1, 1)), np.conj(VDs[0]))
                Xs_column[0] = X2
        for y in range(1, Ly):
            if Bs[y] is not None:
                B_double = oe.contract("abcdefg,cdehi->abhifg", \
                                    Bs[y], np.conj(ALs[y]))
                if VDs[2*y-1] is not None:
                    X1 = oe.contract("abcdef,gdfb,aceh->hg", \
                                    B_double, np.conj(CUs[2*y]), np.conj(VDs[2*y-1]))
                    Xs_column[2*y-1] = X1
                if VDs[2*y] is not None:
                    X2 = oe.contract("abcdef,aceg,gdfh->hb", \
                                    B_double, np.conj(CDs[2*y-1]), np.conj(VDs[2*y]))
                    Xs_column[2*y] = X2
        return Xs_column
    elif mod_nx == 0:
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


def get_Bs_from_mps_overlap(Ms, ALs, ARs, ADs, AUs):
    Nx = len(ALs)
    Lx = Nx // 2
    Ly = len(ALs[0])
    Ny = 2 * Ly - 1
    assert Lx >= Ly
    Ms = [[Ms[nx * Ly + y].copy() for y in range(Ly)] for nx in range(Nx)]

    def contract_boundary(B, Ms_array, As_array, nx, side="left"):
        Ms = Ms_array[nx]
        As = As_array[nx]
        if side == "right":
            Ms = [np.transpose(M, (2, 1, 0)) for M in Ms[::-1]]
            As = [np.transpose(A, (0, 2, 1, 4, 3)) for A in As[::-1]]
            nx += 1
        B_new = B.copy()
        if nx%2 == 0:
            B_new = np.tensordot(Ms[0], B_new, axes=(0, 0))
            A = As[0]
            assert np.shape(A)[1] == np.shape(A)[3] == 1
            A = A[:, 0, :, 0, :]
            B_new = np.tensordot(B_new, np.conj(A), axes=((0, 2), (0, 1)))
            B_new = np.moveaxis(B_new, -1, 1)
            for y in range(1, Ly):
                B_new = np.tensordot(Ms[y], B_new, axes=(0, 0))
                B_new = np.tensordot(B_new, np.conj(As[y]), axes=((0, 2*(y+1)-1, 2*(y+1)), \
                                                                    (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, 2*y)
                B_new = np.moveaxis(B_new, -1, 2*y+1)
        elif nx%2 == 1:
            for y in range(Ly-1):
                B_new = np.tensordot(Ms[y], B_new, axes=(0, 0))
                B_new = np.tensordot(B_new, np.conj(As[y]), axes=((0, 2*(y+1), 2*(y+1)+1), \
                                                                    (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, 2*y+1)
                B_new = np.moveaxis(B_new, -1, 2*(y+1))
            A = As[Ly-1]
            assert np.shape(A)[2] == np.shape(A)[4] == 1
            A = A[:, :, 0, :, 0]
            B_new = np.tensordot(Ms[Ly-1], B_new, axes=(0, 0))
            B_new = np.tensordot(B_new, np.conj(A), axes=((0, 2*Ly), (0, 1)))
            B_new = np.moveaxis(B_new, -1, 2*Ly-1)
        return B_new
    
    print("Left boundaries...")
    Ls = []
    L = np.ones((1,) * (Ny+1))
    Ls.append(L)
    print(f"nx = {0} done.")
    for nx in range(Nx-1):
        L = contract_boundary(L, Ms, ALs, nx)
        Ls.append(L)
        print(f"nx = {nx+1} done.")
    print("Right boundaries...")
    Rs = []
    R = np.ones((1,) * (Ny+1))
    Rs.append(R)
    print(f"nx = {Nx-1} done.")
    for nx in reversed(range(1, Nx)):
        R = contract_boundary(R, Ms, ARs, nx, side="right")
        Rs.append(R)
        print(f"nx = {nx-1} done.")
    inverted_legs = (0,) + tuple(range(R.ndim - 1, 0, -1))
    Rs = [np.transpose(R, inverted_legs) for R in Rs[::-1]]

    Bs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        print(f"Column nx = {nx}...")
        LD = np.expand_dims(Ls[nx].copy(), axis=1)
        RU = np.expand_dims(Rs[nx].copy(), axis=1)
        Ls[nx] = None
        Rs[nx] = None
        if nx%2 == 0:
            # y = 1
            RU_new = RU.copy()
            for y in range(Ly-1, 0, -1):
                RU_new = np.tensordot(Ms[nx][y], RU_new, axes=(2, 0))
                RU_new = np.tensordot(RU_new, AUs[nx][y], axes=((1, 2, 2*y-1+3, 2*y+3), (2, 1, 5, 6)))
                RU_new = np.moveaxis(RU_new, -3, 1)
                RU_new = np.moveaxis(RU_new, -2, 2*y-1+2)
                RU_new = np.moveaxis(RU_new, -1, 2*y+2)
            RU_new = np.reshape(RU_new, np.shape(RU_new)[:3] + (np.prod(np.shape(RU_new)[3:]),))
            LD_new = np.reshape(LD.copy(), np.shape(LD)[:3] + (np.prod(np.shape(LD)[3:]),))
            B = oe.contract("abcd,efgd,ahe->bfhcg", \
                            LD_new, RU_new, Ms[nx][0])
            B = np.expand_dims(B, axis=3)
            B = np.expand_dims(B, axis=5)
            Bs[nx][0] = B.copy()
            print(f"y = {0} done.")
            AD = ADs[nx][0][:, :, :, 0, :, 0, :]
            LD = np.tensordot(Ms[nx][0], LD, axes=(0, 0))
            LD = np.tensordot(LD, AD, axes=((0, 2, 3), (2, 0, 3)))
            LD = np.moveaxis(LD, -2, 1)
            LD = np.moveaxis(LD, -1, 2)
            # y > 1
            for y in range(1, Ly):
                RU_new = RU.copy()
                for y2 in range(Ly-1, y, -1):
                    RU_new = np.tensordot(Ms[nx][y2], RU_new, axes=(2, 0))
                    RU_new = np.tensordot(RU_new, AUs[nx][y2], axes=((1, 2, 2*y2-1+3, 2*y2+3), (2, 1, 5, 6)))
                    RU_new = np.moveaxis(RU_new, -3, 1)
                    RU_new = np.moveaxis(RU_new, -2, 2*y2-1+2)
                    RU_new = np.moveaxis(RU_new, -1, 2*y2+2)
                RU_new = np.moveaxis(RU_new, 2*y-1+2, 2)
                RU_new = np.moveaxis(RU_new, 2*y+2, 3)
                RU_new = np.reshape(RU_new, np.shape(RU_new)[:4] + (np.prod(np.shape(RU_new)[4:]),))
                LD_new = np.moveaxis(LD.copy(), 2*y-1+2, 2)
                LD_new = np.moveaxis(LD_new, 2*y+2, 3)
                LD_new = np.reshape(LD_new, np.shape(LD_new)[:4] + (np.prod(np.shape(LD_new)[4:]),))
                B = oe.contract("abcde,fghie,ajf->bgjcdhi", \
                                LD_new, RU_new, Ms[nx][y])
                Bs[nx][y] = B.copy()
                print(f"y = {y} done.")
                if y < Ly-1:
                    LD = np.tensordot(Ms[nx][y], LD, axes=(0, 0))
                    LD = np.tensordot(LD, ADs[nx][y], axes=((0, 2, 2*y-1+3, 2*y+3), (2, 0, 3, 4)))
                    LD = np.moveaxis(LD, -3, 1)
                    LD = np.moveaxis(LD, -2, 2*y-1+2)
                    LD = np.moveaxis(LD, -1, 2*y+2)
        elif nx%2 == 1:
            # y < Ly
            for y in range(Ly-1):
                RU_new = RU.copy()
                AU = AUs[nx][Ly-1][:, :, :, :, 0, :, 0]
                RU_new = np.tensordot(Ms[nx][Ly-1], RU_new, axes=(2, 0))
                RU_new = np.tensordot(RU_new, AU, axes=((1, 2, 2*Ly+1), (2, 1, 4)))
                RU_new = np.moveaxis(RU_new, -2, 1)
                for y2 in range(Ly-2, y, -1):
                    RU_new = np.tensordot(Ms[nx][y2], RU_new, axes=(2, 0))
                    RU_new = np.tensordot(RU_new, AUs[nx][y2], axes=((1, 2, 2*y2+3, 2*y2+1+3), (2, 1, 5, 6)))
                    RU_new = np.moveaxis(RU_new, -3, 1)
                    RU_new = np.moveaxis(RU_new, -2, 2*y2+2)
                    RU_new = np.moveaxis(RU_new, -1, 2*y2+1+2)
                RU_new = np.moveaxis(RU_new, 2*y+2, 2)
                RU_new = np.moveaxis(RU_new, 2*y+1+2, 3)
                RU_new = np.reshape(RU_new, np.shape(RU_new)[:4] + (np.prod(np.shape(RU_new)[4:]),))
                LD_new = np.moveaxis(LD.copy(), 2*y+2, 2)
                LD_new = np.moveaxis(LD_new, 2*y+1+2, 3)
                LD_new = np.reshape(LD_new, np.shape(LD_new)[:4] + (np.prod(np.shape(LD_new)[4:]),))
                B = oe.contract("abcde,fghie,ajf->bgjcdhi", \
                                LD_new, RU_new, Ms[nx][y])
                Bs[nx][y] = B.copy()
                print(f"y = {y} done.")
                LD = np.tensordot(Ms[nx][y], LD, axes=(0, 0))
                LD = np.tensordot(LD, ADs[nx][y], axes=((0, 2, 2*y+3, 2*y+1+3), (2, 0, 3, 4)))
                LD = np.moveaxis(LD, -3, 1)
                LD = np.moveaxis(LD, -2, 2*y+2)
                LD = np.moveaxis(LD, -1, 2*y+1+2)
            # y = Ly
            RU_new = RU.copy()
            RU_new = np.moveaxis(RU_new, 2*Ly, 2)
            RU_new = np.reshape(RU_new, np.shape(RU_new)[:3] + (np.prod(np.shape(RU_new)[3:]),))
            LD_new = np.moveaxis(LD.copy(), 2*Ly, 2)
            LD_new = np.reshape(LD_new, np.shape(LD_new)[:3] + (np.prod(np.shape(LD_new)[3:]),))
            B = oe.contract("abcd,efgd,ahe->bfhcg", \
                            LD_new, RU_new, Ms[nx][Ly-1])
            B = np.expand_dims(B, axis=4)
            B = np.expand_dims(B, axis=6)
            Bs[nx][Ly-1] = B.copy()
            print(f"y = {Ly-1} done.")
    return Bs


def get_iso_peps_mps_overlap(Ms, ALs, ARs, ADs, AUs, ACs):
    print("Compute overlap of isoPEPS with MPS.")
    Nx = len(ALs)
    Lx = Nx // 2
    Ly = len(ALs[0])
    Ny = 2 * Ly - 1
    assert Lx >= Ly
    Ms = [[Ms[nx * Ly + y].copy() for y in range(Ly)] for nx in range(Nx)]

    nx_center = Lx - 1

    def contract_boundary(B, Ms_array, As_array, nx, side="left"):
        Ms = Ms_array[nx]
        As = As_array[nx]
        if side == "right":
            Ms = [np.transpose(M, (2, 1, 0)) for M in Ms[::-1]]
            As = [np.transpose(A, (0, 2, 1, 4, 3)) for A in As[::-1]]
            nx += 1
        B_new = B.copy()
        if nx%2 == 0:
            B_new = np.tensordot(Ms[0], B_new, axes=(0, 0))
            A = As[0]
            assert np.shape(A)[1] == np.shape(A)[3] == 1
            A = A[:, 0, :, 0, :]
            B_new = np.tensordot(B_new, np.conj(A), axes=((0, 2), (0, 1)))
            B_new = np.moveaxis(B_new, -1, 1)
            for y in range(1, Ly):
                B_new = np.tensordot(Ms[y], B_new, axes=(0, 0))
                B_new = np.tensordot(B_new, np.conj(As[y]), axes=((0, 2*(y+1)-1, 2*(y+1)), \
                                                                    (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, 2*y)
                B_new = np.moveaxis(B_new, -1, 2*y+1)
        elif nx%2 == 1:
            for y in range(Ly-1):
                B_new = np.tensordot(Ms[y], B_new, axes=(0, 0))
                B_new = np.tensordot(B_new, np.conj(As[y]), axes=((0, 2*(y+1), 2*(y+1)+1), \
                                                                    (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, 2*y+1)
                B_new = np.moveaxis(B_new, -1, 2*(y+1))
            A = As[Ly-1]
            assert np.shape(A)[2] == np.shape(A)[4] == 1
            A = A[:, :, 0, :, 0]
            B_new = np.tensordot(Ms[Ly-1], B_new, axes=(0, 0))
            B_new = np.tensordot(B_new, np.conj(A), axes=((0, 2*Ly), (0, 1)))
            B_new = np.moveaxis(B_new, -1, 2*Ly-1)
        return B_new
    
    print("Left boundaries...")
    Ls = [None] * Nx
    Ls[0] = np.ones((1,) * (Ny+1))
    print(f"nx = {0} done.")
    for nx in range(nx_center):
        Ls[nx+1] = contract_boundary(Ls[nx], Ms, ALs, nx)
        print(f"nx = {nx+1} done.")
    print("Right boundaries...")
    Rs = [None] * Nx
    Rs[Nx-1] = np.ones((1,) * (Ny+1))
    print(f"nx = {Nx-1} done.")
    for nx in reversed(range(nx_center+1, Nx)):
        Rs[nx-1] = contract_boundary(Rs[nx], Ms, ARs, nx, side="right")
        print(f"nx = {nx-1} done.")
    inverted_legs = (0,) + tuple(range(Rs[Nx-1].ndim - 1, 0, -1))
    for nx in range(Nx):
        if Rs[nx] is not None:
            Rs[nx] = np.transpose(Rs[nx], inverted_legs)

    Bs = [[None] * Ly for _ in range(Nx)]
    for nx in [nx_center]:
        print(f"Column nx = {nx}...")
        LD = np.expand_dims(Ls[nx].copy(), axis=1)
        RU = np.expand_dims(Rs[nx].copy(), axis=1)
        Ls[nx] = None
        Rs[nx] = None
        if nx%2 == 0:
            # y = 1
            RU_new = RU.copy()
            for y in range(Ly-1, 0, -1):
                RU_new = np.tensordot(Ms[nx][y], RU_new, axes=(2, 0))
                RU_new = np.tensordot(RU_new, AUs[nx][y], axes=((1, 2, 2*y-1+3, 2*y+3), (2, 1, 5, 6)))
                RU_new = np.moveaxis(RU_new, -3, 1)
                RU_new = np.moveaxis(RU_new, -2, 2*y-1+2)
                RU_new = np.moveaxis(RU_new, -1, 2*y+2)
            RU_new = np.reshape(RU_new, np.shape(RU_new)[:3] + (np.prod(np.shape(RU_new)[3:]),))
            LD_new = np.reshape(LD.copy(), np.shape(LD)[:3] + (np.prod(np.shape(LD)[3:]),))
            B = oe.contract("abcd,efgd,ahe->bfhcg", \
                            LD_new, RU_new, Ms[nx][0])
            B = np.expand_dims(B, axis=3)
            B = np.expand_dims(B, axis=5)
            Bs[nx][0] = B.copy()
            print(f"y = {0} done.")
            AD = ADs[nx][0][:, :, :, 0, :, 0, :]
            LD = np.tensordot(Ms[nx][0], LD, axes=(0, 0))
            LD = np.tensordot(LD, AD, axes=((0, 2, 3), (2, 0, 3)))
            LD = np.moveaxis(LD, -2, 1)
            LD = np.moveaxis(LD, -1, 2)
            # y > 1
            for y in range(1, Ly):
                RU_new = RU.copy()
                for y2 in range(Ly-1, y, -1):
                    RU_new = np.tensordot(Ms[nx][y2], RU_new, axes=(2, 0))
                    RU_new = np.tensordot(RU_new, AUs[nx][y2], axes=((1, 2, 2*y2-1+3, 2*y2+3), (2, 1, 5, 6)))
                    RU_new = np.moveaxis(RU_new, -3, 1)
                    RU_new = np.moveaxis(RU_new, -2, 2*y2-1+2)
                    RU_new = np.moveaxis(RU_new, -1, 2*y2+2)
                RU_new = np.moveaxis(RU_new, 2*y-1+2, 2)
                RU_new = np.moveaxis(RU_new, 2*y+2, 3)
                RU_new = np.reshape(RU_new, np.shape(RU_new)[:4] + (np.prod(np.shape(RU_new)[4:]),))
                LD_new = np.moveaxis(LD.copy(), 2*y-1+2, 2)
                LD_new = np.moveaxis(LD_new, 2*y+2, 3)
                LD_new = np.reshape(LD_new, np.shape(LD_new)[:4] + (np.prod(np.shape(LD_new)[4:]),))
                B = oe.contract("abcde,fghie,ajf->bgjcdhi", \
                                LD_new, RU_new, Ms[nx][y])
                Bs[nx][y] = B.copy()
                print(f"y = {y} done.")
                if y < Ly-1:
                    LD = np.tensordot(Ms[nx][y], LD, axes=(0, 0))
                    LD = np.tensordot(LD, ADs[nx][y], axes=((0, 2, 2*y-1+3, 2*y+3), (2, 0, 3, 4)))
                    LD = np.moveaxis(LD, -3, 1)
                    LD = np.moveaxis(LD, -2, 2*y-1+2)
                    LD = np.moveaxis(LD, -1, 2*y+2)
        elif nx%2 == 1:
            # y < Ly
            for y in range(Ly-1):
                RU_new = RU.copy()
                AU = AUs[nx][Ly-1][:, :, :, :, 0, :, 0]
                RU_new = np.tensordot(Ms[nx][Ly-1], RU_new, axes=(2, 0))
                RU_new = np.tensordot(RU_new, AU, axes=((1, 2, 2*Ly+1), (2, 1, 4)))
                RU_new = np.moveaxis(RU_new, -2, 1)
                for y2 in range(Ly-2, y, -1):
                    RU_new = np.tensordot(Ms[nx][y2], RU_new, axes=(2, 0))
                    RU_new = np.tensordot(RU_new, AUs[nx][y2], axes=((1, 2, 2*y2+3, 2*y2+1+3), (2, 1, 5, 6)))
                    RU_new = np.moveaxis(RU_new, -3, 1)
                    RU_new = np.moveaxis(RU_new, -2, 2*y2+2)
                    RU_new = np.moveaxis(RU_new, -1, 2*y2+1+2)
                RU_new = np.moveaxis(RU_new, 2*y+2, 2)
                RU_new = np.moveaxis(RU_new, 2*y+1+2, 3)
                RU_new = np.reshape(RU_new, np.shape(RU_new)[:4] + (np.prod(np.shape(RU_new)[4:]),))
                LD_new = np.moveaxis(LD.copy(), 2*y+2, 2)
                LD_new = np.moveaxis(LD_new, 2*y+1+2, 3)
                LD_new = np.reshape(LD_new, np.shape(LD_new)[:4] + (np.prod(np.shape(LD_new)[4:]),))
                B = oe.contract("abcde,fghie,ajf->bgjcdhi", \
                                LD_new, RU_new, Ms[nx][y])
                Bs[nx][y] = B.copy()
                print(f"y = {y} done.")
                LD = np.tensordot(Ms[nx][y], LD, axes=(0, 0))
                LD = np.tensordot(LD, ADs[nx][y], axes=((0, 2, 2*y+3, 2*y+1+3), (2, 0, 3, 4)))
                LD = np.moveaxis(LD, -3, 1)
                LD = np.moveaxis(LD, -2, 2*y+2)
                LD = np.moveaxis(LD, -1, 2*y+1+2)
            # y = Ly
            RU_new = RU.copy()
            RU_new = np.moveaxis(RU_new, 2*Ly, 2)
            RU_new = np.reshape(RU_new, np.shape(RU_new)[:3] + (np.prod(np.shape(RU_new)[3:]),))
            LD_new = np.moveaxis(LD.copy(), 2*Ly, 2)
            LD_new = np.reshape(LD_new, np.shape(LD_new)[:3] + (np.prod(np.shape(LD_new)[3:]),))
            B = oe.contract("abcd,efgd,ahe->bfhcg", \
                            LD_new, RU_new, Ms[nx][Ly-1])
            B = np.expand_dims(B, axis=4)
            B = np.expand_dims(B, axis=6)
            Bs[nx][Ly-1] = B.copy()
            print(f"y = {Ly-1} done.")

    overlaps = [[None] * Ly for _ in range(Nx)]
    for nx in [nx_center]:
        for y in range(Ly):
            assert np.shape(Bs[nx][y]) == np.shape(ACs[nx][y])
            overlaps[nx][y] = np.abs(oe.contract("...,...->", \
                                                 np.conj(ACs[nx][y]), Bs[nx][y]))
    overlap = np.max(overlaps[nx_center])
    return overlap