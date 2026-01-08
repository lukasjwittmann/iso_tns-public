import numpy as np
import opt_einsum as oe
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from functools import reduce
from copy import deepcopy

from ..matrix_decompositions import qr_positive
from .a_iso_peps.src.utility.tripartite_decomposition.tripartite_decomposition import tripartite_decomposition as tripartite_decomposition_base
from .a_iso_peps.src.utility.utility import split_matrix_svd
from .a_iso_peps.src.utility.utility import split_dims
from .c_mps import Sweep
from .d_expectation_values import get_expectation_value_sum, \
                                  get_flipped_As, get_flipped_hs, get_flipped_Cs, get_flipped_mps
from .e_boundary_compression import BoundaryCompression, BoundaryColumnCompression, get_compressed_boundaries


class DMRGSquaredOneSite(Sweep):
    def __init__(self, iso_peps, h_mpos, chi_max_b, N_sweeps_c, N_sweeps_b):
        iso_peps.move_orthogonality_column_to(0)
        iso_peps.move_ortho_center_to(0)
        super().__init__(psi=iso_peps, N_centers=2*iso_peps.Lx)
        self.h_mpos = [None] + h_mpos + [None]
        self.Es = []
        self.Es_updated = []
        self.D_max = iso_peps.D_max
        self.chi_max_c = iso_peps.chi_max
        self.N_sweeps_c = N_sweeps_c
        self.chi_max_b = chi_max_b
        if N_sweeps_b is not None:
            self.N_sweeps_b = N_sweeps_b
            self.bc = "variational"
        elif N_sweeps_b is None:
            self.bc = "column"
        self.Lhs = [None, None] + [None] * (self.N_centers-2)
        self.Rhs = [None] * (self.N_centers-2) + [None, None]
        self.init_Env()

    def sweep(self):
        for n in range(self.N_centers-1):
            theta_guess = self.get_theta_guess(n)
            theta_updated = self.get_theta_updated(n, theta_guess)
            self.update_psi(n, theta_updated, sweep_dir="forth")
            self.update_Env(n, sweep_dir="forth")
        for n in reversed(range(self.N_centers)):
            theta_guess = self.get_theta_guess(n)
            theta_updated = self.get_theta_updated(n, theta_guess)
            self.update_psi(n, theta_updated, sweep_dir="back")
            self.update_Env(n, sweep_dir="back")
        self.psi.move_ortho_center_to(0)

    def quarter_sweep(self):
        for n in range(self.N_centers//2):
            theta_guess = self.get_theta_guess(n)
            theta_updated = self.get_theta_updated(n, theta_guess)
            self.update_psi(n, theta_updated, sweep_dir="forth")
            self.update_Env(n, sweep_dir="forth")

    def get_theta_guess(self, n):
        assert self.psi.ortho_surface == n-1 or self.psi.ortho_surface == n
        if self.psi.ortho_surface == n-1:
            Cs = [np.transpose(C, (0, 2, 1, 3)) for C in deepcopy(self.psi.get_Cs(n))]
            As = deepcopy(self.psi.get_ARs(n))
        elif self.psi.ortho_surface == n:
            As = deepcopy(self.psi.get_ALs(n+1))
            Cs = deepcopy(self.psi.get_Cs(n+1))
        if n%2 == 0:
            assert self.psi.ortho_center == 0
            Cs = [np.ones((1, 1, 1, 1))] + Cs
        elif n%2 == 1:
            assert self.psi.ortho_center == 2*self.psi.Ly-2
            Cs = Cs + [np.ones((1, 1, 1, 1))]
        AUDs = []
        for y in range(self.psi.Ly):
            AUD = oe.contract("abcde,fdgh,heij->fjabcgi", \
                              As[y], Cs[2*y], Cs[2*y+1])
            AUDs.append(AUD)
        if self.psi.ortho_surface == n-1:
            AUDs = [np.transpose(AUD, (0, 1, 2, 5, 6, 3, 4)) for AUD in AUDs]
        if n%2 == 0:
            AUs_guess = AUDs
        elif n%2 == 1:
            AUs_guess = get_flipped_AUDs(AUDs)
        return AUs_guess

    def get_theta_updated(self, n, AUs_guess):
        print(f"Column {n+1}:")
        Lh = self.Lhs[n]
        hLs = self.h_mpos[n]
        ALs = self.psi.get_ALs(n)
        hRs = self.h_mpos[n+1]
        ARs = self.psi.get_ARs(n+1)
        Rh = self.Rhs[n]
        if n%2 == 1:
            Lh = get_flipped_mps(Lh)
            hLs = get_flipped_hs(hLs)
            ALs = get_flipped_As(ALs)
            hRs = get_flipped_hs(hRs)
            ARs = get_flipped_As(ARs)
            Rh = get_flipped_mps(Rh)
        E, EL, ECL, ECR, ER = get_expectation_value_sum(AUs_guess, Lh, hLs, ALs, hRs, ARs, Rh)
        print(f"E = EL + ECL + ECR + ER = {EL} + {ECL} + {ECR} + {ER} = {E}.")
        self.Es.append(E)
        column_dmrg = ColumnDMRG(AUs_guess, Lh, hLs, ALs, hRs, ARs, Rh)
        column_dmrg.run(self.N_sweeps_c)
        AUs_updated = column_dmrg.psi.AUs
        # print largest Hermiticity error
        max_hermiticity_error = np.max(column_dmrg.hermiticity_errors)
        print(f"- largest Hermiticity error: {max_hermiticity_error}.")
        # print maximal number of matvec operations
        max_matvec_number = np.max(column_dmrg.matvec_counters)
        print(f"- maximal number of matvec operations: {max_matvec_number}.")
        # print and save energy after update
        E_updated = column_dmrg.Es[0]
        print(f"-> E_updated = {E_updated}.")
        self.Es_updated.append(E_updated)
        return AUs_updated
    
    def update_psi(self, n, AUs_updated, sweep_dir):
        if sweep_dir == "forth":  # left -> right
            ALs_updated, CDs_updated = perform_YB_move(AUs_updated, self.D_max, self.chi_max_c, "right")
            self.psi.ortho_surface += 1
            if n%2 == 0:
                self.psi.Ws = [np.transpose(C, (1, 3, 2, 0)) for C in CDs_updated]
                self.psi.ortho_center = 2*self.psi.Ly-2
            elif n%2 == 1:
                ALs_updated = get_flipped_As(ALs_updated)
                CUs_updated = get_flipped_Cs(CDs_updated)
                self.psi.Ws = [np.transpose(C, (1, 3, 2, 0)) for C in CUs_updated]
                self.psi.ortho_center = 0
            x = n // 2
            p = n % 2
            for y in range(self.psi.Ly):
                self.psi.Ts[self.psi.get_index(x, y, p)] = np.transpose(ALs_updated[y], (0, 4, 3, 1, 2))
        elif sweep_dir == "back":  # right -> left
            ARs_updated, CDs_updated = perform_YB_move(AUs_updated, self.D_max, self.chi_max_c, "left")
            if n < self.N_centers - 1:
                self.psi.ortho_surface -= 1
            if n%2 == 0:
                self.psi.Ws = [np.transpose(C, (1, 3, 2, 0)) for C in CDs_updated]
                self.psi.ortho_center = 2*self.psi.Ly-2
            elif n%2 == 1:
                ARs_updated = get_flipped_As(ARs_updated)
                CUs_updated = get_flipped_Cs(CDs_updated)
                self.psi.Ws = [np.transpose(C, (1, 3, 2, 0)) for C in CUs_updated]
                self.psi.ortho_center = 0
            x = n // 2
            p = n % 2
            for y in range(self.psi.Ly):
                self.psi.Ts[self.psi.get_index(x, y, p)] = np.transpose(ARs_updated[y], (0, 2, 1, 3, 4))
        return
    
    def init_Env(self):
        if self.bc == "variational":
            for n in reversed(range(1, self.N_centers)):
                self.update_Env(n, sweep_dir="back")
        elif self.bc == "column":
            _, Rhs, _, _ = get_compressed_boundaries(self.psi, self.h_mpos[1:-1], self.chi_max_b, \
                                                     N_sweeps_b=None, combine_hs=True)
            self.Rhs[:-2] = Rhs[:-1][::-1]
        return

    def update_Env(self, n, sweep_dir):
        # for even/odd n all tensors have to be flipped to achieve the conventional form
        if self.bc == "variational":
            if n == 0 or n == self.N_centers-1:
                return
            if sweep_dir == "forth":
                ALs_updated = self.psi.get_ALs(n+1)
                if n%2 == 1:
                    boundary_compression = BoundaryCompression(self.psi.get_ALs(n), \
                                                               ALs_updated, \
                                                               self.h_mpos[n], \
                                                               self.Lhs[n], \
                                                               self.chi_max_b)
                elif n%2 == 0:
                    boundary_compression = BoundaryCompression(get_flipped_As(self.psi.get_ALs(n)), \
                                                               get_flipped_As(ALs_updated), \
                                                               get_flipped_hs(self.h_mpos[n]), \
                                                               get_flipped_mps(self.Lhs[n]), \
                                                               self.chi_max_b)                     
                boundary_compression.run(self.N_sweeps_b)
                Lh_updated = boundary_compression.psi
                if n%2 == 0:
                    Lh_updated = get_flipped_mps(Lh_updated)
                self.Lhs[n+1] = Lh_updated
                return
            elif sweep_dir == "back":
                ARs_updated = self.psi.get_ARs(n)
                if n%2 == 1:
                    boundary_compression = BoundaryCompression(self.psi.get_ARs(n+1), \
                                                               ARs_updated, \
                                                               self.h_mpos[n+1], \
                                                               self.Rhs[n], \
                                                               self.chi_max_b)
                elif n%2 == 0:
                    boundary_compression = BoundaryCompression(get_flipped_As(self.psi.get_ARs(n+1)), \
                                                               get_flipped_As(ARs_updated), \
                                                               get_flipped_hs(self.h_mpos[n+1]), \
                                                               get_flipped_mps(self.Rhs[n]), \
                                                               self.chi_max_b)                                    
                boundary_compression.run(self.N_sweeps_b)
                Rh_updated = boundary_compression.psi
                if n%2 == 0:
                    Rh_updated = get_flipped_mps(Rh_updated)
                self.Rhs[n-1] = Rh_updated
            return
        elif self.bc == "column":
            if n == 0 or n == self.N_centers-1:
                return
            if sweep_dir == "forth":
                ALs_updated = self.psi.get_ALs(n+1)
                Cs_updated = self.psi.get_Cs(n+1)
                if n%2 == 1:
                    boundary_compression = BoundaryColumnCompression(self.psi.get_ALs(n), \
                                                                     ALs_updated, \
                                                                     self.h_mpos[n], \
                                                                     self.Lhs[n], \
                                                                     Cs_updated, \
                                                                     "left", \
                                                                     self.chi_max_b)
                elif n%2 == 0:
                    boundary_compression = BoundaryColumnCompression(get_flipped_As(self.psi.get_ALs(n)), \
                                                                     get_flipped_As(ALs_updated), \
                                                                     get_flipped_hs(self.h_mpos[n]), \
                                                                     get_flipped_mps(self.Lhs[n]), \
                                                                     get_flipped_Cs(Cs_updated), \
                                                                     "left", \
                                                                     self.chi_max_b)                     
                Lh_updated = boundary_compression.run()
                if n%2 == 0:
                    Lh_updated = get_flipped_mps(Lh_updated)
                self.Lhs[n+1] = Lh_updated
                return
            elif sweep_dir == "back":
                ARs_updated = self.psi.get_ARs(n)
                Cs_updated = self.psi.get_Cs(n)
                if n%2 == 1:
                    boundary_compression = BoundaryColumnCompression(self.psi.get_ARs(n+1), \
                                                                     ARs_updated, \
                                                                     self.h_mpos[n+1], \
                                                                     self.Rhs[n], \
                                                                     Cs_updated, \
                                                                     "right", \
                                                                     self.chi_max_b)
                elif n%2 == 0:
                    boundary_compression = BoundaryColumnCompression(get_flipped_As(self.psi.get_ARs(n+1)), \
                                                                     get_flipped_As(ARs_updated), \
                                                                     get_flipped_hs(self.h_mpos[n+1]), \
                                                                     get_flipped_mps(self.Rhs[n]), \
                                                                     get_flipped_Cs(Cs_updated), \
                                                                     "right", \
                                                                     self.chi_max_b)                                    
                Rh_updated = boundary_compression.run()
                if n%2 == 0:
                    Rh_updated = get_flipped_mps(Rh_updated)
                self.Rhs[n-1] = Rh_updated
            return

    
class ColumnDMRG(Sweep):
    def __init__(self, AUs, Lh, hLs, ALs, hRs, ARs, Rh):
        self.Ly = len(AUs)
        ADs = [None] * self.Ly
        ACs = [AUs[0].copy()] + [None] * (self.Ly-1)
        cmps = ColumnMPS(ADs, ACs, AUs, 0)
        super().__init__(psi=cmps, N_centers=self.Ly)
        if Lh is not None:
            self.Lh = Lh.copy()
            self.Lh.Ms = [np.ones((1, 1, 1, 1))] + self.Lh.Ms
            self.DPLs = [None] * self.Ly
            self.UPLs = [None] * self.Ly
        else:
            self.Lh = None
        if Rh is not None:
            self.Rh = Rh.copy()
            self.Rh.Ms = [np.ones((1, 1, 1, 1))] + self.Rh.Ms
            self.DPRs = [None] * self.Ly
            self.UPRs = [None] * self.Ly 
        else:
            self.Rh = None
        self.hLs = hLs
        self.ALs = ALs
        if hLs is not None:
            self.DPCLs = [None] * self.Ly
            self.UPCLs = [None] * self.Ly
        self.hRs = hRs
        self.ARs = ARs
        if hRs is not None:
            self.DPCRs = [None] * self.Ly
            self.UPCRs = [None] * self.Ly
        self.init_Env()
        self.Es = [None] * self.N_centers
        self.hermiticity_errors = []
        self.matvec_counters = []

    def get_theta_guess(self, n, sweep_dir):
        assert self.psi.center == n
        return self.psi.ACs[n]
    
    def get_theta_updated(self, n, AC_guess):
        shape_AC = np.shape(AC_guess)
        H_eff_L = None
        if self.Lh is not None:
            H_eff_L = HeffBoundary(self.DPLs[n], self.UPLs[n], self.Lh.Ms[2*n], self.Lh.Ms[2*n+1], \
                                   self.Lh.norm, shape_AC, side="left")
            self.hermiticity_errors.append(get_hermiticity_error(H_eff_L))
        H_eff_R = None
        if self.Rh is not None:
            H_eff_R = HeffBoundary(self.DPRs[n], self.UPRs[n], self.Rh.Ms[2*n], self.Rh.Ms[2*n+1], \
                                   self.Rh.norm, shape_AC, side="right")
            self.hermiticity_errors.append(get_hermiticity_error(H_eff_R))
        H_eff_CL = None
        if self.hLs is not None:
            H_eff_CL = HeffCenter(self.DPCLs[n], self.UPCLs[n], self.hLs[2*n], \
                                  shape_AC, side="left")
            # Hermitian because no truncation involved
        H_eff_CR = None
        if self.hRs is not None:
            H_eff_CR = HeffCenter(self.DPCRs[n], self.UPCRs[n], self.hRs[2*n], \
                                  shape_AC, side="right")
            # Hermitian because no truncation involved
        H_eff = HeffSum([H_eff_L, H_eff_R, H_eff_CL, H_eff_CR])
        if H_eff.shape[1] <= 100: 
            Id = np.eye(H_eff.shape[1])
            H_eff_matrix = np.column_stack([H_eff @ Id[:, i] for i in range(H_eff.shape[1])])
            Es, ACs = np.linalg.eigh(H_eff_matrix)
            matvec_counter = 0
        else:
            AC_guess = np.reshape(AC_guess, H_eff.shape[1])
            try:
                Es, ACs = eigsh(H_eff, k=1, which="SA", v0=AC_guess, maxiter=50, tol=1.e-8)
            except ArpackNoConvergence as e:
                print("Warning: eigsh did not converge within maxiter=50 and tol=1e-8.")
                if e.eigenvalues.size > 0 and e.eigenvectors.size > 0:
                    Es, ACs = e.eigenvalues, e.eigenvectors
                else:
                    raise RuntimeError("No converged eigenvalue/eigenvector, ColumnDMRG cannot be continued.")
            matvec_counter = H_eff.matvec_counter
        self.matvec_counters.append(matvec_counter)
        AC_gs = np.reshape(ACs[:, 0], H_eff.shape_AC)
        E_gs = Es[0]
        self.Es[n] = E_gs
        return AC_gs

    def update_psi(self, n, AC_updated, sweep_dir):
        assert self.psi.center == n
        chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(AC_updated)
        if sweep_dir == "forth":
            AC = np.transpose(AC_updated, (0, 2, 3, 4, 5, 6, 1))
            AC = np.reshape(AC, (chi_d * d * Dld * Dlu * Drd * Dru, chi_u))
            AD, c = qr_positive(AC)
            AD = np.reshape(AD, (chi_d, d, Dld, Dlu, Drd, Dru, np.shape(AD)[1]))
            AD = np.transpose(AD, (0, 6, 1, 2, 3, 4, 5))
            self.psi.ADs[n] = AD.copy()
            if n < self.N_centers-1:
                self.psi.ACs[n+1] = oe.contract("ab,bcdefgh->acdefgh", \
                                                c, self.psi.AUs[n+1])
                self.psi.center += 1
        elif sweep_dir == "back":
            AC = np.transpose(AC_updated, (1, 2, 3, 4, 5, 6, 0))
            AC = np.reshape(AC, (chi_u * d * Dld * Dlu * Drd * Dru, chi_d))
            AU, c = qr_positive(AC)
            AU = np.reshape(AU, (chi_u, d, Dld, Dlu, Drd, Dru, np.shape(AU)[1]))
            AU = np.transpose(AU, (6, 0, 1, 2, 3, 4, 5))
            self.psi.AUs[n] = AU.copy()
            if n > 0:
                self.psi.ACs[n-1] = oe.contract("ab,cbdefgh->cadefgh", \
                                                c, self.psi.ADs[n-1])
                self.psi.center -= 1

    def init_Env(self):
        if self.Lh is not None:
            self.DPLs[0] = self.UPLs[-1] = np.ones((1, 1, 1))
        if self.Rh is not None:
            self.DPRs[0] = self.UPRs[-1] = np.ones((1, 1, 1))
        if self.hLs is not None:
            self.DPCLs[0] = np.ones((1, 1, 1, 1, 1))
            UPCL = oe.contract("abcde,fgha,hbcie->dfig", \
                               self.ALs[-1], self.hLs[-1], np.conj(self.ALs[-1]))
            self.UPCLs[-1] = UPCL[:, :, :, :, np.newaxis]
        if self.hRs is not None:
            self.DPCRs[0] = np.ones((1, 1, 1, 1, 1))
            UPCR = oe.contract("abcde,fgha,hbcie->dfig", \
                               self.ARs[-1], self.hRs[-1], np.conj(self.ARs[-1]))
            self.UPCRs[-1] = UPCR[:, :, :, :, np.newaxis]
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return 

    def update_Env(self, n, sweep_dir):
        if sweep_dir == "forth":
            AD_updated = self.psi.ADs[n]
            if self.Lh is not None:
                DPL_updated = oe.contract("abc,adef,fghi,bjkdglm,cnkehlm->ijn", \
                                          self.DPLs[n], self.Lh.Ms[2*n], self.Lh.Ms[2*n+1], \
                                          AD_updated, np.conj(AD_updated))
                self.DPLs[n+1] = DPL_updated
            if self.Rh is not None:
                DPR_updated = oe.contract("abc,adef,fghi,bjklmdg,cnklmeh->ijn", \
                                          self.DPRs[n], self.Rh.Ms[2*n], self.Rh.Ms[2*n+1], \
                                          AD_updated, np.conj(AD_updated))
                self.DPRs[n+1] = DPR_updated
            if self.hLs is not None:
                DPCL_updated = oe.contract("abcde,dfgahij,bklg,emlcnij,opqhr,ksto,tpqnu->rsufm", \
                                           self.DPCLs[n], AD_updated, self.hLs[2*n], np.conj(AD_updated), \
                                           self.ALs[n], self.hLs[2*n+1], np.conj(self.ALs[n]))
                self.DPCLs[n+1] = DPCL_updated
            if self.hRs is not None:
                DPCR_updated = oe.contract("abcde,dfgijah,bklg,emlijcn,opqhr,ksto,tpqnu->rsufm", \
                                           self.DPCRs[n], AD_updated, self.hRs[2*n], np.conj(AD_updated), \
                                           self.ARs[n], self.hRs[2*n+1], np.conj(self.ARs[n]))
                self.DPCRs[n+1] = DPCR_updated
        elif sweep_dir == "back":
            AU_updated = self.psi.AUs[n]
            if self.Lh is not None:
                UPL_updated = oe.contract("abc,defg,ghia,jbkehlm,nckfilm->djn", \
                                          self.UPLs[n], self.Lh.Ms[2*n], self.Lh.Ms[2*n+1], \
                                          AU_updated, np.conj(AU_updated))
                self.UPLs[n-1] = UPL_updated
            if self.Rh is not None:
                UPR_updated = oe.contract("abc,defg,ghia,jbklmeh,ncklmfi->djn", \
                                          self.UPRs[n], self.Rh.Ms[2*n], self.Rh.Ms[2*n+1], \
                                          AU_updated, np.conj(AU_updated))
                self.UPRs[n-1] = UPR_updated
            if self.hLs is not None:
                UPCL_updated = oe.contract("abcde,fdghaij,kblg,melncij,opqrh,skto,tpqun->rsufm", \
                                           self.UPCLs[n], AU_updated, self.hLs[2*n], np.conj(AU_updated), \
                                           self.ALs[n-1], self.hLs[2*n-1], np.conj(self.ALs[n-1]))
                self.UPCLs[n-1] = UPCL_updated
            if self.hRs is not None:
                UPCR_updated = oe.contract("abcde,fdgijha,kblg,melijnc,opqrh,skto,tpqun->rsufm", \
                                           self.UPCRs[n], AU_updated, self.hRs[2*n], np.conj(AU_updated), \
                                           self.ARs[n-1], self.hRs[2*n-1], np.conj(self.ARs[n-1]))
                self.UPCRs[n-1] = UPCR_updated


# effective Hamiltonians

class HeffBoundary(LinearOperator):
    def __init__(self, DP, UP, Bh1, Bh2, norm, shape_AC, side):
        chi_d, chi_u, d, Dld, Dlu, Drd, Dru = shape_AC
        assert np.shape(DP)[1] == np.shape(DP)[2] == chi_d
        assert np.shape(UP)[1] == np.shape(UP)[2] == chi_u
        if side == "left":
            assert np.shape(Bh1)[1] == np.shape(Bh1)[2] == Dld
            assert np.shape(Bh2)[1] == np.shape(Bh2)[2] == Dlu
        elif side == "right":
            assert np.shape(Bh1)[1] == np.shape(Bh1)[2] == Drd
            assert np.shape(Bh2)[1] == np.shape(Bh2)[2] == Dru
        shape = (chi_d * chi_u * d * Dld * Dlu * Drd * Dru, chi_d * chi_u * d * Dld * Dlu * Drd * Dru)
        dtype = reduce(np.promote_types, [DP.dtype, UP.dtype, Bh1.dtype, Bh2.dtype])
        super().__init__(dtype=dtype, shape=shape)
        self.shape_AC = shape_AC
        self.side = side
        self.DP = DP  
        self.UP = UP  
        self.Bh1 = Bh1  
        self.Bh2 = Bh2 
        self.norm = norm

    def _matvec(self, AC):
        AC = np.reshape(AC, self.shape_AC)
        if self.side == "right":
            AC = np.transpose(AC, (0, 1, 2, 5, 6, 3, 4))
        AC_new = oe.contract("abc,adef,fghi,bjkdglm,ijn->cnkehlm", \
                             self.DP, self.Bh1, self.Bh2, AC, self.UP)
        if self.side == "right":
            AC_new = np.transpose(AC_new, (0, 1, 2, 5, 6, 3, 4))
        AC_new = self.norm * np.reshape(AC_new, self.shape[0])
        return AC_new
    

class HeffCenter(LinearOperator):
    def __init__(self, DP, UP, h, shape_AC, side):
        chi_d, chi_u, d, Dld, Dlu, Drd, Dru = shape_AC
        assert np.shape(DP)[3] == np.shape(DP)[4] == chi_d
        assert np.shape(UP)[3] == np.shape(UP)[4] == chi_u
        if side == "left":
            assert np.shape(DP)[0] == np.shape(DP)[2] == Dld
            assert np.shape(UP)[0] == np.shape(UP)[2] == Dlu
        elif side == "right":
            assert np.shape(DP)[0] == np.shape(DP)[2] == Drd
            assert np.shape(UP)[0] == np.shape(UP)[2] == Dru
        shape = (chi_d * chi_u * d * Dld * Dlu * Drd * Dru, chi_d * chi_u * d * Dld * Dlu * Drd * Dru)
        dtype = reduce(np.promote_types, [DP.dtype, UP.dtype, h.dtype])
        super().__init__(dtype=dtype, shape=shape)
        self.shape_AC = shape_AC
        self.side = side
        self.DP = DP  
        self.UP = UP  
        self.h = h

    def _matvec(self, AC):
        AC = np.reshape(AC, self.shape_AC)
        if self.side == "right":
            AC = np.transpose(AC, (0, 1, 2, 5, 6, 3, 4))
        AC_new = oe.contract("abcde,dfgahij,bklg,hkmfn->enlcmij", \
                             self.DP, AC, self.h, self.UP)
        if self.side == "right":
            AC_new = np.transpose(AC_new, (0, 1, 2, 5, 6, 3, 4))
        AC_new = np.reshape(AC_new, self.shape[0])
        return AC_new
    

class HeffSum(LinearOperator):
    """Class for the sum of (at least one not None) effective Hamiltonians of the same shape."""
    def __init__(self, H_effs):
        H_effs = [H_eff for H_eff in H_effs if H_eff is not None]
        assert H_effs
        shapes = [H_eff.shape for H_eff in H_effs]
        assert all(shape == shapes[0] for shape in shapes[1:])
        shape = shapes[0]
        dtypes = [H_eff.dtype for H_eff in H_effs]
        dtype = reduce(np.promote_types, dtypes)
        super().__init__(dtype=dtype, shape=shape)
        self.H_effs = H_effs
        self.shape_AC = H_effs[0].shape_AC
        self.matvec_counter = 0

    def _matvec(self, AC):
        AC_new = np.zeros(self.shape[0], dtype=np.promote_types(self.dtype, AC.dtype))
        for H_eff in self.H_effs:
            AC_new += H_eff._matvec(AC)
        self.matvec_counter += 1
        return AC_new
    

def get_hermiticity_error(H):
    """The Hermitian adjoint of a linear operator H on an inner product space is defined via 
    <phi|H(psi)> = <H^T*(phi)|psi> for all phi, psi. H is called Hermitian if H^T* = H, i.e. if 
    <phi|H(psi)> = <H(phi)|psi> for all phi, psi. For a LinearOperator H (with _matvec method), 
    compute the hermiticity error <phi|H(psi)> - <H(phi)|psi> for random and normalized phi, psi.
    """
    assert np.shape(H)[0] == np.shape(H)[1], "Operator must be square"
    M = np.shape(H)[0]
    psi = np.random.normal(size=(M)) + 1.j * np.random.normal(size=(M))
    phi = np.random.normal(size=(M)) + 1.j * np.random.normal(size=(M))
    psi /= np.linalg.norm(psi)
    phi /= np.linalg.norm(phi)
    H_psi = H @ psi
    H_phi = H @ phi
    hermiticity_error = np.abs(np.inner(np.conj(phi), H_psi) - np.inner(np.conj(H_phi), psi))
    return hermiticity_error


class ColumnMPS:
    def __init__(self, ADs, ACs, AUs, center):
        assert len(ADs) == len(ACs) == len(AUs)
        assert np.all([AD is not None for AD in ADs[:center]])
        assert ACs[center] is not None
        assert np.all([AU is not None for AU in AUs[center+1:]])
        self.ADs = ADs
        self.ACs = ACs
        self.AUs = AUs
        self.center = center
    
def get_flipped_AUDs(AUDs):
    if AUDs is None:
        return None
    AUDs_flipped = []
    for AUD in AUDs[::-1]:
        if AUD is not None:
            AUDs_flipped.append(np.transpose(AUD, (1, 0, 2, 4, 3, 6, 5)))
        else:
            AUDs_flipped.append(None)
    return AUDs_flipped


def get_expectation_value_boundary(AUDs, Bh, side):
    if side == "right":
        AUDs = [np.transpose(AUD, (0, 1, 2, 5, 6, 3, 4)) for AUD in AUDs]
    Ms = [Bh.norm * np.ones((1, 1, 1, 1))] + Bh.Ms
    e = np.ones((1, 1, 1))
    for y in range(len(AUDs)):
        e = oe.contract("abc,adef,fghi,bjkdglm,cnkehlm->ijn", \
                        e, Ms[2*y], Ms[2*y+1], AUDs[y], np.conj(AUDs[y]))
    assert np.shape(e) == (1, 1, 1)
    return np.real_if_close(e.item())

def get_expectation_value_center(AUDs, hs, As, side):
    if side == "right":
        AUDs = [np.transpose(AUD, (0, 1, 2, 5, 6, 3, 4)) for AUD in AUDs]
    e = np.ones((1, 1, 1, 1, 1))
    for y in range(len(AUDs)):
        e = oe.contract("abcde,dfgahij,bklg,emlcnij,opqhr,ksto,tpqnu->rsufm", \
                        e, AUDs[y], hs[2*y], np.conj(AUDs[y]), As[y], hs[2*y+1], np.conj(As[y]))
    assert np.shape(e) == (1, 1, 1, 1, 1)
    return np.real_if_close(e.item())

def get_expectation_value_sum(AUDs, Lh, hLs, ALs, hRs, ARs, Rh):
    E = 0.
    EL = None
    if Lh is not None:
        EL = get_expectation_value_boundary(AUDs, Lh, "left")
        E += EL
    ECL = None
    if hLs is not None:
        ECL = get_expectation_value_center(AUDs, hLs, ALs, "left")
        E += ECL
    ECR = None
    if hRs is not None:
        ECR = get_expectation_value_center(AUDs, hRs, ARs, "right")
        E += ECR
    ER = None
    if Rh is not None:
        ER = get_expectation_value_boundary(AUDs, Rh, "right")
        E += ER
    return E, EL, ECL, ECR, ER


# Yang-Baxter move

def tripartite_decomposition(AC, D_max, chi_max):
    options = {
        "mode" : "svd",
        "disentangle": True,
        "disentangle_options": {
            "mode": "renyi_approx",
            "renyi_alpha": 0.5,
            "method": "trm",
            "N_iters": 100,
        }
    }
    assert np.abs(np.linalg.norm(AC) - 1.) < 1.e-10 
    chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(AC)
    AC = np.transpose(AC, (2, 3, 4, 0, 5, 6, 1))
    AC = np.reshape(AC, (d * Dld * Dlu, chi_d * Drd, Dru * chi_u))
    D1, D2 = split_dims(min(d * Dld * Dlu, chi_d * Drd * Dru * chi_u), D_max)
    chi = min(D1 * chi_d * Drd, D2 * Dru * chi_u, chi_max)
    AL, CD, CC = tripartite_decomposition_base(AC, D1, D2, chi, **options)
    AL = np.reshape(AL, (d, Dld, Dlu, D1, D2))
    CD = np.reshape(CD, (D1, chi_d, Drd, chi))
    CD = np.transpose(CD, (1, 0, 2, 3))
    CC = np.reshape(CC, (D2, chi, Dru, chi_u))
    CC = np.transpose(CC, (1, 0, 2, 3))
    return AL, CD, CC

def bipartite_decomposition(AC, D_max):
    assert np.abs(np.linalg.norm(AC) - 1.) < 1.e-10 
    chi_d, chi_u, d, Dld, Dlu, Drd, Dru = np.shape(AC)
    assert chi_d == Dld == Drd == 1
    AC = np.transpose(AC, (2, 3, 4, 0, 5, 6, 1))
    AC = np.reshape(AC, (d * Dlu, Dru * chi_u))
    D = min(d * Dlu, Dru * chi_u, D_max)
    AL, CC = split_matrix_svd(AC, D)
    AL = np.reshape(AL, (d, 1, Dlu, 1, D))
    CC = np.reshape(CC, (1, D, Dru, chi_u))
    return AL, CC

def perform_YB_move_right(AUs, D_max, chi_max):
    Ly = len(AUs)
    ALs = [None] * Ly
    CDs = [None] * (2*Ly-1)
    for y in range(Ly):
        AU = AUs[y].copy()
        if y == 0:
            AC = AU
            AL, CC = bipartite_decomposition(AC, D_max)
        elif y > 0:
            AC = oe.contract("ab,bcdefgh->acdefgh", c, AU)
            AL, CD1, CC = tripartite_decomposition(AC, D_max, chi_max)
            CDs[2*y-1] = CD1.copy()
        chi, D2, Dru, chi_u = np.shape(CC)
        CC = np.reshape(CC, (chi * D2 * Dru, chi_u))
        CD2, c = qr_positive(CC)
        CD2 = np.reshape(CD2, (chi, D2, Dru, np.shape(CD2)[1]))
        ALs[y] = AL.copy()
        CDs[2*y] = CD2.copy()
    assert abs(np.linalg.norm(c) - 1.) < 1e-10
    return ALs, CDs

def perform_YB_move(AUs, D_max, chi_max, side):
    if side == "right":
        ALs, CDs = perform_YB_move_right(AUs, D_max, chi_max)
        return ALs, CDs
    elif side == "left":
        AUs_flipped = [np.transpose(AU, (0, 1, 2, 5, 6, 3, 4)) for AU in AUs]
        ARs_flipped, CDs_flipped = perform_YB_move_right(AUs_flipped, D_max, chi_max)
        CDs = [np.transpose(CD, (0, 2, 1, 3)) for CD in CDs_flipped]
        return ARs_flipped, CDs