"""Toy code implementing the density matrix renormalization group (DMRG) algorithm over an 
orthogonality column of a diagonal isometric PEPS."""

import numpy as np
import opt_einsum as oe
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from functools import reduce

from .c_mps import TwoSiteSweep


class ColumnDMRG(TwoSiteSweep):
    """Two site sweep engine for performing the density matrix renormalization group (DMRG) 
    algorithm over a column mps (orthogonality column of isometric peps) for a Hamiltonian being the 
    sum of 
        - Lh (boundary mps resulting from mpos already applied to the left), 
        - Rh (boundary mps resulting from mpos already applied to the right), 
        - Ch (results from mpo being applied in zig-zag pattern to isometric site tensors ALs and 
          ARs belonging to the column)

    We choose the following convention (shown for the case L=2):

    E = <c_mps*|H|c_mps> = <c_mps*|Lh|c_mps> + <c_mps*|Ch|c_mps> + <c_mps*|Rh|c_mps> =

       .         ..                            .. (AR[1]h[3]AR[1]*)         ..          .
       |         ||                            ||/      / /                 ||          |
    (Lh[2])==(C[2]C[2]*)                 ==(C[2]C[2]*)==                (C[2]C[2]*)==(Rh[2])
       |         ||                   / /     /||                           ||          |
       |         ||          (AL[1]h[2]AL[1]*) ||                           ||          |
       |         ||                   \ \     \||                           ||          |
    (Lh[1])==(C[1]C[1]*)  +              ==(C[1]C[1]*)==             +  (C[1]C[1]*)==(Rh[1])
       |         ||                            ||\      \ \                 ||          |
       |         ||                            || (AR[0]h[1]AR[0]*)         ||          |
       |         ||                            ||/      / /                 ||          |
    (Lh[0])==(C[0]C[0]*)                 ==(C[0]C[0]*)==                (C[0]C[0]*)==(Rh[0])
       |         ||                   / /     /||                           ||          |
       .         ..          (AL[0]h[0]AL[0]*) ..                           ..          .

    The state we sweep along is a column mps which for every center minimizes the energy

    E = <theta2*|H_eff|theta2> with H_eff = H_eff_L + H_eff_C + H_eff_R, 
    
    where the effective Hamiltonians result from omitting theta2* in the respective expectation 
    values. The solution is given by the ground state two-site tensor of H_eff.
                             
    Parameters
    ----------
    mps, Lh, Rh, ALs, ARs, hs, chi_max, eps: Same as attributes.

    Inherits attributes from parent class TwoSiteSweep.

    Additional attributes
    ---------------------
    Lh: MPS or None
        If not None, boundary mps resulting from mpos already applied to the left.
    Rh: MPS or None
        If not None, boundary mps resulting from mpos already applied to the right.
    hs: MPO or None
        If not None, mpo sandwiched between isometric site tensors ALs and ARs of the column.
    ALs, ARs: list of np.array[ndim=5]s or None
              If hs not None, isometric site tensors between which the mpo has to be sandwiched.
    DPLs, UPLs / DPRs, UPRs: list of np.array[ndim=3]s
                             Latest down, up environments of left/right effective Hamiltonian (if 
                             not None).
    DPCs, DPLs: list of np.array[ndim=5]s
                Latest down, up environments of the center effective Hamiltonian (if not None).
    Es: list of floats
        Latest ground state energies for all centers.
    hermiticity_errors: list of floats
                        Hermiticity errors of all left and right effective Hamiltonians diagonalized
                        so far.
    Note that at least one of Lh, Rh or hs has to be not None.  
    """
    def __init__(self, mps, Lh, Rh, ALs, ARs, hs, chi_max, eps=1.e-15):
        assert mps.is_up_isometries() and mps.norm == 1.
        super().__init__(mps.copy(), chi_max, eps)
        self.Lh = Lh
        if Lh is not None:
            self.DPLs = [None] * self.N_centers
            self.UPLs = [None] * self.N_centers
        self.Rh = Rh
        if Rh is not None:
            self.DPRs = [None] * self.N_centers
            self.UPRs = [None] * self.N_centers 
        self.hs = hs
        if hs is not None:
            self.ALs = ALs
            self.ARs = ARs
            self.DPCs = [None] * self.N_centers
            self.UPCs = [None] * self.N_centers
        self.init_Env()
        self.Es = [None] * self.N_centers
        self.hermiticity_errors = []
        self.matvec_counters = []

    def get_theta_updated(self, n, theta2_guess):
        """For center n, initialize the effective Hamiltonian H_eff as a sum of all (not None) 
        H_effs, and compute its ground state. For dim(H_eff) <= 100 with exact diagonalization, else 
        with Lanczos and initial guess theta2_guess."""
        shape_theta2 = np.shape(theta2_guess)
        H_eff_L = None
        if self.Lh is not None:
            H_eff_L = HeffBoundary(self.DPLs[n], self.UPLs[n], self.Lh.Ms[n], self.Lh.Ms[n+1], \
                                   self.Lh.norm, shape_theta2, side="left")
            self.hermiticity_errors.append(get_hermiticity_error(H_eff_L))
        H_eff_R = None
        if self.Rh is not None:
            H_eff_R = HeffBoundary(self.DPRs[n], self.UPRs[n], self.Rh.Ms[n], self.Rh.Ms[n+1], \
                                   self.Rh.norm, shape_theta2, side="right")
            self.hermiticity_errors.append(get_hermiticity_error(H_eff_R))
        H_eff_C = None
        if self.hs is not None:
            if n%2 == 0:
                H_eff_C = HeffCenter(self.DPCs[n], self.UPCs[n], self.ARs[n//2], self.hs[n+1], \
                                     shape_theta2, side="right")
            elif n%2 == 1:
                H_eff_C = HeffCenter(self.DPCs[n], self.UPCs[n], self.ALs[(n+1)//2], self.hs[n+1], \
                                     shape_theta2, side="left")
            # Hermitian because no truncation involved
        H_eff = HeffSum([H_eff_L, H_eff_R, H_eff_C])
        if H_eff.shape[1] <= 100: 
            Id = np.eye(H_eff.shape[1])
            H_eff_matrix = np.column_stack([H_eff @ Id[:, i] for i in range(H_eff.shape[1])])
            Es, theta2s = np.linalg.eigh(H_eff_matrix)
            matvec_counter = 0
        else:
            theta2_guess = np.reshape(theta2_guess, H_eff.shape[1])
            try:
                Es, theta2s = eigsh(H_eff, k=1, which="SA", v0=theta2_guess, maxiter=50, tol=1.e-8)
            except ArpackNoConvergence as e:
                print("Warning: eigsh did not converge within maxiter=50 and tol=1e-8.")
                if e.eigenvalues.size > 0 and e.eigenvectors.size > 0:
                    Es, theta2s = e.eigenvalues, e.eigenvectors
                else:
                    raise RuntimeError("No converged eigenvalue/eigenvector, ColumnDMRG cannot be continued.")
            matvec_counter = H_eff.matvec_counter
        self.matvec_counters.append(matvec_counter)
        theta2_gs = np.reshape(theta2s[:, 0], H_eff.shape_theta2)
        E_gs = Es[0]
        self.Es[n] = E_gs
        return theta2_gs

    def init_Env(self):
        """Initialize down environments for first center and up environments for last center 
        (and all other up environments by calling update_Env).

        |        ||   |        ||      / /    /||      / /  /||     ||        |   ||        |
        .--(DPL)--. = .        ..  ;  .--(DPC)--. = (ALhAL*) ..  ;  .--(DPR)--. = ..        .

        .--(UPL)--. = .        ..  ;  .--(UPC)--. = .. (ARhAR*)  ;  .--(UPR)--. = ..        .
        |        ||   |        ||     ||/    / /    ||/  / /        ||        |   ||        |
        """
        # Down parts for center 0, Up parts for all centers
        if self.Lh is not None:
            self.DPLs[0] = self.UPLs[-1] = np.ones((1, 1, 1))
        if self.Rh is not None:
            self.DPRs[0] = self.UPRs[-1] = np.ones((1, 1, 1))
        if self.hs is not None:
            assert np.shape(self.ALs[0])[1] == np.shape(self.ALs[0])[3] \
                   == np.shape(self.hs[0])[0] == 1
            AL = self.ALs[0][:, 0, :, 0, :]   
            h = self.hs[0][0, :, :, :]   
            DPC = oe.contract("abc,dea,ebf->cdf", \
                              AL, h, np.conj(AL))
            DPC = DPC[:, :, :, np.newaxis, np.newaxis]
            self.DPCs[0] = DPC
            assert np.shape(self.ARs[-1])[2] == np.shape(self.ARs[-1])[4] \
                   == np.shape(self.hs[-1])[1] == 1
            AR = self.ARs[-1][:, :, 0, :, 0]
            h = self.hs[-1][:, 0, :, :]
            UPC = oe.contract("abc,dea,ebf->cdf", \
                              AR, h, np.conj(AR))
            UPC = UPC[:, :, :, np.newaxis, np.newaxis]
            self.UPCs[-1] = UPC
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return
        
    def update_Env(self, n, sweep_dir):
        """For center n and sweep_dir forth/back, update the environments DP[n+1]/UP[n-1] from
        updated U[n]/V[n+1].
        
        forth:
                             n even:               n odd:
                                      ||\    \ \    / /    /||
                                      || (ARhAR*)  (ALhAL*) ||
         |        ||                  ||/    / /    \ \    \||                  ||        |
        (Lh)=====(UU*)             ==(UU*)==            ==(UU*)==             (UU*)=====(Rh)
         |        ||            / /  /||                    ||\  \ \            ||        |
         .--(DPL)--.     ;   .--(DPC)--.                    .--(DPC)--.   ;     .--(DPR)--.
        
        back:
                             n even:               n odd:
         .--(UPL)--.     ;   .--(UPC)--.                    .--(UPC)--.   ;     .--(UPR)--.   
         |        ||            \ \  \||                    ||/  / /            ||        |
        (Lh)=====(VV*)             ==(VV*)==            ==(VV*)==             (VV*)=====(Rh)
         |        ||                  ||\   \ \     / /    /||                  ||        |
                                      || (ARhAR*)  (ALhAL*) ||
                                      ||/    / /    \ \    \||
        """
        if sweep_dir == "forth":
            U_updated = self.Us[n]
            if self.Lh is not None:
                DPL_updated = oe.contract("abc,adef,bdgh,cegi->fhi", \
                                          self.DPLs[n], self.Lh.Ms[n], U_updated, np.conj(U_updated))
                self.DPLs[n+1] = DPL_updated
            if self.Rh is not None:
                DPR_updated = oe.contract("abc,adef,bgdh,cgei->fhi", \
                                          self.DPRs[n], self.Rh.Ms[n], U_updated, np.conj(U_updated))
                self.DPRs[n+1] = DPR_updated
            if self.hs is not None:
                if n%2 == 0:
                    DPC_updated = oe.contract("abcde,dafg,echi,jklfm,bnoj,oklhp->mnpgi", \
                                              self.DPCs[n], U_updated, np.conj(U_updated), \
                                              self.ARs[n//2], self.hs[n+1], np.conj(self.ARs[n//2]))
                    self.DPCs[n+1] = DPC_updated
                elif n%2 == 1:
                    DPC_updated = oe.contract("abcde,dfag,ehci,jklfm,bnoj,oklhp->mnpgi", \
                                              self.DPCs[n], U_updated, np.conj(U_updated), \
                                              self.ALs[(n+1)//2], self.hs[n+1], np.conj(self.ALs[(n+1)//2]))
                    self.DPCs[n+1] = DPC_updated
            if n >= 2:
                if self.Lh is not None:
                    self.UPLs[n-2] = None
                if self.Rh is not None:
                    self.UPRs[n-2] = None
                if self.hs is not None:
                    self.UPCs[n-2] = None
            return
        elif sweep_dir == "back" and n > 0:
            V_updated = self.psi.Ms[n+1]
            if self.Lh is not None:
                UPL_updated = oe.contract("abc,defa,gehb,ifhc->dgi", \
                                          self.UPLs[n], self.Lh.Ms[n+1], V_updated, np.conj(V_updated))
                self.UPLs[n-1] = UPL_updated
            if self.Rh is not None:
                UPR_updated = oe.contract("abc,defa,gheb,ihfc->dgi", \
                                          self.UPRs[n], self.Rh.Ms[n+1], V_updated, np.conj(V_updated))
                self.UPRs[n-1] = UPR_updated
            if self.hs is not None:
                if n%2 == 0:
                    UPC_updated = oe.contract("abcde,fagd,hcie,jklmg,nboj,oklpi->mnpfh", \
                                              self.UPCs[n], V_updated, np.conj(V_updated), \
                                              self.ARs[n//2], self.hs[n+1], np.conj(self.ARs[n//2]))
                    self.UPCs[n-1] = UPC_updated
                elif n%2 == 1:
                    UPC_updated = oe.contract("abcde,fgad,hice,jklmg,nboj,oklpi->mnpfh", \
                                              self.UPCs[n], V_updated, np.conj(V_updated), \
                                              self.ALs[(n+1)//2], self.hs[n+1], np.conj(self.ALs[(n+1)//2]))
                    self.UPCs[n-1] = UPC_updated
            if n <= self.N_centers-3:
                if self.Lh is not None:
                    self.DPLs[n+2] = None
                if self.Rh is not None:
                    self.DPRs[n+2] = None
                if self.hs is not None:
                    self.DPCs[n+2] = None
            return
    

class HeffBoundary(LinearOperator):
    """Class for the effective Hamiltonian of the boundary mps resulting from mpos already applied 
    to the side left/right.

    side                   left                  right

                           .--(UP)--.            .--(UP)--.
                           |       ||            ||       |
                |          |        |            |        |
               (t)         |       (t)          (t)       |
             --(h)--      (Lh2)==--(h)--      --(h)--==(Rh2)
    matvec:    (e)    =    |       (e)    or    (e)       |
               (t)         |       (t)          (t)       |
             --(a)--      (Lh1)==--(a)--      --(a)--==(Rh1)
               (2')        |       (2)          (2)       |
                |          |        |            |        |
                           |       ||            ||       |
                           .--(DP)--.            .--(DP)--.
    """
    def __init__(self, DP, UP, Bh1, Bh2, norm, shape_theta2, side):
        chi_d, Dl1, Dr1, Dl2, Dr2, chi_u = shape_theta2 
        assert np.shape(DP)[1] == chi_d and np.shape(UP)[1] == chi_u
        if side == "left":
            assert np.shape(Bh1)[1] == Dl1 and np.shape(Bh2)[1] == Dl2
        elif side == "right":
            assert np.shape(Bh1)[1] == Dr1 and np.shape(Bh2)[1] == Dr2
        shape = (chi_d * Dl1 * Dr1 * Dl2 * Dr2 * chi_u, chi_d * Dl1 * Dr1 * Dl2 * Dr2 * chi_u)
        dtype = reduce(np.promote_types, [DP.dtype, UP.dtype, Bh1.dtype, Bh2.dtype])
        super().__init__(dtype=dtype, shape=shape)
        self.shape_theta2 = shape_theta2  # d l1 r1 l2 r2 u
        self.side = side
        self.DP = DP  
        self.UP = UP  
        self.Bh1 = Bh1  
        self.Bh2 = Bh2  
        self.norm = norm

    def _matvec(self, theta2):
        theta2 = np.reshape(theta2, self.shape_theta2)
        if self.side == "right":
            theta2 = np.transpose(theta2, (0, 2, 1, 4, 3, 5))
        theta2_new = oe.contract("abc,adef,fghi,bdjgkl,ilm->cejhkm", \
                                 self.DP, self.Bh1, self.Bh2, theta2, self.UP)
        if self.side == "right":
            theta2_new = np.transpose(theta2_new, (0, 2, 1, 4, 3, 5))
        theta2_new = self.norm * np.reshape(theta2_new, self.shape[0])
        return theta2_new


class HeffCenter(LinearOperator):
    """Class for the effective Hamiltonian of the mpo sandwiched between isometric site tensors ALs 
    and ARs in zig-zag pattern over the column mps.

    side                 right                       left

                         .---(UP)---.                          .---(UP)---.
                           \ \   \ ||                          || /   / /    
                |           \ \   \ |                          | /   / /    
               (t)           \ \   \|                          |/   / /     
             --(h)--             --(t)--                    --(t)--
               (e)                 (h)\  \ \            / /  /(h)       
    matvec:    (t)    =            (e) (ARhAR*)  or  (ALhAL*) (e)
               (a)                 (t)/  / /            \ \  \(t)       
             --(2)--             --(a)--                    --(a)--
               (')           / /  /(2)                        (2)\  \ \     
                |           / /  /  |                          |  \  \ \    
                           / /  /  ||                          ||  \  \ \    
                         .---(DP)---.                          .---(DP)---.
    """
    def __init__(self, DP, UP, A, h, shape_theta2, side):
        chi_d, Dl1, Dr1, Dl2, Dr2, chi_u = shape_theta2 
        assert np.shape(DP)[3] == chi_d and np.shape(UP)[3] == chi_u
        if side == "left":
            assert np.shape(A)[3] == Dl1 and np.shape(A)[4] == Dl2
        elif side == "right":
            assert np.shape(A)[3] == Dr1 and np.shape(A)[4] == Dr2
        shape = (chi_d * Dl1 * Dr1 * Dl2 * Dr2 * chi_u, chi_d * Dl1 * Dr1 * Dl2 * Dr2 * chi_u)
        dtype = reduce(np.promote_types, [DP.dtype, UP.dtype, A.dtype, h.dtype])
        super().__init__(dtype=dtype, shape=shape)
        self.shape_theta2 = shape_theta2  # d l1 r1 l2 r2 u
        self.side = side
        self.DP = DP  
        self.UP = UP  
        self.A = A
        self.h = h

    def _matvec(self, theta2):
        theta2 = np.reshape(theta2, self.shape_theta2)
        if self.side == "right":
            theta2 = np.transpose(theta2, (0, 2, 1, 4, 3, 5))
        theta2_new = oe.contract("abcde,dfaghi,jklfg,bmnj,nklop,hmqir->eocpqr", \
                                 self.DP, theta2, self.A, self.h, np.conj(self.A), self.UP) 
        if self.side == "right":
            theta2_new = np.transpose(theta2_new, (0, 2, 1, 4, 3, 5))
        theta2_new = np.reshape(theta2_new, self.shape[0])
        return theta2_new
    

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
        self.shape_theta2 = H_effs[0].shape_theta2
        self.matvec_counter = 0

    def _matvec(self, theta2):
        theta2_new = np.zeros(self.shape[0], dtype=np.promote_types(self.dtype, theta2.dtype))
        for H_eff in self.H_effs:
            theta2_new += H_eff._matvec(theta2)
        self.matvec_counter += 1
        return theta2_new
    

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