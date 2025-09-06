"""Toy code implementing two boundary compressions for the expectation value of a sum of MPOs acting 
on a diagonal isometric PEPS."""

import numpy as np
import opt_einsum as oe

from ..matrix_decompositions import qr_positive, svd_truncation
from .c_mps import MPS, TwoSiteSweep
from .d_expectation_values import get_flipped_As, get_flipped_hs, get_flipped_Cs, get_flipped_mps


class BoundaryCompression(TwoSiteSweep):
    """Two site sweep engine for compressing a boundary mps resulting from sandwiching a column mpo
    plus (possibly) transferring a boundary mps of already applied mpos, in a diagonal isometric 
    peps.

    We choose the following convention (shown for the case L=2):
                                
                    p3              p3 
                    |               |                          
                 (A2[1])         (h[3])              (A2[1]*)                      
                 /    \           / |                 / |  \                     (A2[1]A2[1]*)
                /      \         / p3*               / p3*  \            .        / /     \ \
               /                /                   /                    |       / /       \ \
        \  p2 /                /p2          \      /                 (Bh1[2])==
         \ | /                / |            \    /                      |
        (A1[1])             (h[2])          (A1[1]*)                     |
         /   \                \ |            / |  \                      |
        /     \                \p2*         / p2*  \                     |
               \                \                   \         ;          |    
                \  p1 /          \  p1               \      /        (Bh1[1])==
                 \ | /            \ |                 \    /             |       \ \       / /
                (A2[0])          (h[1])              (A2[0]*)            |        \ \     / /
                 /   \            / |                 / |  \             |       (A2[0]A2[0]*)
                /     \          / p1*               / p1*  \            |        / /     \ \
               /                /                   /                    |       / /       \ \
        \  p0 /                /p0          \      /                 (Bh1[0])==
         \ | /                / |            \    /                      |
        (A1[0])             (h[0])          (A1[0]*)                     .
                                |              |  
                               p0*            p0*  

        sandwich hs between A1sA2s and A1s*A2s* -> b_mps_start ;     transfer Bh1s with A2sA2s*
                                                                     -> b_mps_transfer

    The state we sweep along is a boundary mps which for every center minimizes the cost function

    ||b_mps - b_mps_start - b_mps_transfer||^2 
    = <b_mps*|b_mps> - <b_mps*|b_mps_start> + norm_Bh1 * <b_mps*|b_mps_transfer> + ... 
    = <theta2*|theta2> - <b_mps*|b_mps_start> + norm_Bh1 * <b_mps*|b_mps_transfer> + ... .

    The solution is given by the sum of effective two site centers 

    |theta2> = d(theta2*)(<b_mps*|b_mps_start> + norm_Bh1 * <b_mps*|b_mps_transfer>), 

    resulting from omitting theta2* in the overlap sum.

    Parameters    
    ----------
    A1s, A2s, hs, Bh1, chi_max, eps: Same as attributes.

    Inherits attributes from parent class TwoSiteSweep.

    Additional attributes
    ---------------------
    A2s: list of np.array[ndim=5]s or None
         Second column of isometric site tensors along which the mpo is applied in a zig-zag pattern
         and/or with which the boundary mps is transferred. Of conventional form as shown above.
    A1s: list of np.array[ndim=5]s or None
         If not None, first column of isometric site tensors along which the mpo is applied in a 
         zig-zag pattern. Of conventional form as shown above.
    hs: list of np.array[ndim=4]s or None
        If not None, 1d Matrix product operator to be applied to the bond column.
    DPs, UPs: list of np.array[ndim=4]s and np.array[ndim=6]s or None
              If hs not None, down/Up parts of the effective two site center for the overlap 
              <b_mps|b_mps_start>.
    Bh1: MPS or None
         If not None, boundary mps of already applied mpos to be transferred. 
    DPBs, UPBs: list of np.array[ndim=2]s and np.array[ndim=4]s
                If Bh1 not None, down/up parts of the effective two site center for the overlap
                <b_mps|b_mps_transfer>.
    Note that at least one of hs and Bh1 has to be not None.
    """
    def __init__(self, A1s, A2s, hs, Bh1, chi_max, eps=1.e-15):
        assert hs is not None or Bh1 is not None
        self.A2s = A2s
        L = len(A2s)
        Ds = []
        for y in range(L):
            D1 = np.shape(A2s[y])[3]
            Ds.append((D1, D1))
            if y < L-1:
                D2 = np.shape(A2s[y])[4]
                Ds.append((D2, D2))
        mps_guess = MPS.from_random_up_isometries(Ds, chi_max, norm=1.)
        super().__init__(mps_guess, chi_max, eps)
        self.A1s = A1s
        self.hs = hs
        if hs is not None:
            assert len(A1s) == len(A2s) == len(hs)//2
            self.DPs = [None] * self.N_centers
            self.UPs = [None] * self.N_centers
        self.Bh1 = Bh1
        if Bh1 is not None:
            assert Bh1.N == 2*len(A2s)-1
            self.DPBs = [None] * self.N_centers
            self.UPBs = [None] * self.N_centers
        self.init_Env()

    def get_theta_updated(self, n, theta2_guess):
        """For site n, compute the new effective two site center theta2 by contracting the up and 
        down environments (and one A2 stack if center is even).
        
        even n:  
                      (if h is not None)                (if Bh1 is not None)        
                                                        .----(UPB)----.
                      .----(UP)----.                    |             |
         |             \ \ \       |                   (B)             
        (t)             \ \ \                          (h)===
        (h)==            \ \ \  / /                    (11)  \ \  / /  
        (e)      =        (A2hA2*)      +    norm_Bh1 * |    (A2A2*)   
        (t)              / / /  \ \                    (B)   / /  \ \    
        (a)==           / / /                          (h)===
        (2)            / / /       |                   (10)
         |            .----(DP)----.                    |             |
                                                        .----(DPB)----.
                          
                     
        odd n:                                                        
                      (if h is not None)                      (if Bh1 is not None)
                      .-------(UP)-------.                    .----(UPB)----.
         |             \ \ \        \ \  |                    |        \ \  |       
        (t)             \ \ \        \ \                      |         \ \ 
        (h)==            \ \ \                                |  
        (e)      =        \ \ \               +    norm_Bh1 * |      
        (t)                \ \ \                              |   
        (a)==               \ \ \                             |         / / 
        (2)                  \ \ \   / /                      |        / /  |
         |                    \ \ \ / /  |                    .----(DPB)----.
                      .-------(DP)-------.                                             
        """                          
        if n%2 == 0:
            if self.hs is not None:
                theta2_updated = oe.contract("abcd,eafgh,bije,jcklm,fikn->dglhmn", \
                                             self.DPs[n], \
                                             self.A2s[n//2], self.hs[n+1], np.conj(self.A2s[n//2]), \
                                             self.UPs[n])
            if self.Bh1 is not None:
                theta2_updated_Bh1 = self.Bh1.norm * oe.contract("ab,acde,efgh,icfjk,idglm,hn->bjlkmn", \
                                                                 self.DPBs[n], \
                                                                 self.Bh1.Ms[n], self.Bh1.Ms[n+1], \
                                                                 self.A2s[n//2], np.conj(self.A2s[n//2]), \
                                                                 self.UPBs[n])
                if self.hs is not None:
                    theta2_updated += theta2_updated_Bh1
                else:
                    theta2_updated = theta2_updated_Bh1
        elif n%2 == 1:
            if self.hs is not None:
                theta2_updated = oe.contract("abcdef,abcghi->fdeghi", \
                                             self.DPs[n], \
                                             self.UPs[n])
            if self.Bh1 is not None:
                theta2_updated_Bh1 = self.Bh1.norm * oe.contract("abcd,aefg->dbcefg", \
                                                                 self.DPBs[n], \
                                                                 self.UPBs[n])
                if self.hs is not None:
                    theta2_updated += theta2_updated_Bh1
                else:
                    theta2_updated = theta2_updated_Bh1
        return theta2_updated

    def init_Env(self):
        """Initialize down environments for first center and up environments for last center 
        (and all other up environments by calling update_Env).

         / / /       |        / / /    |         |             |     |             |
        .----(DP)----.  =  (A1hA1*)    .    ;    .----(DPB)----.  =  .-------------.
        (if h is not None)                       (if Bh1 is not None)

        (if h is not None)                                 (if Bh1 is not None)
        .-------(UP)-------.  =       (A2hA2*)   .    ;    .----(UPB)----.  =   .    (A2A2*)   . 
         \ \ \        \ \  |         / / /  \ \  |         |        \ \  |      |    / /  \ \  |
                       \ \          / / /    \ \                     \ \       (B)  / /    \ \
                                   / / /                                       (h)==
                                (A1hA1*)                                       (1)
                                    \ \ \                                       |
        """                          
        # Down parts for center 0
        if self.hs is not None:
            assert np.shape(self.A1s[0])[1] == np.shape(self.A1s[0])[3] \
                   == np.shape(self.hs[0])[0] == 1
            A1 = self.A1s[0][:, 0, :, 0, :]
            h = self.hs[0][0, :, :, :]
            DP = oe.contract("abc,dea,ebg->cdg", \
                             A1, h, np.conj(A1))  
            self.DPs[0] = DP[:, :, :, np.newaxis]
        if self.Bh1 is not None:
            self.DPBs[0] = np.ones((1, 1)) 
        # Up parts for all centers
        if self.hs is not None:
            assert np.shape(self.A2s[-1])[2] == np.shape(self.A2s[-1])[4] \
                   == np.shape(self.hs[-1])[1] == 1
            A2 = self.A2s[-1][:, :, 0, :, 0] 
            h2 = self.hs[-1][:, 0, :, :]  
            A1 = self.A1s[-1] 
            h1 = self.hs[-2] 
            UP = oe.contract("abc,dea,efg,hijkb,ldmh,mijnf->klncg", \
                             A2, h2, np.conj(A2), \
                             A1, h1, np.conj(A1))
            self.UPs[-1] = UP[:, :, :, :, :, np.newaxis] 
        if self.Bh1 is not None:
            assert np.shape(self.Bh1.Ms[-1])[3] == 1
            A2 = self.A2s[-1][:, :, 0, :, 0] 
            Bh1 = self.Bh1.Ms[-1][:, :, :, 0] 
            UPB = oe.contract("abc,dbe,dcf->aef", \
                              Bh1, A2, np.conj(A2))  
            self.UPBs[-1] = UPB[:, :, :, np.newaxis] 
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")
        return
            
    def update_Env(self, n, sweep_dir):
        """For center n and sweep_dir forth/back, update the environments DP[n+1]/UP[n-1] from
        updated U[n]/V[n+1].
        
        forth:
        even n:                                          odd n:                                                     
                               |             
                              (B)             
                              (h)===
                              (11)  \ \  / /  
            \ \ \  / /         |    (A2A2*)                   / / /
            (A2hA2*)          (B)   / /  \ \               (A1hA1*)         |                     |
           / / /  \ \  |      (h)===      \ \  |              \ \ \       =(U*)                 =(U*)
          / / /      =(U*)    (10)           =(U*)             \ \ \   / /  |                / /  | 
         / / /         |       |               |                \ \ \ / /   |      |        / /   |     
        .-----(DP)-----.   ;   .-----(DPB)-----.    ;    .-------(DP)-------.  ;  .-----(DPB)-----.
        (if h is not None)     (if Bh1 is not None)      (if h is not None)       (if Bh1 is not None)

        back:
        odd n:                                             even n:
        (if h is not None)        (if Bh1 is not None)      (if h is not None)   (if Bh1 is not None)               
        .------(UP)-------.   ;   .----(UPB)-----.    ;     .----(UP)-----.   ;   .-----(UPB)-----.
         \ \ \      \ \   |       |        \ \   |           \ \ \        |       |               |
                     \ \  |                 \ \  |            \ \ \     =(V*)    (B)              |
                        =(V*)                  =(V*)           \ \ \ / /  |      (h)===         =(V*)
                          |                      |             (A2hA2*)          (11)  \ \  / /   |
                                                               / / / \ \          |    (A2A2*) 
                                                              / / /              (B)   / /  \ \
                                                           (A1hA1*)              (h)===
                                                               \ \ \             (10)
                                                                                  |
        """                                                     
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
                    DPB_updated = oe.contract("ab,acde,efgh,icfjk,idglm,bjln->hkmn", \
                                              self.DPBs[n], \
                                              self.Bh1.Ms[n], self.Bh1.Ms[n+1], \
                                              self.A2s[n//2], np.conj(self.A2s[n//2]), \
                                              np.conj(U_updated))
                    self.DPBs[n+1] = DPB_updated 
            elif n%2 == 1:
                if self.hs is not None:
                    DP_updated = oe.contract("abcdef,fdeg,hijak,blmh,mijcn->klng", \
                                             self.DPs[n], \
                                             np.conj(U_updated), \
                                             self.A1s[(n+1)//2], self.hs[n+1], np.conj(self.A1s[(n+1)//2]))
                    self.DPs[n+1] = DP_updated 
                if self.Bh1 is not None:
                    DPB_updated = oe.contract("abcd,dbce->ae", \
                                              self.DPBs[n], \
                                              np.conj(U_updated))  
                    self.DPBs[n+1] = DPB_updated 
            if n >= 2:
                if self.hs is not None:
                    self.UPs[n-2] = None
                if self.Bh1 is not None:
                    self.UPBs[n-2] = None
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
                    UPB_updated = oe.contract("abcd,ebcd->ae", \
                                              self.UPBs[n], \
                                              np.conj(V_updated))  
                    self.UPBs[n-1] = UPB_updated 
            elif n%2 == 0:
                if self.hs is not None:
                    UP_updated = oe.contract("abcd,efagh,ibje,jkclm,nopqf,risn,soptk,uhmd->qrtglu", \
                                             self.UPs[n], \
                                             self.A2s[n//2], self.hs[n+1], np.conj(self.A2s[n//2]), \
                                             self.A1s[n//2], self.hs[n] , np.conj(self.A1s[n//2]), \
                                             np.conj(V_updated))
                    self.UPs[n-1] = UP_updated 
                if self.Bh1 is not None:
                    UPB_updated = oe.contract("ab,cdea,fghc,igdjk,ihelm,nkmb->fjln", \
                                              self.UPBs[n], \
                                              self.Bh1.Ms[n+1], self.Bh1.Ms[n], \
                                              self.A2s[n//2], np.conj(self.A2s[n//2]), \
                                              np.conj(V_updated))
                    self.UPBs[n-1] = UPB_updated 
            if n <= self.N_centers-3:
                if self.hs is not None:
                    self.DPs[n+2] = None
                if self.Bh1 is not None:
                    self.DPBs[n+2] = None
            return
        

class BoundaryColumnCompressionBaseClass:
    def __init__(self, Bh1s, Bh1Cs, Bh2s, Bh2Cs, chi_max, eps=0.):
        assert Bh1s is not None or Bh2s is not None
        self.Bh1s = Bh1s
        if Bh1s is not None:
            self.N = len(Bh1s)
            self.Bh1Cs = Bh1Cs
        self.Bh2s = Bh2s
        if Bh2s is not None:
            self.N = len(Bh2s)
            self.Bh2Cs = Bh2Cs
        self.Bhs = [None] * self.N
        self.norm = None
        self.trunc_errors = [None] * (self.N-1)    
        self.chi_max = chi_max
        self.eps = eps
        
    def run(self):
        for n in range(self.N):
            self.truncate_Bh(n)
        print(f"BoundaryColumnCompression truncated <Bh|C> to chi_max_b = {self.chi_max} for Bh " \
              + f"(maximal truncation error: {np.max(self.trunc_errors)}).")
        Bh = MPS(self.Bhs, self.norm) 
        return Bh     

    def truncate_Bh(self, n):
        if self.Bh1s is not None and self.Bh2s is not None:
            Bh1 = self.Bh1s[n]
            Bh2 = self.Bh2s[n]
            assert np.shape(Bh1)[:3] == np.shape(Bh2)[:3]
            chi_d, D, _, chi_u1 = np.shape(Bh1)
            _, _, _, chi_u2 = np.shape(Bh2)
            Bh1_matrix = np.reshape(Bh1, (chi_d * D * D, chi_u1))
            Bh2_matrix = np.reshape(Bh2, (chi_d * D * D, chi_u2))
            Q, R1, R2 = qr_positive_stacked(Bh1_matrix, Bh2_matrix)
            if n < self.N-1:
                Bh1C = np.tensordot(R1, self.Bh1Cs[n+1], axes=(1, 0))
                Bh2C = np.tensordot(R2, self.Bh2Cs[n+1], axes=(1, 0))
                BhC = Bh1C + Bh2C
                U, _, _, trunc_error = svd_truncation(BhC, self.chi_max, self.eps)
                self.trunc_errors[n] = trunc_error
                Bh_matrix = Q @ U
                Bh = np.reshape(Bh_matrix, (chi_d, D, D, np.shape(U)[1]))
                self.Bhs[n] = Bh
                self.Bh1s[n+1] = oe.contract("ab,bc,cdef->adef", \
                                             np.conj(U).T, R1, self.Bh1s[n+1])
                self.Bh2s[n+1] = oe.contract("ab,bc,cdef->adef", \
                                             np.conj(U).T, R2, self.Bh2s[n+1])
            elif n == self.N-1:
                assert np.shape(R1) == np.shape(R2) == (2, 1)
                U, R = qr_positive(R1+R2)
                Bh_matrix = Q @ U
                Bh = np.reshape(Bh_matrix, (chi_d, D, D, 1))
                self.Bhs[n] = Bh
                self.norm = np.real_if_close(R[0, 0])
        elif self.Bh1s is not None and self.Bh2s is None:
            Bh1 = self.Bh1s[n]
            chi_d, D, _, chi_u1 = np.shape(Bh1)
            Bh1_matrix = np.reshape(Bh1, (chi_d * D * D, chi_u1))
            Q, R1 = qr_positive(Bh1_matrix)
            if n < self.N-1:
                Bh1C = np.tensordot(R1, self.Bh1Cs[n+1], axes=(1, 0))
                U, _, _, trunc_error = svd_truncation(Bh1C, self.chi_max, self.eps)
                self.trunc_errors[n] = trunc_error
                Bh_matrix = Q @ U
                Bh = np.reshape(Bh_matrix, (chi_d, D, D, np.shape(U)[1]))
                self.Bhs[n] = Bh
                self.Bh1s[n+1] = oe.contract("ab,bc,cdef->adef", \
                                             np.conj(U).T, R1, self.Bh1s[n+1])
            elif n == self.N-1:
                assert np.shape(R1) == (1, 1)
                self.norm = np.real_if_close(R1[0, 0])
                Bh = np.reshape(Q, (chi_d, D, D, 1))
                self.Bhs[n] = Bh
        elif self.Bh1s is None and self.Bh2s is not None:
            Bh2 = self.Bh2s[n]
            chi_d, D, _, chi_u2 = np.shape(Bh2)
            Bh2_matrix = np.reshape(Bh2, (chi_d * D * D, chi_u2))
            Q, R2 = qr_positive(Bh2_matrix)
            if n < self.N-1:
                Bh2C = np.tensordot(R2, self.Bh2Cs[n+1], axes=(1, 0))
                U, _, _, trunc_error = svd_truncation(Bh2C, self.chi_max, self.eps)
                self.trunc_errors[n] = trunc_error
                Bh_matrix = Q @ U
                Bh = np.reshape(Bh_matrix, (chi_d, D, D, np.shape(U)[1]))
                self.Bhs[n] = Bh
                self.Bh2s[n+1] = oe.contract("ab,bc,cdef->adef", \
                                             np.conj(U).T, R2, self.Bh2s[n+1])
            elif n == self.N-1:
                assert np.shape(R2) == (1, 1)
                self.norm = np.real_if_close(R2[0, 0])
                Bh = np.reshape(Q, (chi_d, D, D, 1))
                self.Bhs[n] = Bh
        self.delete_initial_tensors(n)
        return 
    
    def delete_initial_tensors(self, n):
        if self.Bh1s is not None:
            self.Bh1s[n] = None
            if n < self.N-1:
                self.Bh1Cs[n+1] = None
        if self.Bh2s is not None:
            self.Bh2s[n] = None
            if n < self.N-1:
                self.Bh2Cs[n+1] = None   
        return 


class BoundaryColumnCompression(BoundaryColumnCompressionBaseClass):
    def __init__(self, A1s, A2s, hs, Bh1, Cs, side, chi_max, eps=0.):
        assert hs is not None or Bh1 is not None
        assert A2s is not None and Cs is not None
        CCdaggers, _ = CCdaggers_to_down_isometric_form(Cs, side)
        Bh1s = None
        Bh1Cs = None
        if hs is not None:
            Bh1s = get_Bhs(A1s, A2s, hs)
            Bh1Cs = get_BhCs(A1s, A2s, hs, CCdaggers)
        Bh2s = None
        Bh2Cs = None
        if Bh1 is not None:
            Bh1_Ms = Bh1.Ms
            Bh1_Ms[0] *= Bh1.norm
            Bh2s = get_Bhs_transfer(Bh1_Ms, A2s)
            Bh2Cs = get_BhCs_transfer(Bh1_Ms, A2s, CCdaggers)
        super().__init__(Bh1s, Bh1Cs, Bh2s, Bh2Cs, chi_max, eps)
        

def qr_positive_stacked(A, B):
    assert np.shape(A)[0] == np.shape(B)[0], "A and B must have the same number of lines"
    _, N = np.shape(A)
    _, L = np.shape(B)
    AB = np.hstack([A, B])
    Q, R = qr_positive(AB)
    RA = R[:, :N]
    RB = R[:, N:(N+L)]
    return Q, RA, RB 


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
    # possible truncation
    #trunc_errors = CCdaggers_mps.compress(chi_max=96, eps=0., N_sweeps=3)
    #print(f"maximal truncation error: {np.max(trunc_errors)}.")
    CCdaggers = CCdaggers_mps.Ms
    norm = CCdaggers_mps.norm
    return CCdaggers, norm


def get_Bhs(A1s, A2s, hs):
    assert len(A1s) == len(A2s) == len(hs)//2
    L = len(A1s) 
    Bhs_double = [None] * L
    for y in range(L):
        Bh_double = oe.contract("abcde,fgha,hbcij,kelmn,gopk,pjqrs->dfimrnsloq", \
                                A1s[y], hs[2*y], np.conj(A1s[y]), \
                                A2s[y], hs[2*y+1], np.conj(A2s[y]))
        shape_Bh_double = np.shape(Bh_double)
        Bh_double = np.reshape(Bh_double, (np.prod(shape_Bh_double[:3]), \
                                           shape_Bh_double[3], \
                                           shape_Bh_double[4], \
                                           shape_Bh_double[5], \
                                           shape_Bh_double[6], \
                                           np.prod(shape_Bh_double[7:10])))
        Bhs_double[y] = Bh_double
    N = 2 * L - 1
    Bhs = [None] * N
    for y in range(L-1):
        Bh_double = Bhs_double[y]
        chi_d, Dd, _, Du, _, chi_u = np.shape(Bh_double)
        chi = min(chi_d * Dd * Dd, Du * Du * chi_u)
        Bh_double = np.reshape(Bh_double, (chi_d * Dd * Dd, Du * Du * chi_u))
        Bhd, Bhu = qr_positive(Bh_double)
        Bhd = np.reshape(Bhd, (chi_d, Dd, Dd, chi))
        Bhu = np.reshape(Bhu, (chi, Du, Du, chi_u))
        Bhs[2*y] = Bhd
        Bhs[2*y+1] = Bhu
    Bh_double = Bhs_double[L-1]
    chi_d, D, _, _, _, _ = np.shape(Bh_double)
    Bh = np.reshape(Bh_double, (chi_d, D, D, 1))
    Bhs[N-1] = Bh
    assert np.shape(Bhs[0])[0] == np.shape(Bhs[-1])[3] == 1
    return Bhs


def get_Bhs_transfer(Bh1s, A2s):
    assert len(Bh1s) == 2*len(A2s)-1
    L = len(A2s)
    Bhs_double = [None] * L
    for y in range(L-1):
        Bh_double = oe.contract("abcd,defg,hbeij,hcfkl->aikjlg", \
                                Bh1s[2*y], Bh1s[2*y+1], A2s[y], np.conj(A2s[y]))
        Bhs_double[y] = Bh_double
    assert np.shape(A2s[-1])[2] == np.shape(A2s[-1])[4] == 1
    A2 = A2s[-1][:, :, 0, :, 0]
    Bh_double = oe.contract("abcd,ebf,ecg->afgd", \
                            Bh1s[-1], A2, np.conj(A2))
    Bhs_double[L-1] = Bh_double
    N = 2 * L - 1
    Bhs = [None] * N
    for y in range(L-1):
        Bh_double = Bhs_double[y]
        chi_d, Dd, _, Du, _, chi_u = np.shape(Bh_double)
        chi = min(chi_d * Dd * Dd, Du * Du * chi_u)
        Bh_double = np.reshape(Bh_double, (chi_d * Dd * Dd, Du * Du * chi_u))
        Bhd, Bhu = qr_positive(Bh_double)
        Bhd = np.reshape(Bhd, (chi_d, Dd, Dd, chi))
        Bhu = np.reshape(Bhu, (chi, Du, Du, chi_u))
        Bhs[2*y] = Bhd
        Bhs[2*y+1] = Bhu
    Bhs[N-1] = Bhs_double[L-1]
    assert np.shape(Bhs[0])[0] == np.shape(Bhs[-1])[3] == 1
    return Bhs

     
def get_BhCs(A1s, A2s, hs, CCdaggers):
    assert len(A1s) == len(A2s) == len(hs)//2 == (len(CCdaggers)+1)//2
    L = len(A1s)
    Bhs = get_Bhs(A1s, A2s, hs)
    N = len(Bhs)
    BhCs_double = [None] * L
    BhCs =  [None] * N
    assert np.shape(A2s[-1])[2] == np.shape(A2s[-1])[4] == np.shape(hs[-1])[1] \
           == np.shape(CCdaggers[-1])[3] == 1
    A2 = A2s[-1][:, :, 0, :, 0]
    h = hs[-1][:, 0, :, :]
    CCdagger = CCdaggers[-1][:, :, :, 0]
    BhC_double = oe.contract("abcde,fgha,hbcij,kel,gmk,mjn,oln->dfio", \
                             A1s[-1], hs[-2], np.conj(A1s[-1]), \
                             A2, h, np.conj(A2), \
                             CCdagger)
    BhCs_double[L-1] = BhC_double
    for y in reversed(range(L-1)):
        BhC_double = oe.contract("abcde,fgha,hbcij,kelmn,gopk,pjqrs,tmru,unsv,loqv->dfit", \
                                 A1s[y], hs[2*y], np.conj(A1s[y]), \
                                 A2s[y], hs[2*y+1], np.conj(A2s[y]), \
                                 CCdaggers[2*y], CCdaggers[2*y+1], \
                                 BhC_double)
        BhCs_double[y] = BhC_double
    for y in range(L):
        BhC_double = BhCs_double[y].copy()
        shape_BhC_double = np.shape(BhC_double)
        BhCs_double[y] = np.reshape(BhC_double, (np.prod(shape_BhC_double[:3]), shape_BhC_double[3]))
    for y in range(L):
        BhCs[2*y] = BhCs_double[y]
        if y < L-1:
            BhCs[2*y+1] = oe.contract("abcd,ebcf,df->ae", \
                                      Bhs[2*y+1], CCdaggers[2*y+1], BhCs_double[y+1])
    return BhCs


def get_BhCs_transfer(Bh1s, A2s, CCdaggers):
    assert len(Bh1s) == 2*len(A2s)-1 == len(CCdaggers)
    L = len(A2s)
    Bhs = get_Bhs_transfer(Bh1s, A2s)
    N = len(Bhs)
    BhCs_double = [None] * L
    BhCs = [None] * N
    assert np.shape(Bh1s[-1])[3] == np.shape(A2s[-1])[2] == np.shape(A2s[-1])[4] \
           == np.shape(CCdaggers[-1])[3] == 1
    Bh1 = Bh1s[-1][:, :, :, 0]
    A2 = A2s[-1][:, :, 0, :, 0]
    CCdagger = CCdaggers[-1][:, :, :, 0]
    BhC_double = oe.contract("abc,dbe,dcf,gef->ag", \
                             Bh1, A2, np.conj(A2), CCdagger)
    BhCs_double[-1] = BhC_double
    for y in reversed(range(L-1)):
        BhC_double = oe.contract("abcd,defg,hbeij,hcfkl,mikn,njlo,go->am", \
                                 Bh1s[2*y], Bh1s[2*y+1], \
                                 A2s[y], np.conj(A2s[y]), \
                                 CCdaggers[2*y], CCdaggers[2*y+1], \
                                 BhC_double)
        BhCs_double[y] = BhC_double
    for y in range(L):
        BhCs[2*y] = BhCs_double[y]
        if y < L-1:
            BhCs[2*y+1] = oe.contract("abcd,ebcf,df->ae", \
                                      Bhs[2*y+1], CCdaggers[2*y+1], BhCs_double[y+1])
    return BhCs


def get_compressed_boundaries(iso_peps, h_mpos, chi_max_b, N_sweeps_b, combine_hs=True):
    """For an iso_peps compute the compressed boundaries for the expectation value of h_mpos. The 
    maximal bond dimension of the boundary_mps is chi_max_b. N_sweeps_b not None gives the number of
    sweeps for a variational BoundaryCompression, otherwise BoundaryColumnCompression is used. 
    
    For orthogonality column at nc, <Lh_n|/|Rh_n> (for n = 1, ..., nc-1 / nc+1, ..., 2*Lx-1) denote 
    the following compressed boundaries:
    - for combine_hs=False: h applied to column n and transferred right/left to orthogonality column,
    - for combine_hs=True: h applied to all columns left/right of n.

    Return all <Lh_n|s, |Rh_n>s, and their truncation errors.
    """
    nc = iso_peps.ortho_surface + 1
    N = 2 * iso_peps.Lx - 1
    NL = nc - 1
    NR = N - nc
    if N_sweeps_b is not None:  # variational BoundaryCompression
        # left boundaries
        Lhs = None
        trunc_errors_Lhs = None
        if NL > 0:
            Lhs = [None] * NL
            trunc_errors_Lhs = [None] * NL
            for i in range(NL):
                n = i + 1
                AL1s = iso_peps.get_ALs(n)
                AL2s = iso_peps.get_ALs(n+1)
                hs = h_mpos[n-1]
                if n%2 == 1:
                    if combine_hs:
                        if n == 1:
                            boundary_compression = BoundaryCompression(AL1s, AL2s, hs, None, \
                                                                       chi_max_b)
                        elif n > 1:
                            boundary_compression = BoundaryCompression(AL1s, AL2s, hs, Lhs[i-1], \
                                                                       chi_max_b)
                        boundary_compression.run(N_sweeps_b)
                        Lhs[i] = boundary_compression.psi
                        trunc_errors_Lhs[i] = boundary_compression.trunc_errors
                    elif not combine_hs:
                        boundary_compression_h = BoundaryCompression(AL1s, AL2s, hs, None, \
                                                                     chi_max_b)
                        boundary_compression_h.run(N_sweeps_b)
                        Lhs[i] = boundary_compression_h.psi
                        trunc_errors_Lhs[i] = boundary_compression_h.trunc_errors
                        if n > 1:
                            for j in range(i):
                                boundary_compression_t = BoundaryCompression(None, AL2s, None, Lhs[j], \
                                                                             chi_max_b)
                                boundary_compression_t.run(N_sweeps_b)
                                Lhs[j] = boundary_compression_t.psi
                                trunc_errors_Lhs[j] = boundary_compression_t.trunc_errors
                elif n%2 == 0:
                    if combine_hs:
                        boundary_compression = BoundaryCompression(get_flipped_As(AL1s), \
                                                                   get_flipped_As(AL2s), \
                                                                   get_flipped_hs(hs), \
                                                                   get_flipped_mps(Lhs[i-1]), \
                                                                   chi_max_b)
                        boundary_compression.run(N_sweeps_b)
                        Lhs[i] = get_flipped_mps(boundary_compression.psi)
                        trunc_errors_Lhs[i] = boundary_compression.trunc_errors[::-1]
                    elif not combine_hs:
                        boundary_compression_h = BoundaryCompression(get_flipped_As(AL1s), \
                                                                     get_flipped_As(AL2s), \
                                                                     get_flipped_hs(hs), \
                                                                     None, \
                                                                     chi_max_b)
                        boundary_compression_h.run(N_sweeps_b)
                        Lhs[i] = get_flipped_mps(boundary_compression_h.psi)
                        trunc_errors_Lhs[i] = boundary_compression_h.trunc_errors[::-1]
                        for j in range(i):
                            boundary_compression_t = BoundaryCompression(None, \
                                                                         get_flipped_As(AL2s), \
                                                                         None, \
                                                                         get_flipped_mps(Lhs[j]), \
                                                                         chi_max_b)
                            boundary_compression_t.run(N_sweeps_b)
                            Lhs[j] = get_flipped_mps(boundary_compression_t.psi)
                            trunc_errors_Lhs[j] = boundary_compression_t.trunc_errors[::-1]
        # right boundaries
        Rhs = None
        trunc_errors_Rhs = None
        if NR > 0:
            Rhs = [None] * NR
            trunc_errors_Rhs = [None] * NR
            for i in range(NR):
                n = N - i 
                AR1s = iso_peps.get_ARs(n)
                AR2s = iso_peps.get_ARs(n-1)
                hs = h_mpos[n-1]
                if n%2 == 1:
                    if combine_hs:
                        if n == N:
                            boundary_compression = BoundaryCompression(get_flipped_As(AR1s), \
                                                                       get_flipped_As(AR2s), \
                                                                       get_flipped_hs(hs), \
                                                                       None, \
                                                                       chi_max_b)
                        elif n < N:
                            boundary_compression = BoundaryCompression(get_flipped_As(AR1s), \
                                                                       get_flipped_As(AR2s), \
                                                                       get_flipped_hs(hs), \
                                                                       get_flipped_mps(Rhs[i-1]), \
                                                                       chi_max_b)
                        boundary_compression.run(N_sweeps_b)
                        Rhs[i] = get_flipped_mps(boundary_compression.psi)  
                        trunc_errors_Rhs[i] = boundary_compression.trunc_errors[::-1]
                    elif not combine_hs:
                        boundary_compression_h = BoundaryCompression(get_flipped_As(AR1s), \
                                                                     get_flipped_As(AR2s), \
                                                                     get_flipped_hs(hs), \
                                                                     None, \
                                                                     chi_max_b)
                        boundary_compression_h.run(N_sweeps_b)
                        Rhs[i] = get_flipped_mps(boundary_compression_h.psi)
                        trunc_errors_Rhs[i] = boundary_compression_h.trunc_errors[::-1]
                        if n < N:
                            for j in range(i):
                                boundary_compression_t = BoundaryCompression(None, \
                                                                             get_flipped_As(AR2s), \
                                                                             None, \
                                                                             get_flipped_mps(Rhs[j]), \
                                                                             chi_max_b)
                                boundary_compression_t.run(N_sweeps_b)
                                Rhs[j] = get_flipped_mps(boundary_compression_t.psi)
                                trunc_errors_Rhs[j] = boundary_compression_t.trunc_errors[::-1]
                elif n%2 == 0:
                    if combine_hs:
                        boundary_compression = BoundaryCompression(AR1s, AR2s, hs, Rhs[i-1], \
                                                                   chi_max_b)
                        boundary_compression.run(N_sweeps_b)
                        Rhs[i] = boundary_compression.psi
                        trunc_errors_Rhs[i] = boundary_compression.trunc_errors
                    elif not combine_hs:
                        boundary_compression_h = BoundaryCompression(AR1s, AR2s, hs, None, \
                                                                     chi_max_b)
                        boundary_compression_h.run(N_sweeps_b)
                        Rhs[i] = boundary_compression_h.psi 
                        trunc_errors_Rhs[i] = boundary_compression_h.trunc_errors    
                        for j in range(i):           
                            boundary_compression_t = BoundaryCompression(None, AR2s, None, Rhs[j], \
                                                                         chi_max_b)
                            boundary_compression_t.run(N_sweeps_b)
                            Rhs[j] = boundary_compression_t.psi
                            trunc_errors_Rhs[j] = boundary_compression_t.trunc_errors
    elif N_sweeps_b is None:  # BoundaryColumnCompression
        # left boundaries
        Lhs = None
        trunc_errors_Lhs = None
        if NL > 0:
            Lhs = [None] * NL
            trunc_errors_Lhs = [None] * NL
            for i in range(NL):
                n = i + 1
                iso_peps_copy = iso_peps.copy()
                iso_peps_copy.move_orthogonality_column_to(n+1, min_dims=True)
                AL1s = iso_peps_copy.get_ALs(n)
                AL2s = iso_peps_copy.get_ALs(n+1)
                Cs = iso_peps_copy.get_Cs(n+1)
                hs = h_mpos[n-1]
                if n%2 == 1:
                    if combine_hs:
                        if n == 1:
                            boundary_column_compression = BoundaryColumnCompression(AL1s, AL2s, hs, \
                                                                                    None, Cs, "left", \
                                                                                    chi_max_b)
                        elif n > 1:
                            boundary_column_compression = BoundaryColumnCompression(AL1s, AL2s, hs, \
                                                                                    Lhs[i-1], Cs, "left", \
                                                                                    chi_max_b)
                        Lhs[i] = boundary_column_compression.run()
                        trunc_errors_Lhs[i] = boundary_column_compression.trunc_errors
                    elif not combine_hs:
                        boundary_column_compression_h = BoundaryColumnCompression(AL1s, AL2s, hs, \
                                                                                  None, Cs, "left", \
                                                                                  chi_max_b)
                        Lhs[i] = boundary_column_compression_h.run()
                        trunc_errors_Lhs[i] = boundary_column_compression_h.trunc_errors
                        if n > 1:
                            for j in range(i):
                                boundary_column_compression_t = BoundaryColumnCompression(None, AL2s, None, \
                                                                                          Lhs[j], Cs, "left", \
                                                                                          chi_max_b)
                                Lhs[j] = boundary_column_compression_t.run()
                                trunc_errors_Lhs[j] = boundary_column_compression_t.trunc_errors
                elif n%2 == 0:
                    if combine_hs:
                        boundary_column_compression = BoundaryColumnCompression(get_flipped_As(AL1s), \
                                                                                get_flipped_As(AL2s), \
                                                                                get_flipped_hs(hs), \
                                                                                get_flipped_mps(Lhs[i-1]), \
                                                                                get_flipped_Cs(Cs), \
                                                                                "left", \
                                                                                chi_max_b)
                        Lhs[i] = get_flipped_mps(boundary_column_compression.run())
                        trunc_errors_Lhs[i] = boundary_column_compression.trunc_errors[::-1]
                    elif not combine_hs:
                        boundary_column_compression_h = BoundaryColumnCompression(get_flipped_As(AL1s), \
                                                                                  get_flipped_As(AL2s), \
                                                                                  get_flipped_hs(hs), \
                                                                                  None, \
                                                                                  get_flipped_Cs(Cs), \
                                                                                  "left", \
                                                                                  chi_max_b)
                        Lhs[i] = get_flipped_mps(boundary_column_compression_h.run())
                        trunc_errors_Lhs[i] = boundary_column_compression_h.trunc_errors[::-1]
                        for j in range(i):
                            boundary_column_compression_t = BoundaryColumnCompression(None, \
                                                                                      get_flipped_As(AL2s), \
                                                                                      None, \
                                                                                      get_flipped_mps(Lhs[j]), \
                                                                                      get_flipped_Cs(Cs), \
                                                                                      "left", \
                                                                                      chi_max_b)
                            Lhs[j] = get_flipped_mps(boundary_column_compression_t.run())
                            trunc_errors_Lhs[j] = boundary_column_compression_t.trunc_errors[::-1]
        # right boundaries
        Rhs = None
        trunc_errors_Rhs = None
        if NR > 0:
            Rhs = [None] * NR
            trunc_errors_Rhs = [None] * NR
            for i in range(NR):
                n = N - i 
                iso_peps_copy = iso_peps.copy()
                iso_peps_copy.move_orthogonality_column_to(n-1, min_dims=True)
                AR1s = iso_peps_copy.get_ARs(n)
                AR2s = iso_peps_copy.get_ARs(n-1)
                Cs = iso_peps_copy.get_Cs(n-1)
                hs = h_mpos[n-1]
                if n%2 == 1:
                    if combine_hs:
                        if n == N:
                            boundary_column_compression = BoundaryColumnCompression(get_flipped_As(AR1s), \
                                                                                    get_flipped_As(AR2s), \
                                                                                    get_flipped_hs(hs), \
                                                                                    None, \
                                                                                    get_flipped_Cs(Cs), \
                                                                                    "right", \
                                                                                    chi_max_b)
                        elif n < N:
                            boundary_column_compression = BoundaryColumnCompression(get_flipped_As(AR1s), \
                                                                                    get_flipped_As(AR2s), \
                                                                                    get_flipped_hs(hs), \
                                                                                    get_flipped_mps(Rhs[i-1]), \
                                                                                    get_flipped_Cs(Cs), \
                                                                                    "right", \
                                                                                    chi_max_b)
                        Rhs[i] = get_flipped_mps(boundary_column_compression.run())  
                        trunc_errors_Rhs[i] = boundary_column_compression.trunc_errors[::-1]
                    elif not combine_hs:
                        boundary_column_compression_h = BoundaryColumnCompression(get_flipped_As(AR1s), \
                                                                                  get_flipped_As(AR2s), \
                                                                                  get_flipped_hs(hs), \
                                                                                  None, \
                                                                                  get_flipped_Cs(Cs), \
                                                                                  "right", \
                                                                                  chi_max_b)
                        Rhs[i] = get_flipped_mps(boundary_column_compression_h.run())
                        trunc_errors_Rhs[i] = boundary_column_compression_h.trunc_errors[::-1]
                        if n < N:
                            for j in range(i):
                                boundary_column_compression_t = BoundaryColumnCompression(None, \
                                                                                          get_flipped_As(AR2s), \
                                                                                          None, \
                                                                                          get_flipped_mps(Rhs[j]), \
                                                                                          get_flipped_Cs(Cs), \
                                                                                          "right", \
                                                                                          chi_max_b)
                                Rhs[j] = get_flipped_mps(boundary_column_compression_t.run())
                                trunc_errors_Rhs[j] = boundary_column_compression_t.trunc_errors[::-1]
                elif n%2 == 0:
                    if combine_hs:
                        boundary_column_compression = BoundaryColumnCompression(AR1s, AR2s, hs, \
                                                                                Rhs[i-1], Cs, "right", \
                                                                                chi_max_b)
                        Rhs[i] = boundary_column_compression.run()
                        trunc_errors_Rhs[i] = boundary_column_compression.trunc_errors
                    elif not combine_hs:
                        boundary_column_compression_h = BoundaryColumnCompression(AR1s, AR2s, hs, \
                                                                                  None, Cs, "right", \
                                                                                  chi_max_b)
                        Rhs[i] = boundary_column_compression_h.run()
                        trunc_errors_Rhs[i] = boundary_column_compression_h.trunc_errors    
                        for j in range(i):           
                            boundary_column_compression_t = BoundaryColumnCompression(None, AR2s, None, \
                                                                                      Rhs[j], Cs, "right", \
                                                                                      chi_max_b)
                            Rhs[j] = boundary_column_compression_t.run()
                            trunc_errors_Rhs[j] = boundary_column_compression_t.trunc_errors
    return Lhs, Rhs, trunc_errors_Lhs, trunc_errors_Rhs