"""Toy code implementing variational quasiparticle excitations on top of an MPS ground state."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import null_space
from functools import reduce

from .a_mps import MPS


class VariationalQuasiparticleExcitationEngine:
    """Simple class for variationally finding quasiparticle excitations on top of an MPS ground 
    state.
    
    ansatz: |phi(X;A)> = sum_n (AL[1])-- ... --(AL[n-1])--(VL[n])-(X[n])--(AR[n+1])-- ... --(AR[N]),
                                  |                |         |                |                |

    where X[n] is the left-gauge parametrization of the tensor --(B[n])-- = --(VL[n])-(X[n])--, 
                                                                   |             | 
    perturbing the ground state |psi(A)> around site n.

    With <phi(X*;A)|H|phi(X;A)> = <X*|H_eff|X>, the variational optimization boils down to 
    diagonalizing H_eff for the lowest-lying eigenvalues.

    For a two-fold degnerate, symmetry-broken ground state, topological domain wall excitations can
    be targeted by taking AL and AR from the two orthogonal ground states (if H has OBC).

    If H has PBC, we can introduce an effective translation operator <X*|T_eff|X> = 
    <phi(X*;A)|T|phi(X;A)>, and diagonalize H_eff - alpha (e^{-ip}T_eff + e^{ip}T_eff^{dagger})
    (where alpha is large positive number) for the lowest-lying real eigenvalues to obtain excited 
    states with definite momentum p. 

    Parameters
    ----------
    mps0: MPS
          The ground state on top of which the excited states are searched.
    mpo, p: Same as attributes.
    mps0_tilde: MPS or None
                If not None, second degnerate, symmetry-broken ground state (if mpo has OBC).

    Attributes
    ----------
    mpo/Ws: list of np.array[ndim=4]s
            The matrix product operator of which the excitations are searched.
    p: int or None
       If not None, momentum targeted with effective translation operator (if mpo has PBC).
    ALs: list of np.array[ndim=3]s
         Left orthonormal tensors of the ground state.
    ARs: np.array[ndim=3]
         Right orthonormal tensors of the (second degenerate) ground state.
    VLs: list of np.array[ndim=3]s
         Left-gauge tensors with legs of dimension D(n-1) x d x D(n-1)*d-Dn.
         If D(n-1)*d-Dn <= 0, the corresponding VL is None.
    shape_Xs: list of tuple of ints
              Shapes (D(n-1)*d-Dn, Dn) of the left-gauge parametrizations Xs.
              If D(n-1)*d-Dn <= 0, the corresponding shape is None.
    shape_vecX: int
                Length of the vector vecX containing all Xs that are not None. 
    N: int
       Length of the chain.
    E0: float
        Ground state energy of mps0.
    """
    def __init__(self, mps0, mpo, p=None, mps0_tilde=None):
        self.Ws = mpo
        self.p = p
        self.ALs, self.ARs, _ = MPS.to_canonical_form(mps0.ARs)
        if mps0_tilde is not None:
            self.ARs = mps0_tilde.ARs
        self.VLs = get_VLs(self.ALs)
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(self.ALs)
        self.N = len(self.ALs)
        self.E0 = mps0.get_mpo_expectation_value(mpo)

    def run(self, k):
        """Compute k excitations of self.Ws and return them as ExcitedMPS instances."""
        if self.p is not None:
            alpha = 2 * np.abs(self.E0)
            t = np.exp(-1.j * 2 * np.pi * self.p / self.N)
            H_eff = Heff(self.Ws, self.ALs, self.ARs)
            T_eff = Teff(self.ALs, self.ARs)
            T_eff_adjoint = TeffAdjoint(self.ALs, self.ARs)
            H_T_eff = H_eff - alpha * (t * T_eff + np.conj(t) * T_eff_adjoint)
            _, vecXs = eigsh(H_T_eff, k, which="SA", tol=1.e-5)
            empss = [ExcitedMPS(self.ALs, self.ARs, vecXs[:, i]) for i in range(k)]
            es = [emps.get_mpo_expectation_value(self.Ws) - self.E0 for emps in empss]
            if k == 1:
                return es[0], empss[0]
            return es, empss
        else:
            H_eff = Heff(self.Ws, self.ALs, self.ARs)
            Es, vecXs = eigsh(H_eff, k, which="SA")
            empss = [ExcitedMPS(self.ALs, self.ARs, vecXs[:, i]) for i in range(k)]
            es = [E - self.E0 for E in Es]
            for i in range(k):
                print(f"Excitation {i+1}:")
                print(f"Optimized excited MPS from Hamiltonian.")
                empss[i].print_all_excitation_norms()
                print(f"=> E_{i+1} = {Es[i]} (e_{i+1} = {es[i]}).")
            if k == 1:
                return es[0], empss[0]
            return es, empss


def get_VLs(ALs):
    """For left orthonormal tensor AL[n], compute tensor VL[n] of dimension D(n-1) x d x D(n-1)*d-Dn, 
    such that

    .--(VL[n])--vvRn
    |     |                                  
    |     |          =  0   <->   vRn*--(AL[n]^{dagger})----(VL[n])--vvRn  =  0.
    |     |                                    |               |          
    .-(AL[n]*)--vRn*                           .---------------.

    Interpreting AL[n] as the first Dn orthonormal columns of a D(n-1)*d x D(n-1)*d unitary matrix, 
    VL[n] corresponds to the remaining D(n-1)*d-Dn columns thereof.

    If D(n-1)*d-Dn <= 0, set the corresponding VL[n] to None.
    """
    N = len(ALs)
    VLs = [None] * N  
    for n in range(N):
        Dnm1, d, Dn = np.shape(ALs[n])  # vL(n-1) pn vRn
        if Dnm1 * d - Dn > 0:
            AL = np.reshape(ALs[n], (Dnm1 * d, Dn))  # vL(n-1).pn vRn
            VL = null_space(np.conj(AL).T)  # vL(n-1).pn vvRn
            VL = np.reshape(VL, (Dnm1, d, Dnm1 * d - Dn))  # vL(n-1) pn vvRn
            VLs[n] = VL
    return VLs

def Xs_to_Bs(Xs, VLs):
    """For X[n] the left-gauge parametrization, compute the perturbation tensor B[n] given by
        
    --(B[n])--  =  --(VL[n])--(X[n])--.   
        |               |
    """
    N = len(Xs)
    Bs = [None] * N
    for n in range(N):
        if VLs[n] is not None:
            Bs[n] = np.tensordot(VLs[n], Xs[n], axes=(2, 0))  
            # vL(n-1) pn [vvRn], [vvLn] vRn -> vL(n-1) pn vRn
    return Bs

def Bs_to_Xs(Bs, VLs):
    """For B[n] the perturbation tensor, compute the left-gauge parametrization given by
    
                   .---(B[n])---  
                   |     |
    --(X[n])--  =  |     |       .
                   |     |
                   .--(VL[n]*)--
    """
    N = len(Bs)
    Xs = [None] * N
    for n in range(N):
        if Bs[n] is not None:
            Xs[n] = np.tensordot(np.conj(VLs[n]), Bs[n], axes=((0, 1), (0, 1)))  
            # [vL(n-1)*] [p*] vvRn*, [vL(n-1)] [p] vRn -> vvLn vRn
    return Xs

def get_shape_Xs_vecX(ALs):
    """For left orthonormal tensor AL[n], compute the shape (D(n-1)*d-Dn, Dn) of the left-gauge
    parametrization X[n]. If D(n-1)*d-Dn <= 0, set the corresponding shape to None. Also return the 
    length of the vector vecX containing all X[n]s that are not None."""
    N = len(ALs)
    shape_Xs = [None] * N
    shape_vecX = 0
    for n in range(N):
        Dnm1, d, Dn = np.shape(ALs[n])  # vL(n-1) p vRn
        if Dnm1 * d - Dn > 0:
            shape_Xs[n] = (Dnm1 * d - Dn, Dn)
            shape_vecX += (Dnm1 * d - Dn) * Dn
    return shape_Xs, shape_vecX

def vec_to_tensors(vecX, shape_Xs):
    """Reshape a vector vecX into tensors of shapes shape_Xs."""
    N = len(shape_Xs)
    Xs = [None] * N
    vec_ind = 0
    for n in range(N):
        if shape_Xs[n] is not None:
            X = vecX[vec_ind : vec_ind + shape_Xs[n][0] * shape_Xs[n][1]]
            X = np.reshape(X, shape_Xs[n])
            Xs[n] = X
            vec_ind += shape_Xs[n][0] * shape_Xs[n][1]
    assert(vec_ind == len(vecX))
    return Xs  

def tensors_to_vec(Xs, shape_vecX):
    """Reshape all tensors in Xs into one vector of length shape_vecX."""
    N = len(Xs)
    vecX = np.zeros(shape_vecX, dtype=complex)
    vec_ind = 0
    for n in range(N):
        if Xs[n] is not None:
            X = Xs[n].flatten()
            vecX[vec_ind : vec_ind + len(X)] = X
            vec_ind += len(X)
    assert(vec_ind == shape_vecX)
    return vecX


class Heff(LinearOperator):
    """Class for the effective Hamiltonian acting on the parametrization X of the perturbation B.
                        
                                                         .---(B[n])---  
                                                         |     |
    --(B[n])--  =  --(VL[n])--(X[n])--,   --(X[n])--  =  |     |   
        |               |                                |     |
                                                         .--(VL[n]*)--

                                          .------(B[n])------.      
                                          |        |         |   
            (if B[n] is not None)         |        |         |    
    matvec:      --(B[n])--        ->  (L[n-1])--(W[n])--(R[n+1])  
                     |                    |        |         |    
                                          |                  |
                                          .------      ------.

                                          (if LB[n-1] is not None)      (if RB[n+1] is not None)
                                             .------(AR[n])-----.         .------(AL[n])-----.
                                             |         |        |         |         |        |
                                             |         |        |         |         |        |
                                       + (LB[n-1])--(W[n])--(R[n+1]) + (L[n-1])--(W[n])--(RB[n+1])
                                             |         |        |         |         |        |
                                             |                  |         |                  |
                                             .-------      -----.         .------       -----.
    """         
    def __init__(self, Ws, ALs, ARs):
        self.Ws = Ws  # [W[1], ..., W[N]]
        self.VLs = get_VLs(ALs)  # [VL[1], ..., VL[N]] (VL[n] is None if D(n-1)*d-Dn <= 0)
        self.ALs = ALs  # [AL[1], ..., AL[N]]
        self.ARs = ARs  # [AR[1], ..., AR[N]]
        self.N = len(self.Ws)
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(ALs)  
        self.shape = (self.shape_vecX, self.shape_vecX)
        self.dtype = reduce(np.promote_types, [Ws[0].dtype, ALs[0].dtype, ARs[0].dtype])
    
    def get_Ls(self):
        """Compute the left environments of Heff not containing any B tensor.

          .-----     .--                        .-----        .------(AL[n])---
          |          |                          |             |         |
          |          |                          |             |         |
        (L[0])--  =  .--,   n = 1, ..., N-1:  (L[n])--  =  (L[n-1])--(W[n])----
          |          |                          |             |         |
          |          |                          |             |         |
          .-----     .--                        .-----        .------(AL[n]*)--
        """
        ALs = self.ALs
        Ws = self.Ws
        N = self.N
        Ls = [None] * N  
        Ls[0] = np.ones((1, 1, 1))  # L[0]: vR0 wR0 vR0*
        for n in range(N-1):  # sites 1, ..., N-1
            L = np.tensordot(Ls[n], ALs[n], axes=(0, 0))  
            # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
            L = np.tensordot(L, Ws[n], axes=((0, 2), (0, 3)))  
            # [wR(n-1)] vR(n-1)* [pn] vRn, [wLnm1] wRn pn [pn*]
            L = np.tensordot(L, np.conj(ALs[n]), axes=((0, 3), (0, 1)))  
            # [vR(n-1)*] vRn wRn [pn], [vLnm1*] [pn*] vRn*
            Ls[n+1] = L  # vRn wRn vRn*
        return Ls  # [L[0], ..., L[N-1]]
    
    def get_Rs(self):
        """Compute the right environments of Heff not containing any B tensor.

        ------.        --.                    -----.       ---(AR[n])------.             
              |          |                         |             |         |
              |          |                         |             |         |
        --(R[N+1])  =  --.,   n = N, ..., 2:  --(R[n])  =  ----(W[n])--(R[n+1])   
              |          |                         |             |         |
              |          |                         |             |         |
        ------.        --.                    -----.       --(AR[n]*)------.
        """
        ARs = self.ARs
        Ws = self.Ws
        N = self.N
        Rs = [None] * N  
        Rs[N-1] = np.ones((1, 1, 1))  # R[N+1]: vLN wLN vLN*
        for n in reversed(range(1, N)):  # sites N, ...., 2
            R = np.tensordot(ARs[n], Rs[n], axes=(2, 0)) 
            # vL(n-1) pn [vRn], [vLn] wLn vLn*
            R = np.tensordot(R, Ws[n], axes=((1, 2), (3, 1)))  
            # vL(n-1) [pn] [wLn] vLn*, wL(n-1) [wRn] pn [pn*]
            R = np.tensordot(R, np.conj(ARs[n]), axes=((3, 1), (1, 2)))  
            # vL(n-1) [vLn*] wL(n-1) [pn], vL(n-1)* [pn*] [vRn*]
            Rs[n-1] = R  # vL(n-1) wL(n-1) vL(n-1)*
        return Rs  # [R[2], ..., R[N+1]]
    
    def get_LBs(self, Ls, Bs):
        """Compute the environments of Heff containing the B tensors on the left.

                                                              (if B[n] is not None)   
           .-----                                  .-----        .------(B[n])----  
           |                                       |             |         |        
           |                                       |             |         |        
        (LB[0])--  =  None,   n = 1, ..., N-1:  (LB[n])--  =  (L[n-1])--(W[n])----  
           |                                       |             |         |          
           |                                       |             |         |               
           .-----                                  .-----        .------(AL[n]*)--         

                                                                (if LB[n-1] is not None)
                                                                     .------(AR[n])--- 
                                                                     |         |
                                                                     |         |
                                                              +  (LB[n-1])--(W[n])---- 
                                                                     |         |
                                                                     |         |
                                                                     .------(AL[n]*)-- 
        """
        ALs = self.ALs
        ARs = self.ARs
        Ws = self.Ws
        N = self.N
        LBs = [None] * N
        for n in range(N-1):  # sites 1, ..., N-1
            if Bs[n] is not None:
                LB = np.tensordot(Ls[n], Bs[n], axes=(0, 0))  
                # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
                LB = np.tensordot(LB, Ws[n], axes=((0, 2), (0, 3)))  
                # [wR(n-1)] vR(n-1)* [pn] vRn, [wL(n-1)] wRn pn [pn*]
                LB = np.tensordot(LB, np.conj(ALs[n]), axes=((0, 3), (0, 1)))  
                # [vR(n-1)*] vRn wRn [pn], [vL(n-1)*] [pn*] vRn*
                LBs[n+1] = LB  # vRn wRn vRn*
            if LBs[n] is not None:
                LB = np.tensordot(LBs[n], ARs[n], axes=(0, 0))  
                # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
                LB = np.tensordot(LB, Ws[n], axes=((0, 2), (0, 3)))  
                # [wR(n-1)] vR(n-1)* [pn] vRn, [wLnm1] wRn pn [pn*]
                LB = np.tensordot(LB, np.conj(ALs[n]), axes=((0, 3), (0, 1)))  
                # [vR(n-1)*] vRn wRn [pn], [vLnm1*] [pn*] vRn* -> vRn wRn vRn*
                if LBs[n+1] is not None:
                    LBs[n+1] += LB  
                else:
                    LBs[n+1] = LB
        return LBs  # [None, LB[1], ..., LB[N-1]]
    
    def get_RBs(self, Rs, Bs):
        """Compute the environments of Heff containing the B tensors on the right.

                                                             (if B[n] is not None)  
        ------.                                  -----.       ----(B[n])------.         
              |                                       |             |         |     
              |                                       |             |         |              
        --(RB[N+1])  =  None,   n = N, ..., 2:  --(RB[n])  =  ----(W[n])--(R[n+1])     
              |                                       |             |         |              
              |                                       |             |         |              
        ------.                                  -----.       --(AR[n]*)------.        

                                                                (if RB[n+1] is not None)  
                                                                 ---(AL[n])------.
                                                                       |         |
                                                                       |         |
                                                              +  ----(W[n])--(RB[n+1])   
                                                                       |         |
                                                                       |         |
                                                                 --(AR[n]*)------.
        """
        ALs = self.ALs
        ARs = self.ARs
        Ws = self.Ws
        N = self.N
        RBs = [None] * N
        for n in reversed(range(1, N)):  # sites N, ..., 2
            if Bs[n] is not None:
                RB = np.tensordot(Bs[n], Rs[n], axes=(2, 0))  
                # vL(n-1) pn [vRn], [vLn] wLn vLn*
                RB = np.tensordot(RB, Ws[n], axes=((1, 2), (3, 1)))  
                # vL(n-1) [pn] [wLn] vLn*, wL(n-1) [wRn] pn [pn*]
                RB = np.tensordot(RB, np.conj(ARs[n]), axes=((3, 1), (1, 2)))  
                # vL(n-1) [vLn*] wL(n-1) [pn], vL(n-1)* [pn*] [vRn*]
                RBs[n-1] = RB  # vL(n-1) wL(n-1) vL(n-1)*
            if RBs[n] is not None:
                RB = np.tensordot(ALs[n], RBs[n], axes=(2, 0))  
                # vL(n-1) pn [vRn], [vLn] wLn vLn*
                RB = np.tensordot(RB, Ws[n], axes=((1, 2), (3, 1)))  
                # vL(n-1) [pn] [wLn] vLn*, wL(n-1) [wRn] pn [p*]
                RB = np.tensordot(RB, np.conj(ARs[n]), axes=((3, 1), (1, 2)))  
                # vL(n-1) [vLn*] wL(n-1) [pn], vL(n-1)* [pn*] [vRn*] -> vL(n-1) wL(n-1) vL(n-1)*
                if RBs[n-1] is not None:
                    RBs[n-1] += RB  
                else:
                    RBs[n-1] = RB  
        return RBs  # [RB[2], ..., RB[N], None]

    def _matvec(self, vecX):
        Ws = self.Ws  # [W[1], ..., W[N]]
        ALs = self.ALs  # [AL[1], ..., AL[N]]
        ARs = self.ARs  # [AR[1], ..., AR[N]]
        Ls = self.get_Ls()  # [L[0], ..., L[N-1]]
        Rs = self.get_Rs()  # [R[2], ..., R[N+1]]
        Xs = vec_to_tensors(vecX, self.shape_Xs)  # [X[1], ..., X[N]]
        Bs = Xs_to_Bs(Xs, self.VLs)  # [B[1], ..., B[N]]
        LBs = self.get_LBs(Ls, Bs)  # [None, LB[1], ..., LB[N-1]]
        RBs = self.get_RBs(Rs, Bs)  # [RB[2], ..., RB[N], None]
        N = self.N
        Bs_new = [None] * N
        for n in range(N):  # sites 1, ..., N
            if Bs[n] is not None:
                B_new_1 = np.tensordot(Ls[n], Bs[n], axes=(0, 0))  
                # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
                B_new_1 = np.tensordot(B_new_1, Ws[n], axes=((0, 2), (0, 3)))  
                # [wR(n-1)] vR(n-1)* [pn] vRn, [wL(n-1)] wRn pn [pn*]
                B_new_1 = np.tensordot(B_new_1, Rs[n], axes=((1, 2), (0, 1)))  
                # vR(n-1)* [vRn] [wRn] pn, [vLn] [wLn] vLn* -> vL(n-1) pn vRn
                Bs_new[n] = B_new_1
                if LBs[n] is not None:
                    B_new_2 = np.tensordot(LBs[n], ARs[n], axes=(0, 0))  
                    # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
                    B_new_2 = np.tensordot(B_new_2, Ws[n], axes=((0, 2), (0, 3)))  
                    # [wR(n-1)] vR(n-1)* [pn] vRn, [wL(n-1)] wRn pn [pn*]
                    B_new_2 = np.tensordot(B_new_2, Rs[n], axes=((1, 2), (0, 1)))  
                    # vR(n-1)* [vRn] [wRn] pn, [vLn] [wLn] vLn* -> vL(n-1) pn vRn
                    Bs_new[n] += B_new_2  
                if RBs[n] is not None:
                    B_new_3 = np.tensordot(Ls[n], ALs[n], axes=(0, 0))  
                    # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
                    B_new_3 = np.tensordot(B_new_3, Ws[n], axes=((0, 2), (0, 3)))  
                    # [wR(n-1)] vR(n-1)* [pn] vRn, [wL(n-1)] wRn pn [pn*]
                    B_new_3 = np.tensordot(B_new_3, RBs[n], axes=((1, 2), (0, 1)))  
                    # vR(n-1)* [vRn] [wRn] pn, [vLn] [wLn] vLn* -> vL(n-1) pn vRn
                    Bs_new[n] += B_new_3 
        Xs_new = Bs_to_Xs(Bs_new, self.VLs)
        vecX_new = tensors_to_vec(Xs_new, self.shape_vecX)
        return vecX_new 
    

class Teff(LinearOperator):
    """Class for the effective translation operator acting on the parametrization X of the 
    perturbation B.

                                                         .---(B[n])---  
                                                         |     |
    --(B[n])--  =  --(VL[n])--(X[n])--,   --(X[n])--  =  |     |   
        |               |                                |     |
                                                         .--(VL[n]*)--                  
                                             
            (if B[n] is not None)             
    matvec:      --(B[n])--        -> 
                     |   
                                   
                                            (if B[m] is not None)
                      (AL[1])--- ... ---(AL[m-1])---(B[m])---(AR[m+1])--- ... ---(AR[N-1])---(AR[N])
                         |                  |         |          |                   |         /         
    sum_{m}                                                                            
                /        |                  |         |          |                   |              
            (AL[1]*)--(AL[2]*)-- ... ---(AL[n-1]*)--      ---(AR[n+1]*)-- ... ---(AR[N]*)
    """   
    def __init__(self, ALs, ARs):
        self.VLs = get_VLs(ALs)  # [VL[1], ..., VL[N]] (VL[n] is None if D(n-1)*d-Dn <= 0)
        self.ALs = ALs  # [AL[1], ..., AL[N]]
        self.ARs = ARs  # [AR[1], ..., AR[N]]
        self.N = len(self.ALs)    
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(ALs)  
        self.shape = (self.shape_vecX, self.shape_vecX)
        self.dtype = reduce(np.promote_types, [ALs[0].dtype, ARs[0].dtype])

    def get_Ls(self):
        """Compute the left environments of Teff not containing any B tensor."""
        ALs = self.ALs
        N = self.N
        Ls = [None] * N
        Ls[1] = np.conj(ALs[0])  # L[1]: vR0 p1* vR1*
        for n in range(1, N-1):  # sites 2, ..., N-1
            L = np.tensordot(Ls[n], ALs[n-1], axes=(0, 0))  
            # [vR(n-2)] p1* vR(n-1)*, [vL(n-2)] p(n-1) vR(n-1)
            L = np.tensordot(L, np.conj(ALs[n]), axes=((1, 2), (0, 1)))
            # p1* [vR(n-1)*] [p(n-1)] vR(n-1), [vL(n-1)*] [pn*] vRn*
            L = np.transpose(L, (1, 0, 2))  # vR(n-1) p1* vRn*
            Ls[n+1] = L 
        return Ls  # [None, L[1], ..., L[N-1]]
    
    def get_Rs(self):
        """Compute the right environments of Teff not containing any B tensor."""
        ARs = self.ARs
        N = self.N
        Rs = [None] * N
        Rs[N-1] = ARs[N-1]  # R[N+1]: vL(N-1) pN vLN*
        for n in reversed(range(1, N)):  # sites N, ..., 2
            R = np.tensordot(ARs[n-1], Rs[n], axes=(2, 0))  
            # vL(n-2) p(n-1) [vR(n-1)], [vL(n-1)] pN vLn*
            R = np.tensordot(R, np.conj(ARs[n]), axes=((1, 3), (1, 2)))
            # vL(n-2) [p(n-1)] pN [vLn*], vL(n-1)* [pn*] [vRn*]
            Rs[n-1] = R  # vL(n-2) pN vL(n-1)*
        return Rs  # [R[2], ..., R[N+1]]

    def get_LBs(self, Ls, Bs):
        """Compute the environments of Teff containing the B tensors on the left."""
        ALs = self.ALs
        ARs = self.ARs
        N = self.N
        LBs = [None] * N
        for n in range(1, N-1):  # sites 2, ..., N-1
            if Bs[n-1] is not None:
                LB = np.tensordot(Ls[n], Bs[n-1], axes=(0, 0))
                # [vR(n-2)] p1* vR(n-1)*, [vL(n-2)] p(n-1) vR(n-1)
                LB = np.tensordot(LB, np.conj(ALs[n]), axes=((1, 2), (0, 1)))
                # p1* [vR(n-1)*] [p(n-1)] vR(n-1), [vL(n-1)*] [pn*] vRn*
                LB = np.transpose(LB, (1, 0, 2))  # vR(n-1) p1* vRn*
                LBs[n+1] = LB
            if LBs[n] is not None:
                LB = np.tensordot(LBs[n], ARs[n-1], axes=(0, 0))
                # [vR(n-2)] p1* vR(n-1)*, [vL(n-2)] p(n-1) vR(n-1)
                LB = np.tensordot(LB, np.conj(ALs[n]), axes=((1, 2), (0, 1)))
                # p1* [vR(n-1)*] [p(n-1)] vR(n-1), [vL(n-1)*] [pn*] vRn*
                LB = np.transpose(LB, (1, 0, 2))  # vR(n-1) p1* vRn*
                if LBs[n+1] is not None:
                    LBs[n+1] += LB
                else:
                    LBs[n+1] = LB
        return LBs  # [None, None, LB[2], ..., LB[N-1]]

    def get_RBs(self, Rs, Bs):
        """Compute the environments of Teff containing the B tensors on the right."""
        ALs = self.ALs
        ARs = self.ARs
        N = self.N
        RBs = [None] * N
        if Bs[N-1] is not None:
            RBs[N-1] = Bs[N-1]  # RB[N+1]: vL(N-1) pN vLN*
        for n in reversed(range(1, N)):  # sites N, ..., 2
            if Bs[n-1] is not None:
                RB = np.tensordot(Bs[n-1], Rs[n], axes=(2, 0))  
                # vL(n-2) p(n-1) [vR(n-1)], [vL(n-1)] pN vLn*
                RB = np.tensordot(RB, np.conj(ARs[n]), axes=((1, 3), (1, 2)))
                # vL(n-2) [p(n-1)] pN [vLn*], vL(n-1)* [pn*] [vRn*]
                RBs[n-1] = RB  # vL(n-2) pN vL(n-1)*
            if RBs[n] is not None:
                RB = np.tensordot(ALs[n-1], RBs[n], axes=(2, 0))  
                # vL(n-2) p(n-1) [vR(n-1)], [vL(n-1)] pN vLn*
                RB = np.tensordot(RB, np.conj(ARs[n]), axes=((1, 3), (1, 2)))
                # vL(n-2) [p(n-1)] pN [vLn*], vL(n-1)* [pn*] [vRn*] -> vL(n-2) pN vL(n-1)*
                if RBs[n-1] is not None:
                    RBs[n-1] += RB  
                else:
                    RBs[n-1] = RB
        return RBs  # [RB[2], ..., RB[N+2]]

    def _matvec(self, vecX):
        ALs = self.ALs  # [AL[1], ..., AL[N]]
        ARs = self.ARs  # [AR[1], ..., AR[N]]
        Ls = self.get_Ls()  # [None, L[1], ..., L[N-1]]
        Rs = self.get_Rs()  # [R[2], ..., R[N+1]]
        Xs = vec_to_tensors(vecX, self.shape_Xs)  # [X[1], ..., X[N]]
        Bs = Xs_to_Bs(Xs, self.VLs)  # [B[1], ..., B[N]]
        LBs = self.get_LBs(Ls, Bs)  # [None, None, LB[2], ..., LB[N-1]]
        RBs = self.get_RBs(Rs, Bs)  # [RB[2], ..., RB[N+2]]
        N = self.N
        Bs_new = [None] * N
        if Bs[0] is not None:
            if RBs[0] is not None:
                Bs_new[0] = RBs[0]  # B_new[1]: vL0 pN vR1 -> vL0 p1 vR1
        for n in range(1, N):  # sites 2, ..., N  
            if Bs[n] is not None:
                if Bs[n-1] is not None:
                    B_new_1 = np.tensordot(Ls[n], Bs[n-1], axes=(0, 0))
                    # [vR(n-2)] p1* vR(n-1)*, [vL(n-2)] p(n-1) vR(n-1)
                    B_new_1 = np.tensordot(B_new_1, Rs[n], axes=((3, 0), (0, 1)))
                    # [p1*] vR(n-1)* p(n-1) [vR(n-1)], [vL(n-1)] [pN] vLn* -> vL(n-1) pn vRn
                    Bs_new[n] = B_new_1
                if LBs[n] is not None:
                    B_new_2 = np.tensordot(LBs[n], ARs[n-1], axes=(0, 0))
                    # [vR(n-2)] p1* vR(n-1)*, [vL(n-2)] p(n-1) vR(n-1)
                    B_new_2 = np.tensordot(B_new_2, Rs[n], axes=((3, 0), (0, 1)))
                    # [p1*] vR(n-1)* p(n-1) [vR(n-1)], [vL(n-1)] [pN] vLn* -> vL(n-1) pn vRn
                    if Bs_new[n] is not None:
                        Bs_new[n] += B_new_2
                    else:
                        Bs_new[n] = B_new_2
                if RBs[n] is not None:
                    B_new_3 = np.tensordot(Ls[n], ALs[n-1], axes=(0, 0))
                    # [vR(n-2)] p1* vR(n-1)*, [vL(n-2)] p(n-1) vR(n-1)
                    B_new_3 = np.tensordot(B_new_3, RBs[n], axes=((3, 0), (0, 1)))
                    # [p1*] vR(n-1)* p(n-1) [vR(n-1)], [vL(n-1)] [pN] vLn* -> vL(n-1) pn vRn
                    if Bs_new[n] is not None:
                        Bs_new[n] += B_new_3
                    else:
                        Bs_new[n] = B_new_3
        Xs_new = Bs_to_Xs(Bs_new, self.VLs)
        vecX_new = tensors_to_vec(Xs_new, self.shape_vecX)
        return vecX_new


class TeffAdjoint(LinearOperator):
    """Class for the effective adjoint translation operator acting on the parametrization X of the 
    perturbation B.

                                                         .---(B[n])---  
                                                         |     |
    --(B[n])--  =  --(VL[n])--(X[n])--,   --(X[n])--  =  |     |   
        |               |                                |     |
                                                         .--(VL[n]*)--                  
                                             
            (if B[n] is not None)             
    matvec:      --(B[n])--        -> 
                     |   
                                   
                                            (if B[m] is not None)
            (AL[1])---(AL[2])--- ... ---(AL[m-1])---(B[m])---(AR[m+1])--- ... ---(AR[N])
               |_        |                  |         |          |                  |                 
    sum_{m}                                                                                   _
                         |                  |         |          |                  |          |  
                      (AL[1]*)-- ... ---(AL[n-1]*)--      ---(AR[n+1]*)-- ... --(AR[N-1]*)--(AR[N]*)
    """   
    def __init__(self, ALs, ARs):
        self.VLs = get_VLs(ALs)  # [VL[1], ..., VL[N]] (VL[n] is None if D(n-1)*d-Dn <= 0)
        self.ALs = ALs  # [AL[1], ..., AL[N]]
        self.ARs = ARs  # [AR[1], ..., AR[N]]
        self.N = len(self.ALs)    
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(ALs)  
        self.shape = (self.shape_vecX, self.shape_vecX)
        self.dtype = reduce(np.promote_types, [ALs[0].dtype, ARs[0].dtype]) 

    def get_Ls(self):
        """Compute the left environments of Teff not containing any B tensor."""
        ALs = self.ALs
        N = self.N
        Ls = [None] * N
        Ls[0] = np.transpose(ALs[0], (2, 1, 0))  # L[0]: vR1 p1 vR0*
        for n in range(N-1):  # sites 1, ..., N-1
            L = np.tensordot(Ls[n], ALs[n+1], axes=(0, 0))  
            # [vRn] p1 vR(n-1)*, [vLn] p(n+1) vR(n+1)
            L = np.tensordot(L, np.conj(ALs[n]), axes=((1, 2), (0, 1)))
            # p1 [vR(n-1)*] [p(n+1)] vR(n+1), [vL(n-1)*] [pn*] vRn*
            L = np.transpose(L, (1, 0, 2))  # vR(n+1) p1 vRn*
            Ls[n+1] = L 
        return Ls  # [L[0], ..., L[N-1]]
    
    def get_Rs(self):
        """Compute the right environments of Teff not containing any B tensor."""
        ARs = self.ARs
        N = self.N
        Rs = [None] * N
        Rs[N-2] = np.transpose(np.conj(ARs[N-1]), (2, 1, 0))  # R[N]: vLN pN* vL(N-1)*
        for n in reversed(range(1, N-1)):  # sites N-1, ..., 2
            R = np.tensordot(ARs[n+1], Rs[n], axes=(2, 0))  
            # vLn p(n+1) [vR(n+1)], [vL(n+1)] pN* vLn*
            R = np.tensordot(R, np.conj(ARs[n]), axes=((1, 3), (1, 2)))
            # vLn [p(n+1)] pN* [vLn*], vL(n-1)* [pn*] [vRn*]
            Rs[n-1] = R  # vLn pN* vL(n-1)*
        return Rs  # [R[2], ..., R[N], None]

    def get_LBs(self, Ls, Bs):
        """Compute the environments of Teff containing the B tensors on the left."""
        ALs = self.ALs
        ARs = self.ARs
        N = self.N
        LBs = [None] * N
        if Bs[0] is not None:
            LBs[0] = np.transpose(Bs[0], (2, 1, 0))  # LB[0]: vR1 p1 vR0*
        for n in range(N-1):  # sites 1, ..., N-1
            if Bs[n+1] is not None:
                LB = np.tensordot(Ls[n], Bs[n+1], axes=(0, 0))
                # [vRn] p1 vR(n-1)*, [vLn] p(n+1) vR(n+1)
                LB = np.tensordot(LB, np.conj(ALs[n]), axes=((1, 2), (0, 1)))
                # p1 [vR(n-1)*] [p(n+1)] vR(n+1), [vL(n-1)*] [pn*] vRn*
                LB = np.transpose(LB, (1, 0, 2))  # vR(n+1) p1 vRn*
                LBs[n+1] = LB
            if LBs[n] is not None:
                LB = np.tensordot(LBs[n], ARs[n+1], axes=(0, 0))
                # [vRn] p1 vR(n-1)*, [vLn] p(n+1) vR(n+1)
                LB = np.tensordot(LB, np.conj(ALs[n]), axes=((1, 2), (0, 1)))
                # p1 [vR(n-1)*] [p(n+1)] vR(n+1), [vL(n-1)*] [pn*] vRn*
                LB = np.transpose(LB, (1, 0, 2))  # vR(n+1) p1 vRn*
                if LBs[n+1] is not None:
                    LBs[n+1] += LB
                else:
                    LBs[n+1] = LB
        return LBs  # [LB[0], ..., LB[N-1]]

    def get_RBs(self, Rs, Bs):
        """Compute the environments of Teff containing the B tensors on the right."""
        ALs = self.ALs
        ARs = self.ARs
        N = self.N
        RBs = [None] * N
        for n in reversed(range(1, N-1)):  # sites N-1, ..., 2
            if Bs[n+1] is not None:
                RB = np.tensordot(Bs[n+1], Rs[n], axes=(2, 0))  
                # vLn p(n+1) [vR(n+1)], [vL(n+1)] pN* vLn*
                RB = np.tensordot(RB, np.conj(ARs[n]), axes=((1, 3), (1, 2)))
                # vLn [p(n+1)] pN* [vLn*], vL(n-1)* [pn*] [vRn*]
                RBs[n-1] = RB  # vLn pN* vL(n-1)*
            if RBs[n] is not None:
                RB = np.tensordot(ALs[n+1], RBs[n], axes=(2, 0))  
                # vLn p(n+1) [vR(n+1)], [vL(n+1)] pN* vLn*
                RB = np.tensordot(RB, np.conj(ARs[n]), axes=((1, 3), (1, 2)))
                # vLn [p(n+1)] pN* [vLn*], vL(n-1)* [pn*] [vRn*] -> # vLn pN* vL(n-1)*
                if RBs[n-1] is not None:
                    RBs[n-1] += RB  
                else:
                    RBs[n-1] = RB
        return RBs  # [RB[2], ..., RB[N-1], None, None]

    def _matvec(self, vecX):
        ALs = self.ALs  # [AL[1], ..., AL[N]]
        ARs = self.ARs  # [AR[1], ..., AR[N]]
        Ls = self.get_Ls()  # [L[0], ..., L[N-1]]
        Rs = self.get_Rs()  # [R[2], ..., R[N], None]
        Xs = vec_to_tensors(vecX, self.shape_Xs)  # [X[1], ..., X[N]]
        Bs = Xs_to_Bs(Xs, self.VLs)  # [B[1], ..., B[N]]
        LBs = self.get_LBs(Ls, Bs)  # [LB[0], ..., LB[N-1]]
        RBs = self.get_RBs(Rs, Bs)  # [RB[2], ..., RB[N-1], None, None]
        N = self.N
        Bs_new = [None] * N
        for n in range(N-1):  # sites 1, ..., N-1  
            if Bs[n] is not None:
                if Bs[n+1] is not None:
                    B_new_1 = np.tensordot(Ls[n], Bs[n+1], axes=(0, 0))
                    # [vRn] p1 vR(n-1)*, [vLn] p(n+1) vR(n+1)
                    B_new_1 = np.tensordot(B_new_1, Rs[n], axes=((3, 0), (0, 1)))
                    # [p1] vR(n-1)* p(n+1) [vR(n+1)], [vL(n+1)] [pN*] vLn* -> vL(n-1) pn vRn
                    Bs_new[n] = B_new_1
                if LBs[n] is not None:
                    B_new_2 = np.tensordot(LBs[n], ARs[n+1], axes=(0, 0))
                    # [vRn] p1 vR(n-1)*, [vLn] p(n+1) vR(n+1)
                    B_new_2 = np.tensordot(B_new_2, Rs[n], axes=((3, 0), (0, 1)))
                    # [p1] vR(n-1)* p(n+1) [vR(n+1)], [vL(n+1)] [pN*] vLn* -> vL(n-1) pn vRn
                    if Bs_new[n] is not None:
                        Bs_new[n] += B_new_2
                    else:
                        Bs_new[n] = B_new_2
                if RBs[n] is not None:
                    B_new_3 = np.tensordot(Ls[n], ALs[n+1], axes=(0, 0))
                    # [vRn] p1 vR(n-1)*, [vLn] p(n+1) vR(n+1)
                    B_new_3 = np.tensordot(B_new_3, RBs[n], axes=((3, 0), (0, 1)))
                    # [p1] vR(n-1)* p(n+1) [vR(n+1)], [vL(n+1)] [pN*] vLn* -> vL(n-1) pn vRn
                    if Bs_new[n] is not None:
                        Bs_new[n] += B_new_3
                    else:
                        Bs_new[n] = B_new_3
        if Bs[N-1] is not None:
            if LBs[N-1] is not None:
                Bs_new[N-1] = np.transpose(LBs[N-1], (2, 1, 0))  # B_new[N]: vL(N-1) pN vRN
        Xs_new = Bs_to_Xs(Bs_new, self.VLs)
        vecX_new = tensors_to_vec(Xs_new, self.shape_vecX)
        return vecX_new
    

class ExcitedMPS:
    """Simple class for a finite excited MPS.

    phi(X;A)> = sum_n (AL[1])-- ... --(AL[n-1])--(VL[n])-(X[n])--(AR[n+1])-- ... --(AR[N])
                         |                |         |                |                |
    """
    def __init__(self, ALs, ARs, vecX):
        self.ALs = ALs
        self.ARs = ARs 
        self.shape_Xs, self.shape_vecX = get_shape_Xs_vecX(ALs)
        self.vecX = vecX
        self.Xs = vec_to_tensors(vecX, self.shape_Xs)
        self.VLs = get_VLs(self.ALs)
        self.Bs = Xs_to_Bs(self.Xs, self.VLs)
        self.N = len(self.ALs)
        self.d = np.shape(self.ALs[0])[1]

    def copy(self):
        return ExcitedMPS([AL.copy() for AL in self.ALs], \
                          [AR.copy() for AR in self.ARs], \
                          self.vecX.copy())

    def print_all_excitation_norms(self):
        for n in range(self.N):
            X = self.Xs[n]
            if X is not None:
                print(f"> {np.shape(X)} excitation parameters at site {n} with ||X_{n}||^2 = {np.linalg.norm(X)**2}.")
        print(f"-> {self.shape_vecX} excitation parameters with ||X||^2 = {np.linalg.norm(self.vecX)**2}.")
        return

    def get_all_single_canonical_configurations(self):
        As_list = [None] * self.N
        for nc in range(self.N):
            if self.Bs[nc] is not None:
                As = [None] * self.N
                for n in range(nc):
                    As[n] = self.ALs[n].copy()
                As[nc] = self.Bs[nc].copy()
                for n in range(nc+1, self.N):
                    As[n] = self.ARs[n].copy()
                As_list[nc] = As
        return As_list

    def get_mpo_expectation_value(self, Ws):
        mpo_eff = Heff(Ws, self.ALs, self.ARs)
        E = np.inner(np.conj(self.vecX), mpo_eff._matvec(self.vecX))
        return np.real_if_close(E)
    
    def get_translation_overlap(self, emps_tilde=None):
        T_eff = Teff(self.ALs, self.ARs)
        vecX_translated = T_eff._matvec(self.vecX)
        if emps_tilde is not None:
            T = np.inner(np.conj(emps_tilde.vecX), vecX_translated)
        else:
            T = np.inner(np.conj(self.vecX), vecX_translated)
        return T

    def get_bond_expectation_values(self, h_bonds):
        """Compute the expectation values of the two-site operators h_bonds.

                                (AL[1])--- ... ---(AL[m-1])---(B[m])---(AR[m+1])--- ... ---(AR[N])
                                   |                  |         |          |                  |
                                                                pm

                                                               pn*      p(n+1)*
                                   |                  |         |          |                  |
        E_bond[n]  =  sum_{m,l}    |       ...        |       (-------h------)      ...       |   , 
                                   |                  |         |          |                  |  
                                                                pn       p(n+1)
                                                    
                                                               pl*
                                   |                  |         |          |                  |              
                                (AL[1]*)-- ... ---(AL[l-1]*)--(B[l]*)--(AR[l+1]*)-- ... --(AR[N]*)

              .---(B[n])---     .--(AL[n])--                       
              |     |           |     |                   
        with  |     |        =  |     |       =  0 for all n.
              |     |           |     |
              .--(AL[n]*)--     .--(B[n]*)--                 
        """
        def get_lh(As, Bs, h):
            """For As = [A1, A2] and Bs = [B1, B2], compute the following tensor:

             .---     .--(A1)--(A2)---                   
             |        |   |     |                            
            (lh)   =  | (----h----)     
             |        |   |     |      
             .---     .-(B1*)-(B2*)--- 
            """
            A1, A2, B1, B2 = As[0], As[1], Bs[0], Bs[1]
            lh = np.tensordot(A1, A2, axes=(2, 0))  # vL0 p1 [vR1], [vL1] p2 vR2
            lh = np.tensordot(h, lh, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL0 [p1] [p2] vR2
            lh = np.tensordot(np.conj(B1), lh, axes=((0, 1), (2, 0)))  
            # [vL0*] [p1*] vR1*, [p1] p2 [vL0] vR2
            lh = np.tensordot(lh, np.conj(B2), axes=((0, 1), (0, 1)))  
            # [vR1*] [p2] vR2, [vL1*] [p2*] vR2* -> vR2 vR2*
            return lh
        def get_rh(As, Bs, h):
            """For As = [A1, A2] and Bs = [B1, B2], compute the following tensor:

            ---.      ---(A1)--(A2)--.
               |          |     |    |
             (Rh)  =    (----h----)  |
               |          |     |    |
            ---.      --(B1*)-(B2*)--.
            """
            A1, A2, B1, B2 = As[0], As[1], Bs[0], Bs[1]
            rh = np.tensordot(A1, A2, axes=(2, 0))  # vL0 p1 [vR1], [vL1] p2 vR2
            rh = np.tensordot(h, rh, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL0 [p1] [p2] vR2
            rh = np.tensordot(rh, np.conj(B2), axes=((1, 3), (1, 2)))  
            # p1 [p2] vL0 [vR2], vL1* [p2*] [vR2*]
            rh = np.tensordot(rh, np.conj(B1), axes=((0, 2), (1, 2)))  
            # [p1] vL0 [vL1*], vL0* [p1*] [vR1*] -> vL0 vL0*
            return rh
        ALs = self.ALs  # [AL[1], ..., AL[N]]
        ARs = self.ARs  # [AR[1], ..., AR[N]]
        Bs = self.Bs  # [B[1], ..., B[N]]
        N = self.N
        # LBB[n]: both Bs on the left of site n
        LBBs = [None] * (N-1)  #[None, LBB[1], ..., LBB[N-2]]
        for n in range(N-2):  # sites 1, ..., N-2
            if Bs[n] is not None:
                LBB = np.tensordot(Bs[n], np.conj(Bs[n]), axes=((0, 1), (0, 1)))  
                # [vL(n-1)] [pn] vRn, [vL(n-1)*] [pn*] vRn* -> vRn vRn*
                LBBs[n+1] = LBB
            if LBBs[n] is not None:
                LBB = np.tensordot(LBBs[n], ARs[n], axes=(0, 0))  
                # [vR(n-1)] vR(n-1)*, [vL(n-1)] pn vRn
                LBB = np.tensordot(LBB, np.conj(ARs[n]), axes=((0, 1), (0, 1)))  
                # [vR(n-1)*] [pn] vRn, [vL(n-1)*] [pn*] vRn* -> vRn vRn*
                if LBBs[n+1] is not None:
                    LBBs[n+1] += LBB
                else:
                    LBBs[n+1] = LBB
        # RB_bot[n]: bottom B on the right of site n
        RB_bots = [None] * (N-1)  # [RB_bot[3], ..., RB_bot[N], None]
        for n in reversed(range(2, N)):  # sites N, ..., 3
            if Bs[n] is not None:
                RB_bot = np.tensordot(ARs[n], np.conj(Bs[n]), axes=((1, 2), (1, 2)))  
                # vL(n-1) [pn] [vRn], vL(n-1)* [pn*] [vRn*]  -> vL(n-1) vL(n-1)*
                RB_bots[n-2] = RB_bot
            if RB_bots[n-1] is not None:
                RB_bot = np.tensordot(ARs[n], RB_bots[n-1], axes=(2, 0))  
                # vL(n-1) pn [vRn], [vLn] vLn*
                RB_bot = np.tensordot(RB_bot, np.conj(ALs[n]), axes=((1, 2), (1, 2)))  
                # vL(n-1) [pn] [vLn*], vL(n-1)* [pn*] [vRn*] -> vL(n-1) vL(n-1)*
                if RB_bots[n-2] is not None:
                    RB_bots[n-2] += RB_bot
                else:
                    RB_bots[n-2] = RB_bot   
        # RB_top[n]: top B on the right of site n
        RB_tops = [None] * (N-1)  # [RB_top[3], ..., RB_top[N], None]
        for n in reversed(range(2, N)):  # sites N, ..., 3
            if Bs[n] is not None:
                RB_top = np.tensordot(Bs[n], np.conj(ARs[n]), axes=((1, 2), (1, 2)))  
                # vL(n-1) [pn] [vRn], vL(n-1)* [pn*] [vRn*]  -> vL(n-1) vL(n-1)*
                RB_tops[n-2] = RB_top
            if RB_tops[n-1] is not None:
                RB_top = np.tensordot(ALs[n], RB_tops[n-1], axes=(2, 0))  
                # vL(n-1) pn [vRn], [vLn] vLn*
                RB_top = np.tensordot(RB_top, np.conj(ARs[n]), axes=((1, 2), (1, 2)))  
                # vL(n-1) [pn] [vLn*], vL(n-1)* [pn*] [vRn*] -> vL(n-1) vL(n-1)*
                if RB_tops[n-2] is not None:
                    RB_tops[n-2] += RB_top
                else:
                    RB_tops[n-2] = RB_top
        # RBB[n]: both Bs on the right of site n
        RBBs = [None] * (N-1)  # [RBB[3], ..., RBB[N], None]
        for n in reversed(range(2, N)):  # sites N, ..., 3
            if Bs[n] is not None:
                RBB = np.tensordot(Bs[n], np.conj(Bs[n]), axes=((1, 2), (1, 2)))  
                # vL(n-1) [pn] [vRn], vL(n-1)* [pn*] [vRn*]  -> vL(n-1) vL(n-1)*
                RBBs[n-2] = RBB
                if RB_bots[n-1] is not None:
                    RBB = np.tensordot(Bs[n], RB_bots[n-1], axes=(2, 0))  
                    # vL(n-1) pn [vRn], [vLn] vLn*
                    RBB = np.tensordot(RBB, np.conj(ALs[n]), axes=((1, 2), (1, 2)))  
                    # vL(n-1) [pn] [vLn*], vL(n-1)* [pn*] [vRn*] -> vL(n-1) vL(n-1)*
                    RBBs[n-2] += RBB
                if RB_tops[n-1] is not None:
                    RBB = np.tensordot(ALs[n], RB_tops[n-1], axes=(2, 0))  
                    # vL(n-1) pn [vRn], [vLn] vLn*
                    RBB = np.tensordot(RBB, np.conj(Bs[n]), axes=((1, 2), (1, 2)))  
                    # vL(n-1) [pn] [vLn*], vL(n-1)* [pn*] [vRn*] -> vL(n-1) vL(n-1)*
                    RBBs[n-2] += RBB
            if RBBs[n-1] is not None:
                RBB = np.tensordot(ALs[n], RBBs[n-1], axes=(2, 0))  
                # vL(n-1) pn [vRn], [vLn] vLn*
                RBB = np.tensordot(RBB, np.conj(ALs[n]), axes=((1, 2), (1, 2)))  
                # vL(n-1) [pn] [vLn*], vL(n-1)* [pn*] [vRn*] -> vL(n-1) vL(n-1)*
                if RBBs[n-2] is not None:
                    RBBs[n-2] += RBB
                else:
                    RBBs[n-2] = RBB
        E_bonds = []
        for n in range(N-1):
            E_bond = 0.
            if Bs[n] is not None:
                lhBB_11 = get_lh([Bs[n], ARs[n+1]], [Bs[n], ARs[n+1]], h_bonds[n])
                E_bond += np.trace(lhBB_11)
                if RB_bots[n] is not None:
                    lhB_10 = get_lh([Bs[n], ARs[n+1]], [ALs[n], ALs[n+1]], h_bonds[n]) 
                    E_bond += np.tensordot(lhB_10, RB_bots[n], axes=((0, 1), (0, 1)))
                if RB_tops[n] is not None:
                    lhB_01 = get_lh([ALs[n], ALs[n+1]], [Bs[n], ARs[n+1]], h_bonds[n])      
                    E_bond += np.tensordot(lhB_01, RB_tops[n], axes=((0, 1), (0, 1)))
            if Bs[n] is not None and Bs[n+1] is not None:
                lhBB_12 = get_lh([Bs[n], ARs[n+1]], [ALs[n], Bs[n+1]], h_bonds[n])
                lhBB_21= get_lh([ALs[n], Bs[n+1]], [Bs[n], ARs[n+1]], h_bonds[n])
                E_bond += np.trace(lhBB_12) + np.trace(lhBB_21)
            if Bs[n+1] is not None:
                lhBB_22 = get_lh([ALs[n], Bs[n+1]], [ALs[n], Bs[n+1]], h_bonds[n])
                E_bond += np.trace(lhBB_22)
                if RB_bots[n] is not None:            
                    lhB_20 = get_lh([ALs[n], Bs[n+1]], [ALs[n], ALs[n+1]], h_bonds[n])
                    E_bond += np.tensordot(lhB_20, RB_bots[n], axes=((0, 1), (0, 1)))
                if RB_tops[n] is not None:
                    lhB_02 = get_lh([ALs[n], ALs[n+1]], [ALs[n], Bs[n+1]], h_bonds[n])
                    E_bond += np.tensordot(lhB_02, RB_tops[n], axes=((0, 1), (0, 1)))              
            if LBBs[n] is not None:
                rh = get_rh([ARs[n], ARs[n+1]], [ARs[n], ARs[n+1]], h_bonds[n]) 
                E_bond += np.tensordot(LBBs[n], rh, axes=((0, 1), (0, 1)))
            if RBBs[n] is not None:
                lh = get_lh([ALs[n], ALs[n+1]], [ALs[n], ALs[n+1]], h_bonds[n])  
                E_bond += np.tensordot(lh, RBBs[n], axes=((0, 1), (0, 1)))
            E_bonds.append(np.real_if_close(E_bond))
        return E_bonds