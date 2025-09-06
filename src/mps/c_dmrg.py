"""Toy code implementing the density matrix renormalization group (DMRG) algorithm."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from functools import reduce

from ..matrix_decompositions import svd_truncation


def dmrg_algorithm(mpo, guess_mps0, D_max, eps, num_runs):
    """Find the MPS ground state of an MPO Hamiltonian with an initial guess and num_runs DMRG runs.
    Allow maximal bond dimension D_max and discard any singular values smaller than eps."""
    dmrg_engine = DMRGEngine(guess_mps0, mpo, D_max, eps)
    for i in range(num_runs):
        dmrg_engine.run()
    mps0 = dmrg_engine.mps
    E0 = mps0.get_mpo_expectation_value(mpo)
    var0 = mps0.get_mpo_variance(mpo)
    bond_dimensions0 = mps0.get_bond_dimensions()
    print(f"Performed MPS ground state search with {num_runs} DMRG runs. \n" \
          + f"Ground state energy: {E0}. \n" \
          + f"Ground state variance: {var0}. \n" \
          + f"Bond dimensions: {bond_dimensions0}.")
    return E0, mps0, var0           


class DMRGEngine:
    """Simple class for the DMRG engine to perform the variational ground state optimization for an
    MPO Hamiltonian H.

    E = <MPS|H|MPS> = <theta2[n,n+1]*|H_eff_2[n,n+1]|theta2[n,n+1]> for all n = 1, ..., N-1.
    Locally update theta2[n,n+1] as ground state of H_eff_2[n,n+1], restore canonical form with SVD
    and truncation, move to next bond. Sweep back and forth the whole chain till convergence in
    energy difference.
    
    Parameters
    ----------
    mps, mpo, D_max, eps: Same as attributes.

    Attributes
    ----------
    mps: MPS
         The current state to be iteratively optimized towards the ground state.
    mpo/Ws: list of np.array[ndim=4]s
            The matrix product operator of which the ground state is searched.
    D_max: int
           Maximum bond dimension, i.e. maximum number of singular values to keep.
    eps: float
         Discard any singular values smaller than that.
    N: int
       Length of the chain.
    Ls, Rs: list of np.array[ndim=3]s
            Left/Right environments of the effective Hamiltonians.
    """
    def __init__(self, mps, mpo, D_max, eps):
        self.mps = mps.copy()  # [AR_1, ..., AR_N] and [C_0, ..., C_(N-1)]
        self.Ws = mpo  # Ws = [W_1, ..., W_N]
        self.D_max = D_max
        self.eps = eps
        N = len(self.mps.ARs)
        self.N = N
        Ls = [None] * N  # [Id, L_1, ..., L_(N-2), None]
        Ls[0] = np.ones((1, 1, 1))  
        self.Ls = Ls  
        Rs = [None] * N  # [None, R_3, ..., R_N, Id]
        Rs[N-1] = np.ones((1, 1, 1))
        self.Rs = Rs
        for n in reversed(range(2, N)):
            self.Rs[n-1] = self.transfer_R(self.Rs[n], self.mps.ARs[n], self.Ws[n]) 

    def run(self):
        """Perform one update of self.mps in the variational ground state optimization for self.Ws,
        consisting of a sweep of bond updates from left to right and back."""
        N = self.N
        # sweep from left to right
        for n in range(N-2):  # bonds [1,2], ..., [N-2,N-1]
            self.update_bond(n)
        # sweep from right to left
        for n in reversed(range(N-1)):  # bonds [N-1,N], ..., [1,2]
            self.update_bond(n)

    def update_bond(self, n):
        """Perform one update of bond [n,n+1].
        
        1) theta2[n,n+1] -> ground state of H_eff_2[n,n+1],
        2) AL[n], C[n], AR[n+1] from SVD of theta2[n,n+1] and truncation,
        3) AR[n] -> C[n-1]^{-1} AL[n] C[n],
        4) transfer L[n-1] to L[n] with AL[n] and W[n], 
        5) transfer R[n+2] to R[n+1] with AR[n+1] and W[n+1].
        """
        H_eff_2 = Heff2(self.Ls[n], self.Ws[n], self.Ws[n+1], self.Rs[n+1])
        theta2_gs = self.get_theta2_gs(H_eff_2, guess=self.mps.get_theta2(n))
        AL_n, C_n, AR_np1 = split_truncate_theta2(theta2_gs, self.D_max, self.eps)
        # vL(n-1) pn vRn, vLn vRn, vLn p(n+1) vR(n+1)
        AR_n = np.tensordot(np.diag(np.diag(self.mps.Cs[n])**(-1)), AL_n, axes=(1, 0))  
        # vL(n-1) [vR(n-1)], [vL(n-1)] pn vRn
        AR_n = np.tensordot(AR_n, C_n, axes=(2, 0))  # vL(n-1) pn [vRn], [vLn] vRn -> vL(n-1) pn vRn
        self.mps.ARs[n] = AR_n
        self.mps.ARs[n+1] = AR_np1
        self.mps.Cs[n+1] = C_n
        self.Ls[n+1] = self.transfer_L(self.Ls[n], AL_n, self.Ws[n])
        self.Rs[n] = self.transfer_R(self.Rs[n+1], AR_np1, self.Ws[n+1])

    @staticmethod
    def get_theta2_gs(H_eff_2, guess):
        """Find the ground state of H_eff_2 with an initial guess."""
        guess = np.reshape(guess, H_eff_2.shape[1])
        _, theta2_gs = eigsh(H_eff_2, k=1, which="SA", v0=guess)
        theta2_gs = np.reshape(theta2_gs[:, 0], H_eff_2.shape_theta2)
        return theta2_gs

    @staticmethod
    def transfer_L(L, AL, W):
        """Transfer the left environment L by one site with AL and W.
        
         .---       .---(AL)--
         |          |    |
         |          |    |
        (L)--  ->  (L)--(W)---
         |          |    |
         |          |    |
         .---       .--(AL*)--
        """
        L_new = np.tensordot(L, AL, axes=(0, 0))  # [vR1] wR1 vR1*, [vL1] p vR2
        L_new = np.tensordot(L_new, W, axes=((0, 2), (0, 3)))  
        # [wR1] vR1* [p] vR2, [wL1] wR2 p [p*]
        L_new = np.tensordot(L_new, np.conj(AL), axes=((0, 3), (0, 1)))  
        # [vR1*] vR2 wR2 [p], [vL1*] [p*] vR2* -> vR2 wR2 vR2*
        return L_new
    
    @staticmethod
    def transfer_R(R, AR, W):
        """Transfer the right environment R by one site with AR and W.

        ---.       --(AR)---.
           |           |    |
           |           |    |
        --(R)  ->  ---(W)--(R)
           |           |    |
           |           |    |
        ---.       --(AR*)--.
        """
        R_new = np.tensordot(AR, R, axes=(2, 0))  # vL1 p [vR2], [vL2] wL2 vL2*
        R_new = np.tensordot(R_new, W, axes=((2, 1), (1, 3)))  
        # vL1 [p] [wL2] vL2*, wL1 [wL2] p [p*]
        R_new = np.tensordot(R_new, np.conj(AR), axes=((3, 1), (1, 2)))  
        # vL1 [vL2*] wL1 [p], vL1* [p*] [vR2*] -> vL1 wL1 vL1*
        return R_new


class Heff2(LinearOperator):
    """Class for the effective Hamiltonian acting on the two-site tensor theta2.

                                .---(theta2)----.
                                |    |    |     |
                                |    |    |     |
    matvec:  --(theta2)--  ->  (L)--(W1)-(W2)--(R)
                 |  |           |    |    |     |
                                |               |
                                .---        ----.
    """
    def __init__(self, L, W1, W2, R):
        self.L = L  # vR0 wR0 vR0*
        self.W1 = W1  # wL0 wR1 p1 p1*
        self.W2 = W2  # wL1 wR2 p2 p2*
        self.R = R  # vL2 wL2 vL2*
        D0 = np.shape(L)[0]
        d1 = np.shape(W1)[2]
        d2 = np.shape(W2)[2]
        D2 = np.shape(R)[0]
        self.shape = (D0 * d1 * d2 * D2, D0 * d1 * d2 * D2)
        self.shape_theta2 = (D0, d1, d2, D2)
        self.dtype = reduce(np.promote_types, [L.dtype, W1.dtype, W2.dtype, R.dtype])

    def _matvec(self, theta2):
        theta2 = np.reshape(theta2, self.shape_theta2)  # vL0 p1 p2 vR2
        theta2_new = np.tensordot(self.L, theta2, axes=(0, 0))  # [vR0] wR0 vR0*, [vL0] p1 p2 vR2
        theta2_new = np.tensordot(theta2_new, self.W1, axes=((0, 2), (0, 3)))  
        # [wR0] vR0* [p1] p2 vR2, [wL0] wR1 p1 [p1*]
        theta2_new = np.tensordot(theta2_new, self.W2, axes=((3, 1), (0, 3)))
        # vR0* [p2] vR2 [wR1] p1, [wL1] wR2 p2 [p2*]
        theta2_new = np.tensordot(theta2_new, self.R, axes=((1, 3), (0, 1)))  
        # vR0* [vR2] p1 [wR2] p2, [vL2] [wL2] vL2* -> vL0 p1 p2 vR2
        theta2_new = np.reshape(theta2_new, self.shape[0])  # vL0.p1.p2.vR2
        return theta2_new
    

def split_truncate_theta2(theta2, D_max, eps):
    """Split and truncate the effective two-site state theta2 with singular value decomposition.

                         SVD                       
    vL0--(theta2)--vR2   -->   vL0--(AL)--vR1  <-trunc->  vL1--(C)--vR1  <-trunc->  vL1--(AR)--vR2    
           |  |                      |                                                    |
           p1 p2                     p1                                                   p2
    """
    # combine legs and perform SVD
    D0, d1, d2, D2 = np.shape(theta2)
    theta2 = np.reshape(theta2, (D0 * d1, d2 * D2))
    # truncation: keep only singular values larger than eps, maximally D_max of them
    U, S, V, trunc_error = svd_truncation(theta2, D_max, eps)
    # split legs to get left orthonormal AL, diagonal C, right orthonormal AR
    D1 = np.shape(U)[1]
    AL = np.reshape(U, (D0, d1, D1))  # vL0 p1 vR1
    C = np.diag(S / np.linalg.norm(S))  # vL1 vR1
    AR = np.reshape(V, (D1, d2, D2))  # vL1 p2 vR2
    return AL, C, AR