"""Toy code implementing a uniform matrix product state (uMPS) in the thermodynamic limit."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from functools import reduce

from ..matrix_decompositions import qr_positive, lq_positive, svd_safe


class UniformMPS:
    """Simple class for a uMPS with single site unit cell in the thermodynamic limit.
    
    Parameters
    ----------
    AL, AR, AC, C: Same as attributes.
    
    Attributes
    ----------
    AL: np.array[ndim=3]
        Left orthonormal tensor with legs vL p vR (virtualLeft physical virtualRight).
        Legs of respective complex conjugate tensor are denoted by vL* p* vR*.
        Contracted legs are put in square brackets.
        Allowed contractions: [p*][p], [vR][vL], [vR*][vL*], [vL][vL*], [vR][vR*].
    AR: np.array[ndim=3]
        Right orthonormal tensor with legs vL p vR.
    AC: np.array[ndim=3]
        Center site tensor with legs vL p vR.
        Note that the canonical form AC = AL C = C AR is not fulfilled in VUMPS before convergence.
    C: np.array[ndim=2]
       Center matrix with legs vL vR.
       Note that C is not necessarily diagonal.
       Actively bring UniformMPS to diagonal C before computing entanglement entropy. 
    D: int
       Bond dimension.
    d: int
       Physical dimension.
    """
    def __init__(self, AL, AR, AC, C):
        self.AL = AL
        self.AR = AR
        self.AC = AC
        self.C = C
        self.D = np.shape(AL)[0]
        self.d = np.shape(AL)[1]

    @staticmethod
    def left_orthonormalize(A, tol=1.e-12):
        """Left orthonormalize the tensor A by successive positive QR decompositions.

        --L(i)--A--  =  --AL(i+1)--L(i+1)--  until convergence up to tolerance tol.
                |            |
        """
        D = np.shape(A)[0]
        d = np.shape(A)[1]
        L = np.random.normal(size=(D, D)) + 1.j*np.random.normal(size=(D, D))        
        maxiter = 10_000
        for i in range(maxiter):
            L /= np.linalg.norm(L)  # vL vR
            L_old = L   
            L_A = np.tensordot(L, A, axes=(1, 0))  # vL [vR], [vL] p vR
            L_A = np.reshape(L_A, (D*d, D))  # vL.p vR
            AL, L = qr_positive(L_A)  # vL.p vR, vL vR
            AL = np.reshape(AL, (D, d, D))  # vL p vR
            L /= np.linalg.norm(L)  # vL vR       
            err = np.linalg.norm(L - L_old)
            if err <= tol:
                print(f"AL, L: Converged up to tol={tol}. Final error after {i} iterations: {err}.")
                return AL, L        
            T = TransferMatrix([A], [AL], transpose=True)  # vL.vL* vR.vR*           
            _, L = T.get_leading_eigenpairs(k=1, guess=L)  # vR.vR*
            L = np.transpose(L, (1, 0))  # vR* vR 
            _, L = qr_positive(L)  # vL vR     
        raise RuntimeError(f"AL, L: Did not converge up to tol={tol}. \
                           Final error after {maxiter} iterations: {err}.")

    @staticmethod      
    def right_orthonormalize(A, tol=1.e-12):
        """Right orthonormalize the tensor A by successive positive LQ decompositions.

        --A--R(i)--  =  --R(i+1)--AR(i+1)--  until convergence up to tolerance tol.
          |                          |
        """
        D = np.shape(A)[0]
        d = np.shape(A)[1]    
        R = np.random.normal(size=(D, D)) + 1.j*np.random.normal(size=(D, D))        
        maxiter = 10_000
        for i in range(maxiter):
            R /= np.linalg.norm(R)  # vL vR
            R_old = R        
            A_R = np.tensordot(A, R, axes=(2, 0))  # vL p [vR], [vL] vR
            A_R = np.reshape(A_R, (D, d*D))  # vL p.vR
            R, AR = lq_positive(A_R)  # vL vR, vL p.vR
            AR = np.reshape(AR, (D, d, D))  # vL p vR
            R /= np.linalg.norm(R)  # vL vR        
            err = np.linalg.norm(R - R_old)
            if err <= tol:
                print(f"AR, R: Converged up to tol={tol}. Final error after {i} iterations: {err}.")
                return AR, R        
            T = TransferMatrix([A], [AR])  # vL.vL* vR.vR*
            _, R = T.get_leading_eigenpairs(k=1, guess=R)  # vL.vL*
            R, _ = lq_positive(R)  # vL vR        
        raise RuntimeError(f"AR, R: Did not converge up to tol={tol}. \
                           Final error after {maxiter} iterations: {err}.")

    @classmethod
    def to_canonical_form(cls, A, tol=1.e-12):
        """Bring tensor A to canonical form up to tolerance tol.
        
        1) Left orthonormalize A to get AL,
        2) Right orthonormalize AL to get AR and C, 
        3) To diagonal gauge via SVD.
        """
        AL, _ = cls.left_orthonormalize(A, tol=tol)  # vL p vR
        AR, C = cls.right_orthonormalize(AL, tol=tol)  # vL p vR, vL vR
        U, S, V = svd_safe(C)
        AL = np.tensordot(np.conj(U).T, np.tensordot(AL, U, axes=(2, 0)), axes=(1, 0))
        # vL [vR], [vL] p [vR], [vL] vR
        AR = np.tensordot(V, np.tensordot(AR, np.conj(V).T, axes=(2, 0)), axes=(1, 0))
        # vL [vR], [vL] p [vR], [vL] vR
        C = np.diag(S)/np.linalg.norm(S)  # vL vR
        AC = np.tensordot(AL, C, axes=(2, 0))  # vL p [vR], [vL] vR
        return AL, AR, AC, C
    
    @classmethod
    def from_non_canonical_tensor(cls, A, tol=1.e-12):
        """Initialize UniformMPS instance from a (non-canonical) injective tensor A."""
        AL, AR, AC, C = cls.to_canonical_form(A, tol=tol)
        return cls(AL, AR, AC, C)
    
    @staticmethod
    def get_random_tensor(D, d):
        """Create a random injective tensor of shape (D, d, D)."""
        A = np.random.normal(size=(D, d, D)) + 1.j * np.random.normal(size=(D, d, D))  # vL p vR
        # set largest eigenvalue of transfer matrix to 1 (normalize the corresponding MPS)
        T = TransferMatrix([A], [A])
        lambda1, _ = T.get_leading_eigenpairs(k=1)
        A /= np.sqrt(np.abs(lambda1))  # vL p vR
        return A
    
    @classmethod
    def from_desired_bond_dimension(cls, D, d=2, tol=1.e-12):
        """Initialize UniformMPS instance from a random tensor of bond/physical dimension D/d."""
        A = cls.get_random_tensor(D, d)
        AL, AR, AC, C = cls.to_canonical_form(A, tol=tol)
        return cls(AL, AR, AC, C)
      
    def copy(self):
        """Create a copy of the UniformMPS instance."""
        return UniformMPS(self.AL.copy(), self.AR.copy(), self.AC.copy(), self.C.copy())
        
    def get_theta2(self):
        """Compute the effective two-site state theta2.
        
        --(theta2)--  =  --(AL)--(AC)--
            |  |            |     |
        """
        return np.tensordot(self.AL, self.AC, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        
    def get_site_expectation_value(self, op1):
        """Compute the expectation value of a one-site operator op1.
        
               .--(theta1)--.     .--(AC)--.
               |     |      |     |   |    |
        e1  =  |   (op1)    |  =  | (op1)  |
               |     |      |     |   |    |
               .--(theta1*)-.     .--(AC*)-.
        """
        assert np.shape(op1) == (self.d, self.d)
        theta1 = self.AC  # vL p vR
        op1_theta1 = np.tensordot(op1, theta1, axes=(1, 1))  # p [p*], vL [p] vR
        theta1_op1_theta1 = np.tensordot(np.conj(theta1), op1_theta1, axes=((0, 1, 2), (1, 0, 2)))
        # [vL*] [p*] [vR*], [p] [vL] [vR]
        return np.real_if_close(theta1_op1_theta1)
    
    def get_bond_expectation_value(self, op2):
        """Compute the expectation value of a two-site operator op2.
        
               .--(theta2)--.  
               |    | |     |
        e2  =  |   (op2)    |
               |    | |     |
               .--(theta2*)-.
        """
        assert np.shape(op2) == (self.d, self.d, self.d, self.d)
        theta2 = self.get_theta2()  # vL p1 p2 vR
        op2_theta2 = np.tensordot(op2, theta2, axes=((2, 3), (1, 2)))
        # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        theta2_op2_theta2 = np.tensordot(np.conj(theta2), op2_theta2, axes=((0, 1, 2, 3), 
                                                                            (2, 0, 1, 3)))
        # [vL*] [p1*] [p2*] [vR*], [p1] [p2] [vL] [vR]
        return np.real_if_close(theta2_op2_theta2)
    
    def test_canonical_form(self):
        """Test the canonical form of the UniformMPS instance.
         
            .--(AL)--     .--         --(AR)--.     --.
            |   |         |              |    |       |
        [1] |   |      =  |   ,   [2]    |    |  =    |,   [3] --(AL)--(C)--  =  --(AC)--,    
            |   |         |              |    |       |           |                 |
            .-(AL*)--     .--         --(AR*)-.     --.

        [4] --(C)--(AR)-- = --(AC)--.
                    |          |
        """
        canonical_form = np.zeros(4)
        Id = np.eye(self.D)  # vR vR* or vL vL*
        Id_L = np.tensordot(self.AL, np.conj(self.AL), axes=((0, 1), (0, 1)))
        # [vL] [p] vR, [vL*] [p*] vR* -> vR vR*
        Id_R = np.tensordot(self.AR, np.conj(self.AR), axes=((1, 2), (1, 2)))
        # vL [p] [vR], vL* [p*] [vR*] -> vL vL*
        canonical_form[0] = np.linalg.norm(Id_L - Id)  
        canonical_form[1] = np.linalg.norm(Id_R - Id) 
        AC = self.AC  # vL p vR
        AC_L = np.tensordot(self.AL, self.C, axes=(2, 0))  # vL p [vR], [vL] vR -> vL p vR
        AC_R = np.tensordot(self.C, self.AR, axes=(1, 0))  # vL [vR], [vL] p vR -> vL p vR
        canonical_form[2] = np.linalg.norm(AC_L - AC)
        canonical_form[3] = np.linalg.norm(AC_R - AC)
        return canonical_form
    
    def to_diagonal_gauge(self):
        """Bring the UniformMPS instance to normalized diagonal gauge.
         
        Compute SVD C=USV and transform: --(AL)--  ->  --(U^{dagger})--(AL)--(U)--,
                                            |                           |
                        
                                         --(AR)--  ->  --(V)--(AR)--(V^{dagger})--,
                                            |                  |
    
                                         --(C)--  ->  1/norm(S) --(S)--.
        """
        U, S, V = svd_safe(self.C)  # vL vR
        self.AL = np.tensordot(np.conj(U).T, np.tensordot(self.AL, U, axes=(2, 0)), axes=(1, 0))
        # vL [vR], [vL] p [vR], [vL] vR 
        self.AR = np.tensordot(V, np.tensordot(self.AR, np.conj(V).T, axes=(2, 0)), axes=(1, 0))
        # vL [vR], [vL] p [vR], [vL] vR
        self.C = np.diag(S)/np.linalg.norm(S)  # vL vR
        self.AC = np.tensordot(self.AL, self.C, axes=(2, 0))  # vL p [vR], [vL] vR
    
    def get_entanglement_entropy(self):
        """Compute the entanglement entropy for a bipartition of the UniformMPS instance.

        S = -sum_{alpha=1}^D c_{alpha}^2 log(c_{alpha}^2) with c_{alpha} the singular values of C.
        """
        _, C, _ = svd_safe(self.C)
        C = C[C > 1.e-20] # 0*log(0) should give 0 and won't contribute to the sum
        # avoid warning or NaN by discarding the very small values of S
        assert abs(np.linalg.norm(C) - 1.) < 1.e-13
        C2 = C * C
        return -np.sum(C2 * np.log(C2))
    
    def get_correlation_length(self):
        """Compute the correlation length by diagonalizing the transfer matrix.
        
        xi = -1/log(|lambda_2|), with |lambda_2| second largest eigenvalue magnitude.
        """
        T = TransferMatrix([self.AL], [self.AL])
        lambdas, _ = T.get_leading_eigenpairs(k=2)
        xi = -1./np.log(np.abs(lambdas[1]))
        return xi
        
    def get_correlation_functions(self, X, Y, N):
        """Compute the correlation functions C_XY(n) = <psi(A)|(X_0)(Y_n)|psi(A)> for n = 1,...,N.
                
                    .--(AC)--  --(AR)--^{n-1}  --(AR)--.      .--  --(AR)--^{n-1}  --.     
                    |   |         |               |    |      |       |              |
        C_XY(n)  =  |  (X)        |              (Y)   |  =  (LX)     |            (RY)
                    |   |         |               |    |      |       |              |
                    .--(AC*)-  --(AR*)-        --(AR*)-.      .--  --(AR*)-        --.
        """
        LX = np.tensordot(X, self.AC, axes=(1, 1))  # p [p*], vL [p] vR
        LX = np.tensordot(LX, np.conj(self.AC), axes=((1, 0), (0, 1)))
        # [p] [vL] vR, [vL*] [p*] vR*
        RY = np.tensordot(Y, self.AR, axes=(1, 1))  # p [p*], vL [p] vR
        RY = np.tensordot(RY, np.conj(self.AR), axes=((0, 2), (1, 2)))
        # [p] vL [vR], vL* [p*] [vR*]
        Cs = []
        for n in range(N):
            C = np.tensordot(LX, RY, axes=((0, 1), (0, 1)))  # [vR] [vR*], [vL] [vL*]
            Cs.append(C.item())
            LX = np.tensordot(LX, self.AR, axes=(0, 0))  # [vR] vR*, [vL] p vR
            LX = np.tensordot(LX, np.conj(self.AR), axes=((0, 1), (0, 1)))
            # [vR*] [p] vR, [vL*] [p*] vR*
        return np.real_if_close(Cs)
    
       
class TransferMatrix(LinearOperator):
    """Class for a transfer matrix.
    
                                 ---.       ---(A1)-- ... --(AL)---.
                                    |           |            |     |
    matvec for transpose=False:    (X)  ->      |            |    (X)
                                    |           |            |     |
                                 ---.       --(B1*)- ... --(BL*)---.
                                                                        
                                 .---       .---(A1)-- ... --(AL)---
                                 |          |    |            |
    matvec for transpose=True:  (X)    ->  (X)   |            |
                                 |          |    |            |   
                                 .---       .--(B1*)-- ... -(BL*)---
    """ 
    def __init__(self, As, Bs, transpose=False):
        self.As = As
        self.Bs = Bs
        self.transpose = transpose
        DA = np.shape(self.As[0])[0]  
        DB = np.shape(self.Bs[0])[0]  
        self.shape = (DA * DB, DA * DB)
        self.shape_X = (DA, DB)
        self.dtype = reduce(np.promote_types, [As[0].dtype, Bs[0].dtype])

    def _matvec(self, X):
        X = np.reshape(X, self.shape_X)  # vL vL* or vR vR*
        if not self.transpose:
            for A, B in zip(reversed(self.As), reversed(self.Bs)):
                X = np.tensordot(A, X, axes=(2, 0))  # vL p [vR], [vL] vL*
                X = np.tensordot(X, np.conj(B), axes=((1, 2), (1, 2)))
                # vL [p] [vL*], vL* [p*] [vR*]
            X = np.reshape(X, self.shape[1])  # vL.vL*
            return X
        else:
            for A, B in zip(self.As, self.Bs):
                X = np.tensordot(X, A, axes=(0, 0))  # [vR] vR*, [vL] p vR
                X = np.tensordot(X, np.conj(B), axes=((0, 1), (0, 1)))
                # [vR*] [p] vR, [vL*] [p*] vR*
            X = np.reshape(X, self.shape[0])  # vR.vR*
            return X
        
    def get_leading_eigenpairs(self, k=1, guess=None):
        """Compute the k largest eigenvalues (in magnitude) and the corresponding eigenvectors."""
        if guess is not None:
            guess = np.reshape(guess, self.shape[1])
        lambdas, Vs = eigs(self, k=k, which="LM", v0=guess)
        # sort the eigenvalues in decreasing order
        ind_sort = np.argsort(np.abs(lambdas))[::-1]
        lambdas = lambdas[ind_sort]
        Vs = Vs[:, ind_sort]  # vL.vL* or vR.vR*
        Vs_matrices = []
        for i in range(k):
            V = Vs[:, i]
            Vs_matrices.append(np.reshape(V, self.shape_X))  # vL vL* or vR vR*
        if k == 1:
            return lambdas[0], Vs_matrices[0]
        return lambdas, Vs_matrices