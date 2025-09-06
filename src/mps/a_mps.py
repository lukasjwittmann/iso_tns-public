"""Toy code implementing a finite matrix product state (MPS) with open boundary conditions."""

import numpy as np

from ..matrix_decompositions import lq_positive, svd_safe


class MPS:
    """Simple class for a finite MPS in right canonical form.
    
    Parameters
    ----------
    ARs, Cs: Same as attributes.
    
    Attributes
    ----------
    ARs: list of np.array[ndim=3]s
         Right orthonormal tensors with legs vL p vR (virtualLeft physical virtualRight).
         Legs of respective complex conjugate tensor are denoted by vL* p* vR*.
         Contracted legs are put in square brackets.
         Allowed contractions: [p*][p], [vR][vL], [vR*][vL*], [vL][vL*], [vR][vR*].
    Cs: list of np.array[ndim=2]s
        Diagonal center matrices with legs vL vR, containing the Schmidt values on the diagonal.
    N: int
       Length of the MPS.
    d: int
       Physical dimension.
    """
    def __init__(self, ARs, Cs):
        self.ARs = ARs  # [AR[1], ..., AR[N]]
        self.Cs = Cs  # [C[0], ..., C[N-1]]
        self.N = len(self.ARs)
        self.d = np.shape(self.ARs[0])[1]

    @staticmethod
    def right_orthonormalize(As):
        """Right orthonormalize the tensors As by performing LQ decompositions from right to left."""
        N = len(As)
        ARs = [None] * N  # [AR[1], ..., AR[N]]
        L = np.ones((1, 1))  # L[N+1]
        for n in reversed(range(N)):  # sites N, ..., 1
            A = np.tensordot(As[n], L, axes=(2, 0))  # vL(n-1) pn [vRn], [vLn] vRn -> vL(n-1) pn vRn
            Dnm1, d, Dn = np.shape(A)
            A = np.reshape(A, (Dnm1, d * Dn))  # vL(n-1) pn.vRn
            L, Q = lq_positive(A)  # vL(n-1) vR(n-1), vL(n-1) pn.vRn
            Dnm1_new = np.shape(Q)[0]
            ARs[n] = np.reshape(Q, (Dnm1_new, d, Dn))  # vL(n-1) pn vRn
        return ARs
    
    @classmethod
    def to_canonical_form(cls, As):
        """Bring the tensors As to canonical form by right orthonormalizing them from right to left
        and subsequently performing SVDs from left to right."""
        N = len(As)
        ARs = cls.right_orthonormalize(As)  # [AR[1], ..., AR[N]]
        ALs = [None] * N   # [AL[1], ..., AL[N]]
        Cs = [None] * N  # [C[0], ..., C[N-1]]
        Cs[0] = np.ones((1, 1))  # C[0]
        for n in range(N):  # sites 1, ..., N
            AC = np.tensordot(Cs[n], ARs[n], axes=(1, 0))  
            # vL(n-1) [vR(n-1)], [vL(n-1)] pn vRn -> vL(n-1) pn vRn
            Dnm1, d, Dn = np.shape(AC)
            AC = np.reshape(AC, (Dnm1 * d, Dn))  # vL(n-1).pn vRn
            U, S, V = svd_safe(AC)  # vL(n-1).pn vRn, vLn vRn, vLn vRn
            Dn_new = np.shape(U)[1]
            ALs[n] = np.reshape(U, (Dnm1, d, Dn_new))  # vL(n-1) pn vRn
            ARs[n] = np.tensordot(ARs[n], np.conj(V).T, axes=(2, 0))  # vL(n-1) pn [vRn], [vLn] vRn
            if n < N-1:
                ARs[n+1] = np.tensordot(V, ARs[n+1], axes=(1, 0))  # vLn [vRn], [vLn] pn vR(n+1)
                Cs[n+1] = np.diag(S / np.linalg.norm(S))  # vLn vRn
        return ALs, ARs, Cs
    
    @classmethod
    def from_non_canonical_tensors(cls, As):
        """Initialize MPS instance from (non-canonical) tensors As."""
        _, ARs, Cs = cls.to_canonical_form(As)
        return cls(ARs, Cs)
    
    @classmethod
    def from_desired_bond_dimension(cls, N, D_max, d=2):
        """Initialize MPS instance from random tensors of maximal bond dimension D_max."""
        As_random = [None] * N
        As_random[0] = np.random.normal(size=(1, d, D_max)) \
                       + 1.j * np.random.normal(size=(1, d, D_max))
        for n in range(1, N-1):
            As_random[n] = np.random.normal(size=(D_max, d, D_max)) \
                           + 1.j * np.random.normal(size=(D_max, d, D_max))
        As_random[N-1] = np.random.normal(size=(D_max, d, 1)) \
                         + 1.j * np.random.normal(size=(D_max, d, 1))
        return cls.from_non_canonical_tensors(As_random)
    
    @classmethod
    def from_qubit_product_state(cls, N, spin_orientation="up"):
        """Initialize MPS instance from a product state with all N spin-1/2s in spin_orientation."""
        C = np.ones((1, 1))
        Cs = [C] * N
        AR = np.zeros((1, 2, 1))
        if spin_orientation == "up":
            AR[0, 0, 0] = 1.
        elif spin_orientation == "down":
            AR[0, 1, 0] = 1.
        elif spin_orientation == "right":
            AR[0, 0, 0] = 1/np.sqrt(2)
            AR[0, 1, 0] = 1/np.sqrt(2)
        elif spin_orientation == "left":
            AR[0, 0, 0] = 1/np.sqrt(2)
            AR[0, 1, 0] = -1/np.sqrt(2)
        else:
            raise ValueError(f"choose spin orientation \"up\", \"down\", \"right\" or \"left\".")
        ARs = [AR] * N
        return cls(ARs, Cs)            

    def copy(self):
        """Create a copy of the MPS instance."""
        return MPS([AR.copy() for AR in self.ARs], [C.copy() for C in self.Cs])
    
    def get_bond_dimensions(self):
        """Return all bond dimensions D0, ..., DN."""
        N = self.N
        Ds = []  # D0=1, D1, ..., D(N-1), DN=1
        for n in range(N):  
            Ds.append(np.shape(self.ARs[n])[0])  
        Ds.append(np.shape(self.ARs[N-1])[2])
        return Ds

    def get_theta1(self, n):
        """Compute the effective one-site state theta1 for site n.
        
        --(theta1[n])--  =  --(C[n-1])--(AR[n])--
              |                            |
        """
        AC = np.tensordot(self.Cs[n], self.ARs[n], axes=(1, 0))  
        # vL(n-1) [vR(n-1)], [vL(n-1)] pn vRn -> vL(n-1) pn vRn
        return AC
    
    def get_theta2(self, n):
        """Compute the effective two-site state theta2 for sites n and n+1.
        
        --(theta2[n,n+1])--  =  --(AC[n])--(AR[n+1])--
              |     |                |         |
        """
        theta2 = np.tensordot(self.get_theta1(n), self.ARs[n+1], axes=(2, 0))  
        # vL(n-1) pn [vRn], [vLn] p(n+1) vR(n+1) -> vL(n-1) pn p(n+1) vR(n+1)
        return theta2
    
    def get_site_expectation_values(self, op1s):
        """Compute the expectation values of a list of self.N one-site operators op1s.
        
                  .--(theta1[n])--.     
                  |       |       |     
        e1[n]  =  |   (op1[n])    |  
                  |       |       |     
                  .--(theta1[n]*)-.     
        """
        e1s = []
        for n in range(self.N):  # sites 1, ..., N
            theta1 = self.get_theta1(n)  # vL(n-1) pn vRn
            op1 = op1s[n]  # pn pn*
            op1_theta1 = np.tensordot(op1, theta1, axes=(1, 1))  # pn [pn*], vL(n-1) [pn] vRn
            theta1_op1_theta1 = np.tensordot(np.conj(theta1), op1_theta1, axes=((0, 1, 2), 
                                                                                (1, 0, 2)))
            # [vL(n-1)*] [pn*] [vRn*], [pn] [vL(n-1)] [vRn]
            e1s.append(np.real_if_close(theta1_op1_theta1))
        return e1s
    
    def get_bond_expectation_values(self, op2s):
        """Compute the expectation values of a list of self.N-1 two-site operators op2s.

                  .--(theta2[n,n+1])--.  
                  |      |     |      |
        e2[n]  =  |    (op2[n,n+1])   |
                  |      |     |      |
                  .--(theta2[n,n+1]*)-.        
        """
        e2s = []
        for n in range(self.N-1):  # bonds 1, ..., N-1
            theta2 = self.get_theta2(n)  # vL(n-1) pn p(n+1) vR(n+1)
            op2 = op2s[n]  # pn p(n+1) pn* p(n+1)*
            op2_theta2 = np.tensordot(op2, theta2, axes=((2, 3), (1, 2)))
            # pn p(n+1) [pn*] [p(n+1)*], vL(n-1) [pn] [p(n+1)] vR(n+1)
            theta2_op2_theta2 = np.tensordot(np.conj(theta2), op2_theta2, axes=((0, 1, 2, 3), 
                                                                                (2, 0, 1, 3)))
            # [vL(n-1)*] [pn*] [p(n+1)*] [vR(n+1)*], [pn] [p(n+1)] [vL(n-1)] [vR(n+1)]
            e2s.append(np.real_if_close(theta2_op2_theta2))
        return e2s
    
    def get_mpo_expectation_value(self, Ws):
        """Compute the expectation value of a matrix product operator.
        
                          .--(AR[1])--(AR[2])-- ... --(AR[N-1])--(AR[N])--.
                          |     |        |                |         |     |
        <mps|mpo|mps>  =  .--(W[1])---(W[2])--- ... --(W[N-1])---(W[N])---.
                          |     |        |                |         |     |               
                          .--(AR[1]*)-(AR[2]*)- ... --(AR[N-1]*)-(AR[N]*)-.
        """
        N = self.N
        AR_1 = self.ARs[0][0, :, :]  # p1 vR1
        W_1 = Ws[0][0, :, :, :]  # wR1 p1 p1*
        E = np.tensordot(AR_1, W_1, axes=(0, 2))  # [p1] vR1, wR1 p1 [p1*]
        E = np.tensordot(E, np.conj(AR_1), axes=(2, 0))  # vR1 wR1 [p1], [p1*] vR1* -> vR1 wR1 vR1*
        for n in range(1, N-1):  # sites 2, ..., N-1
            E = np.tensordot(E, self.ARs[n], axes=(0, 0))  
            # [vR(n-1)] wR(n-1) vR(n-1)*, [vL(n-1)] pn vRn
            E = np.tensordot(E, Ws[n], axes=((0, 2), (0, 3)))  
            # [wR(n-1)] vR(n-1)* [pn] vRn, [wL(n-1)] wRn pn [pn*]
            E = np.tensordot(E, np.conj(self.ARs[n]), axes=((0, 3), (0, 1)))  
            # [vR(n-1)*] vRn wRn [pn], [vL(n-1)*] [pn*] vRn* -> vRn wRn vRn*
        AR_N = self.ARs[N-1][:, :, 0]  # vL(N-1) pN
        W_N = Ws[N-1][:, 0, :, :]  # wL(N-1) pN pN*
        E = np.tensordot(E, AR_N, axes=(0, 0))  # [vR(N-1)] wR(N-1) vR(N-1)*, [vL(N-1)] pN
        E = np.tensordot(E, W_N, axes=((0, 2), (0, 2)))  
        # [wR(N-1)] vR(N-1)* [pN], [wL(N-1)] pN [pN*]
        E = np.tensordot(E, np.conj(AR_N), axes=((0, 1), (0, 1)))  # [vR(N-1)*] [pN], [vL(N-1)] [pN]
        return np.real_if_close(E)
    
    def get_mpo_variance(self, Ws):
        """Compute the variance of a matrix product operator.
        
        var = <mps|mpo^2|mps> - <mps|mpo|mps>^2.
        """
        N = self.N
        W2s = [None] * N
        for n in range(N):
            chi_L = np.shape(Ws[n])[0]
            chi_R = np.shape(Ws[n])[1]
            W2 = np.tensordot(Ws[n], Ws[n], axes=(3, 2))  # wL1 wR1 p [p*], wL2 wR2 [p] p*
            W2 = np.transpose(W2, (0, 3, 1, 4, 2, 5))  # wL1 wL2 wR1 wR2 p p*
            W2 = np.reshape(W2, (chi_L**2, chi_R**2, self.d, self.d))  # wL1.wL2 wR1.wR2 p p*
            W2s[n] = W2
        E = self.get_mpo_expectation_value(Ws)
        E2 = self.get_mpo_expectation_value(W2s)
        var = E2 - E**2
        return np.real_if_close(var)
    
    def get_entanglement_entropies(self):
        """Compute the entanglement entropies for all self.N-1 bipartitions of the MPS instance."""
        Ss = []
        for n in range(1, self.N):
            C = np.diag(self.Cs[n])
            C = C[C > 1.e-20]
            assert abs(np.linalg.norm(C) - 1.) < 1.e-13
            C2 = C * C
            Ss.append(-np.sum(C2 * np.log(C2)))
        return Ss

    def get_correlation_functions(self, X, Y, n, N):
        """Compute the correlation functions C_XY(n,m) = <mps|(X_n)(Y_m)|mps> for m = n+1, ..., N.

                      .--(AC[n])--  --(AR[n+1])-- ... --(AR[m-1])-- --(AR[m])--.           
                      |     |             |                 |            |     |      
        C_XY(n,m)  =  |    (X)            |                 |           (Y)    |  
                      |     |             |                 |            |     |      
                      .--(AC[n]*)-  --(AR[n+1]*)- ... --(AR[m-1]*)- --(AR[m]*)-.    
        
        Also compute the connected correlation functions <(X_n)(Y_m)> - <X_n><Y_m>.
        """
        LX = np.tensordot(X, self.get_theta1(n), axes=(1, 1))  # pn [pn*], vL(n-1) [pn] vRn
        LX = np.tensordot(LX, np.conj(self.get_theta1(n)), axes=((1, 0), (0, 1)))  
        # [pn] [vL(n-1)] vRn, [vL(n-1)*] [pn*] vRn* -> vRn vRn*
        Cs = []
        eXs = self.get_site_expectation_values([X] * self.N)
        eYs = self.get_site_expectation_values([Y] * self.N)
        Cs_connected = []
        for m in range(n+1, N):
            RY = np.tensordot(Y, self.ARs[m], axes=(1, 1))  # pm [pm*], vL(m-1) [pm] vRm
            RY = np.tensordot(RY, np.conj(self.ARs[m]), axes=((0, 2), (1, 2)))  
            # [pm] vL(m-1) [vRm], vL(m-1)* [pm*] [vRm*] -> vL(m-1) vL(m-1)*
            C = np.tensordot(LX, RY, axes=((0, 1), (0, 1)))  
            # [vR(m-1)] [vR(m-1)*], [vL(m-1)] [vL(m-1)*]
            C_connected = C - eXs[n] * eYs[m]
            Cs.append(C)  
            Cs_connected.append(C_connected)
            LX = np.tensordot(LX, self.ARs[m], axes=(0, 0))  # [vR(m-1)] vR(m-1)*, [vL(m-1)] pm vRm
            LX = np.tensordot(LX, np.conj(self.ARs[m]), axes=((0, 1), (0, 1)))
            # [vR(m-1)*] [pm] vRm, [vL(m-1)*] [pm*] vRm* -> vRm vRm*
        return np.real_if_close(Cs), np.real_if_close(Cs_connected)
    
    def get_translation_expectation_value(self):
        """Compute the expectation value of the translation operator.

                                  (AR[1])--- ... --(AR[N-1])--(AR[N])
        <mps|T|mps>  =       ________|_________________|________/    .                                                                          
                            /        |                 |                 
                        (AR[1]*)--(AR[2]*)-- ... --(AR[N]*)
        """
        t = np.conj(self.ARs[0])  # vR0 p1* vR1*
        for n in range(self.N-1):
            t = np.tensordot(t, np.conj(self.ARs[n+1]), axes=(2, 0))  
            # vR(n-1) p1* [vRn*], [vLn*] p(n+1)* vR(n+1)*
            t = np.tensordot(t, self.ARs[n], axes=((0, 2), (0, 1)))  
            # [vR(n-1)] p1* [p(n+1)*] vR(n+1)*, [vL(n-1)] [pn] vRn
            t = np.transpose(t, (2, 0, 1))  # vRn p1* vR(n+1)*
        t = np.tensordot(t, self.ARs[-1], axes=((0, 1, 2), (0, 1, 2)))  
        # [vR(N-1)] [p1*] [vRN*], [vL(N-1)] [pN] [vRN]
        return np.real_if_close(t)
    
    def get_translation2_expectation_value(self):
        """Compute the expectation value of the squared translation operator.

                                              (AR[1])--- ... --(AR[N-2])--(AR[N-1])--(AR[N])
                                          _______|_________________|_________/_________/    
        <mps|T^2|mps>  =       __________/_______|_________________|________/               .                                                               
                              /         /        |                 |                 
                          (AR[1]*)--(AR[2]*)--(AR[3]*)-- ... --(AR[N]*)

        """
        t2 = np.tensordot(np.conj(self.ARs[0]), np.conj(self.ARs[1]), axes=(2, 0))  
        # vR0 p1* [vR1*], [vL1*] p2* vR2*
        for n in range(self.N-2):
            t2 = np.tensordot(t2, np.conj(self.ARs[n+2]), axes=(3, 0))  
            # vR(n-1) p1* p2* [vR(n+1)*], [vL(n+1)*] p(n+2)* vR(n+2)*
            t2 = np.tensordot(t2, self.ARs[n], axes=((0, 3), (0, 1)))
            # [vR(n-1)] p1* p2* [p(n+2)*] vR(n+2)*, [vL(n-1)] [pn] vRn
            t2 = np.transpose(t2, (3, 0, 1, 2))
            # vRn p1* p2* vR(n+2)*
        t2 = np.tensordot(t2, self.ARs[-2], axes=((0, 1), (0, 1)))
        # [vR(N-2)] [p1*] p2* vRN*, [vL(N-2)] [p(N-1)] vR(N-1)
        t2 = np.tensordot(t2, self.ARs[-1], axes=((0, 1, 2), (1, 2, 0)))
        # [p2*] [vRN*] [vR(N-1)], [vL(N-1)] [pN] [vRN]
        return np.real_if_close(t2)
    
    def get_translation_variance(self):
        """Compute the variance of the translation operator.

        var(T) = <mps|T^2|mps> - <mps|T|mps>^2.
        """
        var = self.get_translation2_expectation_value() - self.get_translation_expectation_value()**2
        return np.real_if_close(var)