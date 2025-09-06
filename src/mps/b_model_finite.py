"""Toy code implementing the transverse field Ising (TFI) model on a finite chain."""

import numpy as np
import scipy.sparse as sparse


class TFIModelFinite:
    """Class generating different representations of the TFI Hamiltonian on a finite chain.
    
    OBC: H = -J * sum_{n=1}^{N-1} sigma^{x}_{n} sigma^{x}_{n+1} - g * sum_{n=1}^{N} sigma^{z}_{n},
    PBC: H = -J * sum_{n=1}^{N} sigma^{x}_{n} sigma^{x}_{n+1} - g * sum_{n=1}^{N} sigma^{z}_{n}.

    Parameters
    ----------
    N, J, g: Same as attributes.

    Attributes
    ----------
    N: int
       Length of the chain.
    J, g: float
          Coupling parameters of the above defined TFI Hamiltonian.
    sigma_x, sigma_y, sigma_z, Id: np.array[ndim=2]
                                   Pauli matrices and identity, legs p p*.
    """
    def __init__(self, N, g, J=1.):
        self.N = N
        self.J = J
        self.g = g
        self.sigma_x = np.array([[0., 1.], [1., 0.]])
        self.sigma_y = np.array([[0., -1.j], [1.j, 0.]])
        self.sigma_z = np.array([[1., 0.], [0., -1.]])
        self.Id = np.eye(2)

    # Hamiltonian and translation matrices for exact diagonalization
    def get_H_bonds(self, bc):
        """Generate the TFI two-site Hamiltonians as sparse matrices."""
        N = self.N
        J = self.J
        g = self.g
        Id = sparse.csr_matrix(self.Id)
        sigma_x = sparse.csr_matrix(self.sigma_x)
        sigma_z = sparse.csr_matrix(self.sigma_z)

        def singlesite_to_full(op, n):
            """For a site operator op acting on site n, generate the full operator acting with
            identities on all other sites."""
            ops = [Id] * N
            ops[n] = op
            Op = ops[0]
            for m in range(1, N):
                Op = sparse.kron(Op, ops[m], format="csr")
            return Op
        
        if bc == "open":
            H_bonds = [None] * (N-1)
            # bond 1
            H_bonds[0] = - J * singlesite_to_full(sigma_x, 0) @ singlesite_to_full(sigma_x, 1) \
                         - g * singlesite_to_full(sigma_z, 0) @ singlesite_to_full(Id, 1) \
                         - (0.5 * g) * singlesite_to_full(Id, 0) @ singlesite_to_full(sigma_z, 1)
            for n in range(1, N-2):  # bonds 2, ..., N-2
                H_bonds[n] = - J * singlesite_to_full(sigma_x, n) @ singlesite_to_full(sigma_x, n+1) \
                             - (0.5 * g) * singlesite_to_full(sigma_z, n) @ singlesite_to_full(Id, n+1) \
                             - (0.5 * g) * singlesite_to_full(Id, n) @ singlesite_to_full(sigma_z, n+1)
            # bond N-1
            H_bonds[N-2] = - J * singlesite_to_full(sigma_x, N-2) @ singlesite_to_full(sigma_x, N-1) \
                           - (0.5 * g) * singlesite_to_full(sigma_z, N-2) @ singlesite_to_full(Id, N-1) \
                           - g * singlesite_to_full(Id, N-2) @ singlesite_to_full(sigma_z, N-1)
            return H_bonds
        elif bc == "periodic":
            H_bonds = [None] * N
            for n in range(N):  # bonds 1, ..., N
                H_bonds[n] = - J * singlesite_to_full(sigma_x, n) @ singlesite_to_full(sigma_x, (n+1)%N) \
                             - (0.5 * g) * singlesite_to_full(sigma_z, n) @ singlesite_to_full(Id, (n+1)%N) \
                             - (0.5 * g) * singlesite_to_full(Id, n) @ singlesite_to_full(sigma_z, (n+1)%N)
            return H_bonds
        else:
            raise ValueError(f"The boundary conditions (bc) must be either \"open\" or \"periodic\".")
    
    def get_H(self, bc):
        """Generate the full TFI Hamiltonian as a sparse matrix, equal to the sum of all two-site 
        Hamiltonians."""
        N = self.N
        H = sparse.csr_matrix((2**N, 2**N))
        if bc == "open":
            H_bonds = self.get_H_bonds(bc="open")
            for n in range(N-1):
                H += H_bonds[n]
            return H
        elif bc == "periodic":
            H_bonds = self.get_H_bonds(bc="periodic")
            for n in range(N):
                H += H_bonds[n]
            return H
        else:
            raise ValueError(f"The boundary conditions (bc) must be either \"open\" or \"periodic\".")

    def integer_to_binary(self, s):
        """Convert an integer number to a binary number."""
        return bin(s)[2:].zfill(self.N)

    def translate_binary(self, s):
        """Translate a binary number by one bit to the right."""
        return s[-1] + s[:-1]

    def binary_to_integer(self, s):
        """Convert a binary number to an integer number."""
        return int(s, base=2)
        
    def get_T(self):
        """Generate the translation operator T|s1, ..., sN> = |sN, s1, ..., s(N-1)> as a sparse
        matrix."""
        N = self.N
        T = sparse.lil_matrix((2**N, 2**N))
        for s in range(2**N):
            T[s, self.binary_to_integer(self.translate_binary(self.integer_to_binary(s)))] = 1.
        T = sparse.csr_matrix(T)
        return T

    def get_T2(self):
        """Generate the squared translation operator T^2|s1, ..., sN> = |s(N-1), sN, s1, ..., s(N-2)>
        as a sparse matrix."""
        N = self.N
        T2 = sparse.lil_matrix((2**N, 2**N))
        for s in range(2**N):
            T2[s, self.binary_to_integer(self.translate_binary(self.translate_binary(self.integer_to_binary(s))))] = 1.
        T2 = sparse.csr_matrix(T2)
        return T2
        
    # bond Hamiltonians and MPO for MPS methods
    def get_h_bonds(self):
        """Generate the TFI two-site Hamiltonians as local tensors."""
        N = self.N
        J = self.J
        g = self.g
        Id = self.Id
        sigma_x = self.sigma_x
        sigma_z = self.sigma_z
        h_bonds = [None] * (N-1)
        # bond 1
        h = - J * np.kron(sigma_x, sigma_x) \
            - g * np.kron(sigma_z, Id) \
            - (0.5 * g) * np.kron(Id, sigma_z) # p1.p2 p1*.p2*  
        h_bonds[0] = np.reshape(h, (2, 2, 2, 2))  # p1 p2 p1* p2*   
        for n in range(1, N-2):  # bonds 2, ..., N-2
            h = - J * np.kron(sigma_x, sigma_x) \
                - (0.5 * g) * np.kron(sigma_z, Id) \
                - (0.5 * g) * np.kron(Id, sigma_z) # pn.p(n+1) pn*.p(n+1)*
            h_bonds[n] = np.reshape(h, (2, 2, 2, 2))  # pn p(n+1) pn* p(n+1)* 
        # bond N-1
        h = - J * np.kron(sigma_x, sigma_x) \
            - (0.5 * g) * np.kron(sigma_z, Id) \
            - g * np.kron(Id, sigma_z) # p(N-1).pN p(N-1)*.pN*
        h_bonds[N-2] = np.reshape(h, (2, 2, 2, 2))  # p(N-1) pN p(N-1)* pN*     
        return h_bonds

    def get_mpo(self, bc):
        """Generate the TFI Hamiltonian as a matrix product operator."""
        N = self.N
        J = self.J
        g = self.g
        Id = self.Id
        sigma_x = self.sigma_x
        sigma_z = self.sigma_z
        Ws = [None] * N
        if bc == "open":
            # site 1
            W = np.zeros((1, 3, 2, 2))
            W[0, 0] = Id
            W[0, 1] = sigma_x
            W[0, 2] = - g * sigma_z
            Ws[0] = W  # wL0 wR1 p1 p1*
            # sites 2, ..., N-1
            W = np.zeros((3, 3, 2, 2))
            W[0, 0] = Id
            W[0, 1] = sigma_x
            W[0, 2] = - g * sigma_z
            W[1, 2] = - J * sigma_x
            W[2, 2] = Id
            for n in range(1, N-1):
                Ws[n] = W  # wL(n-1) wRn pn pn*
            # site N
            W = np.zeros((3, 1, 2, 2))
            W[0, 0] = - g * sigma_z
            W[1, 0] = - J * sigma_x
            W[2, 0] = Id
            Ws[N-1] = W  # wL(N-1) wRN pN pN*
        elif bc == "periodic":
            # site 1
            W = np.zeros((1, 3, 2, 2))
            W[0, 0] = Id
            W[0, 1] = sigma_x
            W[0, 2] = - g * sigma_z
            Ws[0] = W  # wL0 wR1 p1 p1*
            # site 2
            W = np.zeros((3, 4, 2, 2))
            W[0, 0] = Id
            W[0, 1] = sigma_x
            W[0, 3] = - g * sigma_z
            W[1, 2] = Id
            W[1, 3] = - J * sigma_x
            W[2, 3] = Id
            Ws[1] = W  # wL1 wR2 p2 p2*
            # sites 3, ..., N-2
            W = np.zeros((4, 4, 2, 2))            
            W[0, 0] = Id              
            W[0, 1] = sigma_x
            W[0, 3] = - g * sigma_z
            W[1, 3] = - J * sigma_x
            W[2, 2] = Id
            W[3, 3] = Id
            for n in range(2, N-2):
                Ws[n] = W  # wL(n-1) wRn pn pn*
            # site N-1
            W = np.zeros((4, 3, 2, 2))
            W[0, 0] = Id
            W[0, 1] = sigma_x
            W[0, 2] = - g * sigma_z
            W[1, 2] = - J * sigma_x
            W[2, 1] = Id
            W[3, 2] = Id
            Ws[N-2] = W  # wL(N-2) wR(N-1) p(N-1) p(N-1)*
            # site N
            W = np.zeros((3, 1, 2, 2))
            W[0, 0] = - g * sigma_z
            W[1, 0] = - J * sigma_x
            W[2, 0] = Id
            Ws[N-1] = W  # wL(N-1) wRN pN pN*
        else:
            raise ValueError(f"The boundary conditions (bc) must be either \"open\" or \"periodic\".")
        return Ws
    
    def get_bond_mpo(self, n, m):
        """Generate the TFI Hamiltonian as a matrix product operator."""
        N = self.N
        J = self.J
        g = self.g / 2
        Id = self.Id
        sigma_x = self.sigma_x
        sigma_z = self.sigma_z
        Ws = [None] * N
        # l < n
        for l in range(n):
            W = np.zeros((1, 1, 2, 2))
            W[0, 0] = Id
            Ws[l] = W
        # l = n
        W = np.zeros((1, 3, 2, 2))
        W[0, 0] = Id
        W[0, 1] = sigma_x
        W[0, 2] = - g * sigma_z
        Ws[n] = W 
        # n < l < m
        for l in range(n+1, m):
            W = np.zeros((3, 3, 2, 2))
            W[0, 0] = Id
            W[1, 1] = Id
            W[2, 2] = Id
            Ws[l] = W
        # l = m
        W = np.zeros((3, 1, 2, 2))
        W[0, 0] = - g * sigma_z
        W[1, 0] = - J * sigma_x
        W[2, 0] = Id
        Ws[m] = W
        # l > m
        for l in range(m+1, N):
            W = np.zeros((1, 1, 2, 2))
            W[0, 0] = Id
            Ws[l] = W
        return Ws
    
    # exact analytical results for periodic bc
    """
    By performing Jordan-Wigner, Fourier and Bogoliubov transformations, the TFI model with PBC can 
    be diagonalized analytically. The Hamiltonian in terms of fermionic creation and annihilation 
    operators gamma_{p}^{dagger} and gamma_{p} reads:

    H = (sum_{p} epsilon(p) gamma_{p}^{dagger}gamma_{p}) + E0.

        - Single particle excitation energy: epsilon(p) = 2 sqrt{J^2 - 2Jgcos(p) + g^2}.

        - Ground state energy: E0 = -sum_{p} epsilon(p)/2. 

    For details see Subir Sachdev, Quantum Phase Transitions, 2nd ed, Cambridge University Press, 2011.
    """
    def get_epsilon(self, p):
        """Compute the exact single particle excitation energy for momentum p."""
        return 2 * np.sqrt(self.g**2 - 2 * self.J * self.g * np.cos((2 * np.pi / self.N) * p) + self.J**2)
    
    def get_ps(self):
        """Compute the discrete momenta for a chain of finite length N."""
        N = self.N
        if N % 2 == 0:
            ps = np.arange(-N//2 + 1, N//2 + 1)
        elif N % 2 == 1:
            ps = np.arange(-(N-1)//2, (N-1)//2 + 1)
        return ps

    def get_exact_gs_energy(self):
        """Compute the exact ground state energy E0."""
        ps = self.get_ps()
        ps_apbc = ps - 0.5
        epsilons = self.get_epsilon(ps_apbc)
        E0 = -np.sum(epsilons/2.)
        return E0
    
    def get_exact_excitation_dispersion(self):
        """Compute the exact excitation dispersion relation for discrete and continuous momenta."""
        ps_discrete = self.get_ps()
        es_discrete = self.get_epsilon(ps_discrete)
        N = self.N
        if N % 2 == 0:
            ps_continuous = np.arange(-N//2, N//2 + 0.1, 0.1)
        elif N % 2 == 1:
            ps_continuous = np.arange(-(N-1)//2, (N-1)//2 + 1 + 0.1, 0.1)
        es_continuous = self.get_epsilon(ps_continuous)
        return ps_discrete, es_discrete, ps_continuous, es_continuous