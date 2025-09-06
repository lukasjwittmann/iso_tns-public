"""Toy code implementing the transverse field Ising (TFI) model on an infinite chain."""

import numpy as np
import scipy


class TFIModelInfinite:
    """Class generating the two-site Hamiltonian and the exact spectrum of the TFI model on an 
    infinite chain.

    H = sum_{n} h_{n,n+1},
    h = - J*(sigma^{x} otimes sigma^{x}) - (g/2)*(sigma^{z} otimes 1) - (g/2)*(1 otimes sigma^{z}).

    Parameters
    ----------
    J, g: Same as attributes.

    Attributes
    ----------
    J, g: float
          Coupling parameters of the above defined TFI Hamiltonian.
    sigma_x, sigma_y, sigma_z, Id: np.array[ndim=2]
                                   Pauli matrices and identity, legs p p*.
    """
    def __init__(self, g, J=1.):
        self.J = J
        self.g = g
        self.sigma_x = np.array([[0., 1.], [1., 0.]])
        self.sigma_y = np.array([[0., -1.j], [1.j, 0.]])
        self.sigma_z = np.array([[1., 0.], [0., -1.]])
        self.Id = np.eye(2)

    def get_h(self):
        """Generate the two-site Hamiltonian h defined above as a tensor with legs p1 p2 p1* p2*."""
        h = - self.J * np.kron(self.sigma_x, self.sigma_x) \
            - (self.g/2) * np.kron(self.sigma_z, self.Id) \
            - (self.g/2) * np.kron(self.Id, self.sigma_z) # p1.p2 p1*.p2*
        h = np.reshape(h, (2, 2, 2, 2))  # p1 p2 p1* p2*
        return h
    
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
        return 2 * np.sqrt(self.g**2 - 2 * self.J * self.g * np.cos(p) + self.J**2)

    def get_exact_gs_energy_density(self):
        """Compute the exact ground state energy density e0 in the thermodynamic limit by replacing
        sum_k -> integral dk/2pi in the above formula for E0."""
        e0_exact = -1 / (2 * np.pi) * scipy.integrate.quad(self.get_epsilon, -np.pi, np.pi)[0]/2
        return e0_exact
    
    def get_exact_excitation_dispersion(self):
        """Compute the exact excitation dispersion relation for the thermodynamic limit with momenta
        between -pi and pi."""
        ps_exact = np.arange(-np.pi, np.pi, 0.01)
        es_exact = self.get_epsilon(ps_exact)
        return ps_exact, es_exact