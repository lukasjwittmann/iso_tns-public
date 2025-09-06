"""Toy code implementing the density matrix renormalization group squared (DMRG^2) algorithm over a 
diagonal isometric PEPS."""

import numpy as np

from .c_mps import MPS, Sweep
from .d_expectation_values import get_expectation_value_sum, \
                                  get_flipped_As, get_flipped_hs, get_flipped_Cs, get_flipped_mps
from .e_boundary_compression import BoundaryCompression, get_compressed_boundaries
from .f_column_dmrg import ColumnDMRG      


class DMRGSquared(Sweep):
    """Sweep engine for performing the density matrix renormalization group squared (DMRG^2) 
    algorithm over an isometric peps on a diagonal square lattice, for a Hamiltonian H being the sum 
    of bond column mpos h. The "zero-site" orthogonality columns of the isometric peps are chosen as 
    the sweep centers.

    The state we sweep along is an isometric peps which for every center minimizes the energy
     
    E = <iso_peps*|H|iso_peps> = sum_c <iso_peps*|h_c|iso_peps> = <c_mps*|H_eff|c_mps>,

    H_eff = Lh + Ch + Rh,

    where the effective Hamiltonians Lh/Rh are compressed boundary mps resulting from hs already 
    applied to the left/right, and Ch results from h being applied in zig-zag pattern to the 
    isometric site tensors ALs and ARs belonging to the orthogonality column. The solution is given 
    by the ground state column mps of H_eff.

    Parameters
    ----------
    iso_peps, h_mpos, chi_max_b, N_sweeps_c, N_sweeps_b: Same as attributes.

    Inherits attributes from parent class Sweep.

    Additional Attributes
    ---------------------
    h_mpos: list of MPOs / list of lists of np.array[ndim=4]s
            The Hamiltonian of which the ground state is searched.
    Es: list of floats
        All energies E = <Lh|CC*> + <ALhAL*|CC*|ARhAR*> + <CC*|Rh> before ColumnDMRG.
    Es_updated: list of floats
                All ground state energies from ColumnDMRG.
    N_sweeps_c = int
                 Number of sweeps for the ColumnDMRG.
    chi_max_b: int
               Maximal bond dimension for the boundary compression.
    N_sweeps_b: int or None
                If not None, number of sweeps for the variational BoundaryCompression.
                If None, BoundaryColumnCompression is used for the boundary compression.
    bc: string
        "variational"/"column" if BoundaryCompression/BoundaryColumnCompression is used.
    Lhs, Rhs: list of BoundaryMPS instances
              Latest left/right boundary mps.
    """
    def __init__(self, iso_peps, h_mpos, chi_max_b, N_sweeps_c, N_sweeps_b):
        assert (iso_peps.ortho_surface, iso_peps.ortho_center) == (-1, 2*iso_peps.Ly-2)
        super().__init__(psi=iso_peps, N_centers=2*iso_peps.Lx+1)
        self.h_mpos = [None] + h_mpos + [None]
        self.Es = []
        self.Es_updated = []
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

    def get_theta_guess(self, n, sweep_dir):
        """For sweep center n odd/even, take the to be updated orthogonality column with 
        orthogonality center down/up."""
        if n%2 == 1:
            assert (self.psi.ortho_surface, self.psi.ortho_center) == (n-1, 0)
        elif n%2 == 0:
            assert (self.psi.ortho_surface, self.psi.ortho_center) == (n-1, 2*self.psi.Ly-2) 
        Cs_guess = self.psi.get_Cs(n)
        return Cs_guess

    def get_theta_updated(self, n, Cs_guess):
        """For sweep center n, update the orthogonality column by finding the ground state of the 
        three effective Hamiltonians with ColumnDMRG (initialized with Cs_guess). Note that for even 
        n all tensors have to be flipped to achieve the conventional form in ColumnDMRG."""
        print(f"Orthogonality column {n}:")
        # print leg dimensions before update
        print(f"- C leg dimensions: " \
              + f"{[np.shape(C) for C in Cs_guess]}.")
        if self.Lhs[n] is not None:
            print(f"- Lh leg dimensions: " \
                  + f"{[np.shape(Lh) for Lh in self.Lhs[n].Ms]}.")
        if self.Rhs[n] is not None:
            print(f"- Rh leg dimensions: " \
                  + f"{[np.shape(Rh) for Rh in self.Rhs[n].Ms]}.")
        # compute, print and save energy before update
        if n%2 == 1:
            E, EL, EC, ER = get_expectation_value_sum(Cs_guess, \
                                                      self.Lhs[n], self.Rhs[n], \
                                                      self.psi.get_ALs(n), self.psi.get_ARs(n), \
                                                      self.h_mpos[n])
        elif n%2 == 0:
            E, EL, EC, ER = get_expectation_value_sum(get_flipped_Cs(Cs_guess), \
                                                      get_flipped_mps(self.Lhs[n]), \
                                                      get_flipped_mps(self.Rhs[n]), \
                                                      get_flipped_As(self.psi.get_ALs(n)), \
                                                      get_flipped_As(self.psi.get_ARs(n)), \
                                                      get_flipped_hs(self.h_mpos[n]))
        print(f"-> E = <Lh|CC*> + <ALhAL*|CC*|ARhAR*> + <CC*|Rh> = {EL} + {EC} + {ER} = {E}.")
        self.Es.append(E)
        # update C with DMRG
        if n%2 == 1:
            column_dmrg = ColumnDMRG(MPS(Cs_guess, norm=1.), \
                                     self.Lhs[n], self.Rhs[n], \
                                     self.psi.get_ALs(n), self.psi.get_ARs(n), self.h_mpos[n], \
                                     self.psi.chi_max)
        elif n%2 == 0:
            column_dmrg = ColumnDMRG(get_flipped_mps(MPS(Cs_guess, norm=1.)), \
                                     get_flipped_mps(self.Lhs[n]), \
                                     get_flipped_mps(self.Rhs[n]), \
                                     get_flipped_As(self.psi.get_ALs(n)), \
                                     get_flipped_As(self.psi.get_ARs(n)), \
                                     get_flipped_hs(self.h_mpos[n]), \
                                     self.psi.chi_max)
        column_dmrg.run(self.N_sweeps_c)
        mps_updated = column_dmrg.psi
        if n%2 == 0:
            mps_updated = get_flipped_mps(mps_updated)
        Cs_updated = mps_updated.Ms
        # print leg dimensions after update
        print(f"- C_updated leg dimensions: " \
              + f"{[np.shape(C) for C in Cs_updated]}.")
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
        return Cs_updated
    
    def update_psi(self, n, Cs_updated, sweep_dir):
        """Put the updated orthogonality column Cs_updated of sweep center n back into the iso_peps
        and move it to the right/left for sweep_dir forth/back, using the Yang-Baxter move. In the 
        case of odd n, already considering the necessary flipping for the next update on even n, 
        move the orthogonality center up in Yang-Baxter."""
        self.psi.Ws = [np.transpose(C, (1, 3, 2, 0)) for C in Cs_updated]
        if sweep_dir == "forth":  # left -> right
            if n%2 == 1:
                self.psi.move_ortho_surface_right(min_dims=True, force=True, move_upwards=True)
            elif n%2 == 0:
                self.psi.move_ortho_surface_right(min_dims=True, force=True, move_upwards=False)
        elif sweep_dir == "back" and n > 0:  # right -> left
            if n%2 == 1:
                self.psi.move_ortho_surface_left(min_dims=True, force=True, move_upwards=True)
            elif n%2 == 0:
                self.psi.move_ortho_surface_left(min_dims=True, force=True, move_upwards=False)
        return
    
    def init_Env(self):
        """Initialize the right compressed environment(s)

        - Rh[n] for all n > 0 if variational BoundaryCompression,
        - Rh[0] if BoundaryColumnCompression.
        """
        if self.bc == "variational":
            for n in reversed(range(1, self.N_centers)):
                self.update_Env(n, sweep_dir="back")
        elif self.bc == "column":
            _, Rhs, _, _ = get_compressed_boundaries(self.psi, self.h_mpos[1:-1], self.chi_max_b, \
                                                     N_sweeps_b=None, combine_hs=True)
            self.Rhs[0] = Rhs[-1]
        return

    def update_Env(self, n, sweep_dir):
        """After DMRG update (and Yang-Baxter move) at sweep center n and sweep_dir forth/back, 
        update the environments 

        - Lh[n+1]/Rh[n-1] from updated ALs[n+1]/ARs[n-1] if variational BoundaryCompression,
        - Lh[n+1],Rh[n+1]/Lh[n-1],Rh[n-1] from updated global state if BoundaryColumnCompression.
        """
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
                ARs_updated = self.psi.get_ARs(n-1)
                if n%2 == 1:
                    boundary_compression = BoundaryCompression(get_flipped_As(self.psi.get_ARs(n)), \
                                                               get_flipped_As(ARs_updated), \
                                                               get_flipped_hs(self.h_mpos[n]), \
                                                               get_flipped_mps(self.Rhs[n]), \
                                                               self.chi_max_b)                   
                elif n%2 == 0:
                    boundary_compression = BoundaryCompression(self.psi.get_ARs(n), \
                                                               ARs_updated, \
                                                               self.h_mpos[n], \
                                                               self.Rhs[n], \
                                                               self.chi_max_b)                 
                boundary_compression.run(self.N_sweeps_b)
                Rh_updated = boundary_compression.psi
                if n%2 == 1:
                    Rh_updated = get_flipped_mps(Rh_updated)
                self.Rhs[n-1] = Rh_updated
            return
        elif self.bc == "column":
            Lhs, Rhs, _, _ = get_compressed_boundaries(self.psi, self.h_mpos[1:-1], self.chi_max_b, \
                                                       N_sweeps_b=None, combine_hs=True)
            if sweep_dir == "forth":
                if Lhs is not None:
                    self.Lhs[n+1] = Lhs[-1]
                if Rhs is not None:
                    self.Rhs[n+1] = Rhs[-1]
            if sweep_dir == "back":
                if Lhs is not None:
                    self.Lhs[n-1] = Lhs[-1]
                if Rhs is not None:
                    self.Rhs[n-1] = Rhs[-1]