"""Toy code implementing the transverse field Ising (TFI) model on a finite diagonal square lattice
with open boundary conditions."""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm

from tenpy.models import lattice
from tenpy.models import tf_ising
from tenpy.networks.site import SpinHalfSite
from tenpy.tools.params import Config


class DiagonalSquareLattice:
    """Class for the sites and bonds of a diagonal square lattice.

    site vector (x, y, p) with x = 0, ..., Lx - 1 and y = 0, ..., Ly - 1 the horizontal and vertical
    coordinates of the unit cell and p = 0, 1 the diagonal coordinate within the unit cell
    <->
    site scalar 2 * (x * Ly + y) + p = 0, ..., 2 * Lx * Ly - 1

    bond vector (bx, by) with bx = 0, ..., 2 * Lx - 1 and by = 0, ..., 2 * Ly - 1 the horizontal and 
    vertical coordinate of the bond
    <->
    bond scalar (2 * Ly - 1) * bx + by
    
    Parameters
    ----------
    Lx, Ly: Same as attributes.

    Attributes
    ----------
    Lx, Ly: int
            Lengths of the lattice, i.e. number of unit cells in horizontal, vertical direction.
    N_sites, N_bonds: int
                      Total number of sites, bonds in the lattice.
    """
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        self.N_sites = 2 * Lx * Ly
        self.N_bonds = (2 * Lx - 1) * (2 * Ly - 1)

    def get_site_scalar(self, site_vector):
        """Convert a site vector into its corresponding scalar as specified above."""
        x, y, p = site_vector
        return 2 * (x * self.Ly + y) + p
    
    def get_site_vector(self, site_scalar):
        """Convert a site scalar into its corresponding vector as specified above."""
        p = site_scalar % 2
        uc = (site_scalar - p) // 2
        x = uc // self.Ly
        y = uc % self.Ly
        return (x, y, p)
    
    def get_bond_scalar(self, bond_vector):
        """Convert a bond vector into its corresponding scalar as specified above."""
        bx, by = bond_vector
        return (2 * self.Ly - 1) * bx + by
    
    def get_bond_vector(self, bond_scalar):
        """Convert a bond scalar into its corresponding vector as specified above."""
        by = bond_scalar % (2 * self.Ly - 1)
        bx = (bond_scalar - by) // (2 * self.Ly - 1)
        return (bx, by)

    def get_sites(self, bond_vector, order):
        """For a bond given as a vector, return the two associated sites as vectors ordered 
        bottom_top or left_right."""
        bx, by = bond_vector
        x, y = bx//2, by//2
        px, py = bx%2, by%2
        # return (x, y, 1), (x+px, y+py, 0) -> but specify order of the sites
        if order == "bottom_top":
            if py == 0:
                return (x+px, y, 0), (x, y, 1)
            else:
                return (x, y, 1), (x+px, y+1, 0)
        elif order == "left_right":
            if px == 0:
                return (x, y+py, 0), (x, y, 1)
            else:
                return (x, y, 1), (x+1, y+py, 0)
        else:
            raise ValueError(f"The order must be either \"bottom_top\" or \"left_right\".")
            
    def count_nearest_neighbors(self, site_vector):
        """For a site given as a vector, return the number of nearest neighbor sites."""
        x, y, p = site_vector
        if site_vector == (0, 0, 0) or site_vector == (self.Lx-1, self.Ly-1, 1):
            return 1
        elif (x, p) == (0, 0) or (x, p) == (self.Lx-1, 1) \
             or (y, p) == (0, 0) or (y, p) == (self.Ly-1, 1):
            return 2
        else:
            return 4


class TFIModelDiagonalSquare:
    """Class generating different representations of the TFI Hamiltonian on a finite diagonal square 
    lattice with open boundary conditions.
    
    H = -J * sum_<n,m> sigma^{x}_{n} sigma^{x}_{m} - g * sum_{n} sigma^{z}_{n}.

    Parameters
    ----------
    Lx, Ly, J, g: Same as attributes.

    Attributes
    ----------
    Lx, Ly: int
            Lengths of the lattice, i.e. number of unit cells in horizontal, vertical direction.
    lattice: DiagonalSquareLattice
             Lattice instance corresponding to the lattice lengths.
    J, g: float
          Coupling parameters of the above defined TFI Hamiltonian.
    sigma_x, sigma_y, sigma_z, Id: np.array[ndim=2]
                                   Pauli matrices and identity, legs p p*.
    """
    def __init__(self, Lx, Ly, g, J=1.):
        self.Lx = Lx
        self.Ly = Ly
        self.lattice = DiagonalSquareLattice(Lx, Ly)
        self.g = g
        self.J = J
        self.sigma_x = np.array([[0., 1.], [1., 0.]])
        self.sigma_y = np.array([[0., -1.j], [1.j, 0.]])
        self.sigma_z = np.array([[1., 0.], [0., -1.]])
        self.Id = np.eye(2)
    
    def get_H_bonds(self):
        """Generate the TFI two-site Hamiltonians as sparse matrices."""
        Id = sparse.csr_matrix(self.Id)
        sigma_x = sparse.csr_matrix(self.sigma_x)
        sigma_z = sparse.csr_matrix(self.sigma_z)

        def singlesite_to_full(op, site_scalar):
            """For a site operator op acting on site_scalar, generate the full operator acting with
            identities on all other sites."""
            ops = [Id] * self.lattice.N_sites
            ops[site_scalar] = op
            Op = ops[0]
            for m in range(1, self.lattice.N_sites):
                Op = sparse.kron(Op, ops[m], format="csr")
            return Op
        
        H_bonds = [None] * self.lattice.N_bonds
        for bond_scalar in range(self.lattice.N_bonds):
            bond_vector = self.lattice.get_bond_vector(bond_scalar)
            bot_site_vector, top_site_vector = self.lattice.get_sites(bond_vector, order="bottom_top")
            bot_site_scalar = self.lattice.get_site_scalar(bot_site_vector)
            top_site_scalar = self.lattice.get_site_scalar(top_site_vector)
            gb = self.g / self.lattice.count_nearest_neighbors(bot_site_vector)
            gt = self.g / self.lattice.count_nearest_neighbors(top_site_vector)
            XX = singlesite_to_full(sigma_x, bot_site_scalar) @ singlesite_to_full(sigma_x, top_site_scalar)
            ZId = singlesite_to_full(sigma_z, bot_site_scalar) @ singlesite_to_full(Id, top_site_scalar)
            IdZ = singlesite_to_full(Id, bot_site_scalar) @ singlesite_to_full(sigma_z, top_site_scalar)
            H_bonds[bond_scalar] = - self.J * XX - gb * ZId - gt * IdZ
        return H_bonds
    
    def get_H_bonds_uniform(self):
        """Generate the uniform TFI two-site Hamiltonians as sparse matrices."""
        Id = sparse.csr_matrix(self.Id)
        sigma_x = sparse.csr_matrix(self.sigma_x)
        sigma_z = sparse.csr_matrix(self.sigma_z)

        def singlesite_to_full(op, site_scalar):
            """For a site operator op acting on site_scalar, generate the full operator acting with
            identities on all other sites."""
            ops = [Id] * self.lattice.N_sites
            ops[site_scalar] = op
            Op = ops[0]
            for m in range(1, self.lattice.N_sites):
                Op = sparse.kron(Op, ops[m], format="csr")
            return Op
        
        H_bonds = [[None] * (2*self.Ly-1) for _ in range(2*self.Lx-1)]
        for bond_scalar in range(self.lattice.N_bonds):
            bond_vector = self.lattice.get_bond_vector(bond_scalar)
            bot_site_vector, top_site_vector = self.lattice.get_sites(bond_vector, order="bottom_top")
            bot_site_scalar = self.lattice.get_site_scalar(bot_site_vector)
            top_site_scalar = self.lattice.get_site_scalar(top_site_vector)
            XX = singlesite_to_full(sigma_x, bot_site_scalar) @ singlesite_to_full(sigma_x, top_site_scalar)
            ZId = singlesite_to_full(sigma_z, bot_site_scalar) @ singlesite_to_full(Id, top_site_scalar)
            IdZ = singlesite_to_full(Id, bot_site_scalar) @ singlesite_to_full(sigma_z, top_site_scalar)
            bx, by = bond_vector
            H_bonds[bx][by] = - self.J * XX - (self.g / 2) * ZId - (self.g / 2) * IdZ
        return H_bonds
            
    def get_H(self):
        """Generate the full TFI Hamiltonian as a sparse matrix, equal to the sum of all two-site 
        Hamiltonians."""
        H = sparse.csr_matrix((2**self.lattice.N_sites, 2**self.lattice.N_sites))
        H_bonds = self.get_H_bonds()
        for H_bond in H_bonds:
            H += H_bond
        return H

    def get_h_bonds(self):
        """Generate the TFI two-site Hamiltonians as local tensors."""
        h_bonds = [None] * self.lattice.N_bonds
        for bond_scalar in range(self.lattice.N_bonds):
            bond_vector = self.lattice.get_bond_vector(bond_scalar)
            bot_site_vector, top_site_vector = self.lattice.get_sites(bond_vector, order="bottom_top")
            gb = self.g / self.lattice.count_nearest_neighbors(bot_site_vector)
            gt = self.g / self.lattice.count_nearest_neighbors(top_site_vector)
            h = - self.J * np.kron(self.sigma_x, self.sigma_x) \
                - gb * np.kron(self.sigma_z, self.Id) \
                - gt * np.kron(self.Id, self.sigma_z)  # p1.p2 p1*.p2* 
            h_bonds[bond_scalar] = np.reshape(h, (2, 2, 2, 2))  # p1 p2 p1* p2* 
        return h_bonds
    
    def get_h_bonds_uniform(self):
        """Generate the uniform TFI two-site Hamiltonians as local tensors."""
        h_bonds = [None] * self.lattice.N_bonds
        for bond_scalar in range(self.lattice.N_bonds):
            h = - self.J * np.kron(self.sigma_x, self.sigma_x) \
                - (self.g / 2) * np.kron(self.sigma_z, self.Id) \
                - (self.g / 2) * np.kron(self.Id, self.sigma_z)  # p1.p2 p1*.p2* 
            h_bonds[bond_scalar] = np.reshape(h, (2, 2, 2, 2))  # p1 p2 p1* p2* 
        return h_bonds
    
    def get_u_bonds(self, dt):
        """Generate the TFI two-site real time evolution operators for small time step dt."""
        h_bonds = self.get_h_bonds()
        u_bonds = []
        for h_bond in h_bonds:
            h_bond = np.reshape(h_bond, (4, 4))
            u_bond = expm(-dt * h_bond)  # p1.p2 p1*.p2*
            u_bonds.append(np.reshape(u_bond, (2, 2, 2, 2)))  # p1 p2 p1* p2* 
        return u_bonds

    def get_h_mpos(self):
        """Generate the TFI Hamiltonian as a list of 2 * Lx - 1 matrix product operators."""
        Nx_bonds = (2 * self.Lx - 1)
        Ny_sites = 2 * self.Ly

        def get_h_mpo(gs):
            """Generate the TFI mpo for transverse field values gs."""
            assert len(gs) == Ny_sites
            Ws = [None] * Ny_sites
            # site 1
            W = np.zeros((1, 3, 2, 2))
            W[0, 0] = self.Id
            W[0, 1] = self.sigma_x
            W[0, 2] = - gs[0] * self.sigma_z
            Ws[0] = W  # d1 u1 p1 p1*
            # sites 2, ..., Ny
            for n in range(1, Ny_sites-1):
                W = np.zeros((3, 3, 2, 2))
                W[0, 0] = self.Id
                W[0, 1] = self.sigma_x
                W[0, 2] = - gs[n] * self.sigma_z
                W[1, 2] = - self.J * self.sigma_x
                W[2, 2] = self.Id
                Ws[n] = W  # dn un pn pn*
            # site N
            W = np.zeros((3, 1, 2, 2))
            W[0, 0] = - gs[Ny_sites-1] * self.sigma_z
            W[1, 0] = - self.J * self.sigma_x
            W[2, 0] = self.Id
            Ws[Ny_sites-1] = W  # d(N-1) uN pN pN*
            return Ws
        
        if Nx_bonds == 1:
            return [get_h_mpo([self.g] * Ny_sites)]
        
        h_mpos = [None] * Nx_bonds
        # bond column 1
        gs = [self.g/2] * Ny_sites
        for ny in range(Ny_sites):
            if ny%2 == 0:
                gs[ny] = self.g
        h_mpos[0] = get_h_mpo(gs)
        # bond columns 2, ..., Nx_bonds-1
        for nx in range(1, Nx_bonds-1):
            h_mpos[nx] = get_h_mpo([self.g/2] * Ny_sites)
        # bond column Nx_bond
        gs = [self.g/2] * Ny_sites
        for ny in range(Ny_sites):
            if ny%2 == 1:
                gs[ny] = self.g
        h_mpos[Nx_bonds-1] = get_h_mpo(gs)
        return h_mpos   

    def get_H_single_particle(self):
        assert self.g >= 3.0
        t = self.J
        N = 2 * self.Lx * self.Ly
        H = sparse.csr_matrix((N, N))
        for site_scalar in range(N):
            site_vector = self.lattice.get_site_vector(site_scalar)
            mu = - 2 * self.g - self.lattice.count_nearest_neighbors(site_vector) * (self.J**2) / (4*self.g)
            H[site_scalar, site_scalar] = - mu
        for bond_scalar in range(self.lattice.N_bonds):
            bond_vector = self.lattice.get_bond_vector(bond_scalar)
            bot_site_vector, top_site_vector = self.lattice.get_sites(bond_vector, order="bottom_top")
            bot_site_scalar = self.lattice.get_site_scalar(bot_site_vector)
            top_site_scalar = self.lattice.get_site_scalar(top_site_vector)
            H[bot_site_scalar, top_site_scalar] = H[top_site_scalar, bot_site_scalar] = - t
        return H
    
    def get_H_bonds_uniform_single_particle(self):
        assert self.g >= 3.0
        t = self.J
        mu = - 2 * self.g - (self.J**2) / self.g
        N = 2 * self.Lx * self.Ly
        H_bonds = [[None] * (2*self.Ly-1) for _ in range(2*self.Lx-1)]
        for bx in range(2*self.Lx-1):
            for by in range(2*self.Ly-1):
                H_bond = sparse.csr_matrix((N, N))
                bot_site_vector, top_site_vector = self.lattice.get_sites((bx, by), order="bottom_top")
                bot_site_scalar = self.lattice.get_site_scalar(bot_site_vector)
                top_site_scalar = self.lattice.get_site_scalar(top_site_vector)
                H_bond[bot_site_scalar, top_site_scalar] = H_bond[top_site_scalar, bot_site_scalar] = - t
                H_bond[bot_site_scalar, bot_site_scalar] = H_bond[top_site_scalar, top_site_scalar] = - mu/2
                #H_bond[bot_site_scalar, bot_site_scalar] = -mu / self.lattice.count_nearest_neighbors(bot_site_vector)
                #H_bond[top_site_scalar, top_site_scalar] = -mu / self.lattice.count_nearest_neighbors(top_site_vector)
                H_bonds[bx][by] = H_bond
        return H_bonds
    
    def get_es_bond_single_particle(self, k):
        assert self.g >= 3.0
        H = self.get_H_single_particle()
        Es, psis = eigsh(H, k=k, which="SA")
        H_bonds = self.get_H_bonds_uniform_single_particle()
        es_bond_list = []
        for i in range(k):
            psi = psis[:, i]
            es_bond = [[None] * (2*self.Ly-1) for _ in range(2*self.Lx-1)]
            for bx in range(2*self.Lx-1):
                for by in range(2*self.Ly-1):
                    es_bond[bx][by] = np.inner(np.conj(psi), H_bonds[bx][by] @ psi)
            es_bond_list.append(es_bond)
        return es_bond_list, Es


class DiagonalSquareLatticeTenpy(lattice.Lattice):
    """Tenpy class for the diagonal square lattice."""
    dim = 2 
    Lu = 2 
    def __init__(self, Lx, Ly, sites, **kwargs):
        if not hasattr(sites, '__iter__') or isinstance(sites, str):
            sites = [sites] * self.Lu
        elif len(sites) != self.Lu:
            raise ValueError(f"Expected {self.Lu} site(s), got {len(sites)}")
        basis = np.array(([1., 0.], [0., 1.]))
        delta = np.array([0.5, 0.5])
        pos = (-delta / 2., delta / 2)
        NN = [(0, 1, np.array([0, 0])), \
              (1, 0, np.array([1, 0])), \
              (1, 0, np.array([0, 1])), \
              (1, 0, np.array([1, 1]))]
        kwargs.setdefault('basis', basis)
        kwargs.setdefault('positions', pos)
        kwargs.setdefault('pairs', {}).setdefault('nearest_neighbors', NN)
        lattice.Lattice.__init__(self, [Lx, Ly], sites, **kwargs)

    def ordering(self, order):
        """Provide possible orderings of the lattice sites."""
        if isinstance(order, str):
            if order == "down_to_up":
                priority = (0, 2, 1)
                snake_winding = (False, False, False)
                return lattice.get_order(self.shape, snake_winding, priority)
            elif order == "left_to_right":
                priority = (2, 0, 1)
                snake_winding = (False, False, False)
                return lattice.get_order(self.shape, snake_winding, priority) 
            else:
                raise ValueError(f"Unknown ordering: {order}.")
        return super().ordering(order)


class TFIModelDiagonalSquareTenpy(tf_ising.TFIModel):
    """Tenpy class for the TFI model on the diagonal square lattice."""

    default_lattice = DiagonalSquareLatticeTenpy
    force_default_lattice = True

    @classmethod
    def initialize(cls, Lx, Ly, g, order="down_to_up", J=1.):
        """Initialize the TFI model with transverse field g on a finite diagonal square lattice of 
        lengths Lx, Ly and site ordering order. Choose open boundary conditions and do not consider
        symmetries."""
        config = {
            "Lx": Lx, 
            "Ly": Ly,  
            "J": J,  
            "g": g,  
            "bc_x": "open",  
            "bc_y": "open",  
            "conserve": None,  
            "sort_charge": None,
            "order": order
        }
        name = "tfi_model_diagonal_square"
        model_params = Config(config, name)
        return cls(model_params)

    def get_np_mpo(self):
        """Get the TFI MPO and convert it to numpy arrays."""
        Ws = [W.to_ndarray() for W in self.calc_H_MPO()._W]
        Ws[0] = Ws[0][:1, :, :, :]
        Ws[-1] = Ws[-1][:, -1:, :, :]
        return Ws