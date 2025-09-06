"""Toy code implementing a finite boundary/column matrix product state with two "physical" legs, and
a two-site sweep engine along this MPS."""

import numpy as np
import opt_einsum as oe
from functools import reduce

from ..matrix_decompositions import qr_positive, lq_positive, svd_safe, svd_truncation


class MPS:
    """Simple class for a finite boundary/column matrix product state with two "physical" legs.
    
    Parameters
    ----------
    Ms, norm: Same as attributes

    Attributes
    ----------
    Ms: list of np.array[ndim=4]s
        Tensors defining the MPS, with legs d b t u (down bottom top up) for boundary MPS and 
        d l r u (down left right up) for column MPS. 
        Comparing it to a normal MPS defining a physical state, d u correspond to the virtual and 
        b t / l r to the physical legs. In this anology we denote latter as "physical" legs, with
        quotation marks indicating that they are not actual spin degrees of freedom.
        Not necessarily in any canonical form or normalized.
    norm: float
          Positive global norm of the MPS, in product to the (possibly >1) norm contained in the 
          tensors Ms.
    N: int
       Length of the MPS.
    Ds: list of int tuples
        Site dependent dimensions of the "physical" legs.
    """
    def __init__(self, Ms, norm):
        self.N = len(Ms)
        self.Ms = Ms
        self.norm = norm
        self.Ds = [(np.shape(M)[1], np.shape(M)[2]) for M in Ms]
    
    @classmethod
    def from_random_up_isometries(cls, Ds, chi_max, norm=1.):
        """Initialize MPS instance with tensors equal to random isometries from the down leg to the 
        "physical" and up legs, with a (possibly >1) global norm. The "physical" legs have dimensions
        Ds and the maximal bond dimension is chi_max."""
        N = len(Ds)
        Ms = []
        chi_u = 1
        for n in reversed(range(N)):
            D_prod = 1
            for Dl, Dr in Ds[:n]:
                D_prod *= Dl * Dr
                if D_prod > chi_max:
                    D_prod = chi_max
                    break
            Dl, Dr = Ds[n]
            chi_d = min(chi_u * Dr * Dl, D_prod, chi_max)
            A = np.random.normal(size=(chi_d, chi_u * Dr * Dl)) #+ 1.j * np.random.normal(size=(chi_d, chi_u * Dr * Dl))   
            _, Q = lq_positive(A)
            V = np.reshape(Q, (chi_d, Dl, Dr, chi_u))
            VVdagger = np.tensordot(V, np.conj(V), axes=((1, 2, 3), (1, 2, 3)))
            assert np.allclose(VVdagger, np.eye(chi_d))
            Ms.append(V)
            chi_u = chi_d
        return cls(Ms[::-1], norm)
    
    @classmethod
    def from_perturbed_up_isometries(cls, Ds, chi_max, eps, norm=1.):
        """Initialize MPS instance with trivial tensors (M[0, 0, 0, 0] = 1) perturbed with random 
        isometries from the down leg to the "physical" and up legs:

        M ~ (1-eps) * M_trivial + eps * M_random.

        The "physical" legs have dimensions Ds and the maximal bond dimension is chi_max. Initialize
        a (possibly >1) global norm."""
        N = len(Ds)
        Ms = []
        chi_u = 1
        for n in reversed(range(N)):
            D_prod = 1
            for Dl, Dr in Ds[:n]:
                D_prod *= Dl * Dr
                if D_prod > chi_max:
                    D_prod = chi_max
                    break
            Dl, Dr = Ds[n]
            chi_d = min(chi_u * Dr * Dl, D_prod, chi_max)
            A_product = np.zeros((chi_d, Dl, Dr, chi_u))
            A_product[0, 0, 0, 0] = 1.
            A_product = np.reshape(A_product, (chi_d, chi_u * Dr * Dl))
            A_random = np.random.normal(size=(chi_d, chi_u * Dr * Dl)) #+ 1.j * np.random.normal(size=(chi_d, chi_u * Dr * Dl))  
            A = (1-eps) * A_product + eps * A_random 
            _, Q = lq_positive(A)
            V = np.reshape(Q, (chi_d, Dl, Dr, chi_u))
            VVdagger = np.tensordot(V, np.conj(V), axes=((1, 2, 3), (1, 2, 3)))
            assert np.allclose(VVdagger, np.eye(chi_d))
            Ms.append(V)
            chi_u = chi_d
        return cls(Ms[::-1], norm)

    @classmethod
    def from_random_product_state(cls, Ds, norm=1.):
        """Initialize MPS instance with "physical" legs of dimensions Ds in random normalized 
        product states and a (possibly >1) global norm."""
        Ms = []
        for Dl, Dr in Ds:
            M = np.zeros(shape=(1, Dl, Dr, 1))  # dtype=np.complex128
            state = np.random.normal(size=(Dl, Dr)) #+ 1.j * np.random.normal(size=(Dl, Dr)) 
            state /= np.linalg.norm(state)
            M[0, :, :, 0] = state
            Ms.append(M)
        return cls(Ms, norm)
    
    @classmethod
    def from_identity_product_state(cls, Ds, norm=1.):
        """Initialize MPS instance with "physical" legs of dimensions Ds in normalized identity
        product states and a (possibly >1) global norm."""
        Ms = []
        for Dl, Dr in Ds:
            assert Dl == Dr
            M = np.zeros(shape=(1, Dl, Dl, 1)) 
            state = np.eye(Dl) 
            state /= np.linalg.norm(state)
            M[0, :, :, 0] = state
            Ms.append(M)
        return cls(Ms, norm)
    
    def copy(self):
        """Create a copy of the MPS instance."""
        return MPS([M.copy() for M in self.Ms], self.norm)

    def is_up_isometries(self):
        """Test if Ms are isometries from the down leg to the "physical" and up legs."""
        up_isometries = [False] * self.N
        for n in reversed(range(self.N)):
            M = self.Ms[n]
            MMdagger = np.tensordot(M, np.conj(M), axes=((1, 2, 3), (1, 2, 3)))
            chi_d = np.shape(M)[0]
            if n == 0:
                assert chi_d == 1
            up_isometries[n] = np.allclose(MMdagger, np.eye(chi_d))
        return all(up_isometries)
    
    def is_down_isometries(self):
        flipped_mps = MPS([np.transpose(M, (3, 1, 2, 0)) for M in self.Ms[::-1]], 1.)
        return flipped_mps.is_up_isometries()
    
    def to_up_isometries(self):
        """By successive LQ decompositions, bring Ms to isometries from the down leg to the 
        "physical" and up legs, and absorb their norms into the global norm."""
        for n in reversed(range(self.N)):
            M = self.Ms[n]
            chi_d, Dl, Dr, chi_u = np.shape(M)
            chi = min(chi_d, Dl * Dr * chi_u)
            M = np.reshape(M, (chi_d, Dl * Dr * chi_u))
            L, Q = lq_positive(M)
            V = np.reshape(Q, (chi, Dl, Dr, chi_u))
            VVdagger = np.tensordot(V, np.conj(V), axes=((1, 2, 3), (1, 2, 3)))
            assert np.allclose(VVdagger, np.eye(chi))
            assert np.shape(L) == (chi_d, chi)
            self.Ms[n] = V
            if n > 0:
                self.Ms[n-1] = np.tensordot(self.Ms[n-1], L, axes=(3, 0))
            elif n == 0:
                assert chi_d == chi == 1
                self.norm *= np.real_if_close(L[0, 0])
        return
    
    def to_down_isometries(self):
        """By successive QR decompositions, bring Ms to isometries from the up leg to the "physical" 
        and down legs, and absorb their norms into the global norm."""
        for n in range(self.N):
            M = self.Ms[n]
            chi_d, Dl, Dr, chi_u = np.shape(M)
            chi = min(chi_d * Dl * Dr, chi_u)
            M = np.reshape(M, (chi_d * Dl * Dr, chi_u))
            Q, R = qr_positive(M)
            U = np.reshape(Q, (chi_d, Dl, Dr, chi))
            UdaggerU = np.tensordot(np.conj(U), U, axes=((0, 1, 2), (0, 1, 2)))
            assert np.allclose(UdaggerU, np.eye(chi))
            assert np.shape(R) == (chi, chi_u)
            self.Ms[n] = U
            if n < self.N-1:
                self.Ms[n+1] = np.tensordot(R, self.Ms[n+1], axes=(1, 0))
            elif n == self.N-1:
                assert chi == chi_u == 1
                self.norm *= np.real_if_close(R[0, 0])
        return
    
    def get_canonical_form(self):
        """Bring the tensors Ms to canonical form by first bringing them in up isometric form and 
        subsequently performing SVDs from down to up."""
        N = self.N
        mps = self.copy()
        mps.to_up_isometries()
        Vs = mps.Ms  # up orthonormal tensors [V[1], ..., V[N]]
        Us = [None] * N  # down orthonormal tensors [U[1], ..., U[N]]
        Ss = [None] * N  # Schmidt value matrices [S[0], ..., S[N-1]]
        Ss[0] = np.ones((1, 1))
        for n in range(N):
            SV = np.tensordot(Ss[n], Vs[n], axes=(1, 0))
            chi_d, Dl, Dr, chi_u = np.shape(SV)
            SV = np.reshape(SV, (chi_d * Dl * Dr, chi_u))
            U, S, v = svd_safe(SV)
            chi = np.shape(U)[1]
            Us[n] = np.reshape(U, (chi_d, Dl, Dr, chi))
            Vs[n] = np.tensordot(Vs[n], np.conj(v).T, axes=(3, 0))
            if n < N-1:
                Vs[n+1] = np.tensordot(v, Vs[n+1], axes=(1, 0))
                Ss[n+1] = np.diag(S / np.linalg.norm(S))
        return Us, Vs, Ss, self.norm
    
    def get_singular_values(self):
        """Extract Schmidt values from canonical form."""
        _, _, Ss, _ = self.get_canonical_form()
        return [np.diag(S) for S in Ss[1:]]

    def get_overlap(self, mps):
        """Compute the overlap with another mps of same length and "physical" dimensions."""
        assert self.N == mps.N and self.Ds == mps.Ds
        M1s = self.Ms
        M2s = mps.Ms
        overlap = oe.contract("abcd,ebcd->ae", \
                              M1s[-1], np.conj(M2s[-1]))
        for n in reversed(range(mps.N-1)):
            overlap = oe.contract("ab,cdea,fdeb->cf", \
                                  overlap, M1s[n], np.conj(M2s[n]))
        assert np.shape(overlap) == (1, 1)
        overlap = overlap[0, 0] * self.norm * mps.norm
        return np.real_if_close(overlap)

    def get_fidelity(self, mps):
        """Compute the fidelity with another mps of same length and "physical" dimensions."""
        fidelity = np.abs(self.get_overlap(mps))**2 / (self.get_overlap(self) * mps.get_overlap(mps))
        return np.real_if_close(fidelity)
    
    def get_total_norm(self):
        """Compute total norm equal to product of global norm and local norms contained in Ms."""
        return np.sqrt(self.get_overlap(self))
    
    def is_equal(self, mps, tol=1.e-10):
        """Check if self is equal to mps, i.e. if they have fidelity one and the same total norm."""
        if (np.abs(self.get_fidelity(mps) - 1.) < tol) \
           and (np.abs(self.get_total_norm() - mps.get_total_norm()) < tol):
            return True
        else:
            return False
    
    def add(self, mps, alpha, beta):
        """For another mps of same length and "physical" dimensions, initialize the sum 
        |mps_sum> = alpha * |self> + beta * |mps> with block diagonal tensors of increased bond 
        dimensions."""
        assert self.N == mps.N and self.Ds == mps.Ds
        M1s = self.Ms
        M2s = mps.Ms
        Ms = []
        for n in range(self.N):
            M1 = M1s[n]
            M2 = M2s[n]
            chi_d1, Dl, Dr, chi_u1 = np.shape(M1)
            chi_d2, _, _, chi_u2 = np.shape(M2)
            if n == 0:
                M1 = alpha * self.norm * M1
                M2 = beta * mps.norm * M2
                assert chi_d1 == chi_d2 == 1
                M = np.zeros(shape=(1, Dl, Dr, chi_u1 + chi_u2), \
                             dtype=reduce(np.promote_types, [M1.dtype, M2.dtype]))
                M[0, :, :, :chi_u1] = M1[0, :, :, :]
                M[0, :, :, chi_u1:] = M2[0, :, :, :]
            elif n == self.N - 1:
                assert chi_u1 == chi_u2 == 1
                M = np.zeros(shape=(chi_d1 + chi_d2, Dl, Dr, 1), \
                             dtype=reduce(np.promote_types, [M1.dtype, M2.dtype]))
                M[:chi_d1, :, :, 0] = M1[:, :, :, 0]
                M[chi_d1:, :, :, 0] = M2[:, :, :, 0]
            else:
                M = np.zeros(shape=(chi_d1 + chi_d2, Dl, Dr, chi_u1 + chi_u2), \
                             dtype=reduce(np.promote_types, [M1.dtype, M2.dtype]))
                M[:chi_d1, :, :, :chi_u1] = M1
                M[chi_d1:, :, :, chi_u1:] = M2
            Ms.append(M)
        mps_sum = MPS(Ms, norm=1.)
        mps_sum.to_up_isometries()
        return mps_sum
    
    def compress(self, chi_max, eps, N_sweeps):
        """Compress the mps to maximal bond dimension chi_max with N_sweeps two site sweeps, 
        discarding any singular values smaller than eps."""
        mps_compression = MPSCompression(self, chi_max, eps)
        mps_compression.run(N_sweeps)
        self.Ms = mps_compression.psi.Ms
        trunc_errors = np.array(mps_compression.trunc_errors)
        return trunc_errors
            

class Sweep:
    """Simple base class for an engine sweeping forth and back along a tensor network state and 
    locally updating center tensors surrounded by isometries.
    
    Parameters
    ----------
    psi, N_centers: Same as attributes.

    Attributes
    ----------
    psi: tensor network state
         The state to be sweeped through and updated locally.
    N_centers: int
               Number of centers.
    convergence: list of floats
                 Convergence measure to possibly end sweeping.
    """
    def __init__(self, psi, N_centers):
        self.psi = psi
        self.N_centers = N_centers
        self.convergence = [np.inf] * N_centers

    def run(self, N_sweeps, tol=None):
        """Run the engine performing N_sweeps sweeps and (possibly) stop if the convergence measure 
        falls below a tolerance tol."""
        for i in range(N_sweeps):
            self.sweep()
            if tol is not None:
                if np.allclose(np.abs(self.convergence), 0, rtol=tol):
                    print(self.__class__.__name__ \
                          + f" converged after {i+1} sweeps up to tolerance {tol}.")
                    return
        if tol is not None:
            print(self.__class__.__name__ \
                  + f" has not converged after {N_sweeps} sweeps up to tolerance {tol}.")
            return
        else:
            #print(self.__class__.__name__ + f" performed {N_sweeps} sweeps.")
            return

    def sweep(self):
        """Perform one sweep forth and back the state and locally update the center tensors theta 
        (with the help of a guess). To not recalculate parts of the network over and over again, 
        safe environments (i.e. partial contractions) for all centers and update them when updating 
        psi."""
        for n in range(self.N_centers-1):
            theta_guess = self.get_theta_guess(n, sweep_dir="forth")
            theta_updated = self.get_theta_updated(n, theta_guess)
            self.update_psi(n, theta_updated, sweep_dir="forth")
            self.update_Env(n, sweep_dir="forth")
        for n in reversed(range(self.N_centers)):
            if n == self.N_centers-1:
                theta_guess = self.get_theta_guess(n, sweep_dir="forth")
            else:
                theta_guess = self.get_theta_guess(n, sweep_dir="back")
            theta_updated = self.get_theta_updated(n, theta_guess)
            self.update_psi(n, theta_updated, sweep_dir="back")
            if n > 0:
                self.update_Env(n, sweep_dir="back")

    def get_theta_guess(self, n, sweep_dir):
        raise NotImplementedError("Method get_theta_guess() not implemented in base class Sweep.")

    def get_theta_updated(self, n, theta_guess):
        raise NotImplementedError("Method get_theta_updated() not implemented in base class Sweep.")
        
    def update_psi(self, n, theta_updated, sweep_dir):
        raise NotImplementedError("Method update_psi() not implemented in base class Sweep.")
    
    def update_Env(self, n, sweep_dir):
        raise NotImplementedError("Method update_Env() not implemented in base class Sweep.")
        

class TwoSiteSweep(Sweep):
    """Sweep engine for an MPS instance performing local updates of two-site centers theta2 and 
    subsequent splitting and truncation via SVD.
    
    Parameters
    ----------
    mps: MPS
         Boundary/Column matrix product state to be sweeped through and updated locally, assumed to
         be initialized in up isometric form (and kept in this form at any time).
    chi_max, eps: Same as attributes.

    Inherits attributes from parent class Sweep.

    Additional attributes
    ---------------------
    chi_max: int
             Maximum bond dimension, i.e. maximum number of singular values to keep in SVD.
    eps: float
         Discard any singular values smaller than that.
    Us: list of np.array[ndim=4]s
        Tensors of the mps in down isometric form, i.e. isometries from the up to the "physical" and
        down legs.
    Ss: list of np.array[ndim=2]s
        Diagonal zero site matrices containing the normalized Schmidt values.
    trunc_errors: list of floats
                  Local truncation errors for every two-site center.
    """
    def __init__(self, mps, chi_max, eps):
        assert mps.is_up_isometries()
        super().__init__(psi=mps, N_centers=mps.N-1)
        self.Us = [None] * self.N_centers  # [U[1], ..., U[N-1]]
        self.Ss = [None] * self.N_centers  # [S[1], ..., S[N-1]]
        self.chi_max = chi_max
        self.eps = eps
        self.trunc_errors = [0.] * self.N_centers

    def get_theta_guess(self, n, sweep_dir):
        """Return the latest two-site tensor theta2 for center n, from up/down orthonormal tensors 
        and normalized Schmidt matrix down/up of the center for sweep_dir forth/back.

        --(U[n-1])--  --(U[n])-- --(S[n])-- --(V[n+1])--  --(V[n+2])--
             ||           ||                     ||            ||
                                      
              <------------------- updated -------------------->
                      back                       forth
        """
        if sweep_dir == "forth":
            theta2_guess = np.tensordot(self.psi.Ms[n], self.psi.Ms[n+1], axes=(3, 0))
            if n != 0:
                theta2_guess = np.tensordot(self.Ss[n-1], theta2_guess, axes=(1, 0))
        elif sweep_dir == "back":
            theta2_guess = np.tensordot(self.Us[n], self.Us[n+1], axes=(3, 0))
            if n != self.N_centers-1:
                theta2_guess = np.tensordot(theta2_guess, self.Ss[n+1], axes=(5, 0))
        theta2_guess *= self.psi.norm
        return theta2_guess  
    
    def update_psi(self, n, theta2_updated, sweep_dir):
        """From the updated, split and truncated theta2 of center n, extract the new down 
        orthonormal U (for site n), diagonal normalized S (for bond n), and up orthonormal V (for 
        site n+1 in psi)."""
        self.psi.norm = np.linalg.norm(theta2_updated)
        U, S, V, trunc_error = self.split_truncate_theta2(theta2_updated)
        # d(n-1) ln rn un, dn un, dn l(n+1) r(n+1) u(n+1)
        self.Us[n] = U
        self.Ss[n] = S
        self.psi.Ms[n+1] = V
        self.trunc_errors[n] = trunc_error
        if sweep_dir == "back" and n == 0:
            self.psi.Ms[n] = np.tensordot(U, S, axes=(3, 0))
        return
        
    def split_truncate_theta2(self, theta2):
        """Split and truncate the two-site tensor theta2 with singular value decomposition.
                              
            l1   l2                  l1                                              l2
             |    |        SVD        |                                               |
        d0--(theta2)--u2   -->   d0--(U)--u1  <-trunc->  d1--(S)--u1  <-trunc->  d1--(V)--u2    
             |    |                   |                                               |
            r1   r2                  r1                                              r2
        """
        # combine legs
        chi0, Dl1, Dr1, Dl2, Dr2, chi2 = np.shape(theta2)
        theta2 = np.reshape(theta2, (chi0 * Dl1 * Dr1, Dl2 * Dr2 * chi2))
        # perform svd truncation
        U, S, V, trunc_error = svd_truncation(theta2, self.chi_max, self.eps)
        # split legs to get down orthonormal U, diagonal normalized S, up orthonormal V
        chi1 = np.shape(U)[1]
        U = np.reshape(U, (chi0, Dl1, Dr1, chi1))  # d0 l1 r1 u1
        S = np.diag(S / np.linalg.norm(S))  # d1 u1
        V = np.reshape(V, (chi1, Dl2, Dr2, chi2))  # d1 l2 r2 u2
        return U, S, V, trunc_error
    

class MPSCompression(TwoSiteSweep):
    """Two site sweep engine for compressing an mps to (smaller) maximal bond dimension.
    
    The state we sweep along is a new mps' which for every center minimizes the cost function 

    ||mps' - mps||^2 = <mps'*|mps'> - <mps'*|mps> - <mps*|mps'> + <mps|mps>
                     = <theta2'*|theta2'> - <mps'*|mps> + ... .

    The solution is given by the effective two site center |theta2'> = d(theta2'*)<mps'*|mps>, 
    resulting from omitting theta2'* in the overlap <mps'*|mps>.

    Parameters
    ----------
    mps: MPS
         Boundary/column matrix product state to be compressed, no specific canonical form is 
         required.
    chi_max, eps: Same as attributes.

    Inherits attributes from parent class TwoSiteSweep.

    Additional attributes
    ---------------------
    Ms: list of np.array[ndim=4]s
        Tensors of the mps to be compressed, kept fix.
    norm: float
          Positive global norm of the mps to be compressed, kept fix.
    DPs, UPs: list of np.array[ndim=2]s
              Down/Up Parts of the effective two site center.
    """
    def __init__(self, mps, chi_max, eps):
        mps_guess = MPS.from_random_up_isometries(mps.Ds, chi_max, mps.norm)
        super().__init__(mps_guess, chi_max, eps)
        self.Ms = mps.Ms
        self.norm = mps.norm
        self.DPs = [None] * self.N_centers
        self.DPs[0] = np.ones(shape=(1, 1))
        self.UPs = [None] * self.N_centers
        self.UPs[-1] = np.ones(shape=(1, 1))
        for n in reversed(range(1, self.N_centers)):
            self.update_Env(n, sweep_dir="back")

    def get_theta_updated(self, n, theta2_guess):
        """For site n, compute the new effective two site center theta2 by contracting M[n], M[n+1]
        with the environments DP[n], UP[n] and multiplying with the global norm. The local 
        convergence measure is chosen as the norm of the difference between updated and previous 
        theta2.

                                 .----(M[n])---(M[n+1])----.
                                 |      ||        ||       |
        ---(theta2[n])---  =  (DP[n])                   (UP[n]) * norm
             ||    ||            |                         |
                                 .----                 ----.

        """
        theta2_updated = oe.contract("ab,acde,efgh,hi->bcdfgi", \
                                     self.DPs[n], self.Ms[n], self.Ms[n+1], self.UPs[n])
        theta2_updated *= self.norm
        self.convergence[n] = np.linalg.norm(theta2_updated - theta2_guess)
        return theta2_updated

    def update_Env(self, n, sweep_dir):
        """For center n and sweep_dir forth/back, update the environment DP[n+1]/UP[n-1] from 
        updated U[n]/V[n+1]. 

            .-----         .----(M[n])---     -----.         ---(M[n+1])----.
            |              |      ||               |               ||       |
        (DP[n+1])    =  (DP[n])   ||     ,     (UP[n-1])  =        ||    (UP[n])
            |              |      ||               |               ||       |
            .-----         .----(U[n]*)--     -----.         --(V[n+1]*)----.

            ---------------------------->     <------------------------------
                        forth                              back               
        """
        if sweep_dir == "forth":
            U_updated = self.Us[n]
            DP_updated = oe.contract("ab,acde,bcdf->ef", \
                                     self.DPs[n], self.Ms[n], np.conj(U_updated))
            self.DPs[n+1] = DP_updated
            if n >= 2:
                self.UPs[n-2] = None
            return
        elif sweep_dir == "back" and n > 0:
            V_updated = self.psi.Ms[n+1]
            UP_updated = oe.contract("ab,cdea,fdeb->cf", \
                                     self.UPs[n], self.Ms[n+1], np.conj(V_updated))
            self.UPs[n-1] = UP_updated
            if n <= self.N_centers-3:
                self.DPs[n+2] = None
            return