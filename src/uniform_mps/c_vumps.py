"""Toy code implementing the variational uniform matrix product states (VUMPS) algorithm."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, eigsh
from functools import reduce

from ..matrix_decompositions import polar_right, polar_left
from .a_umps import UniformMPS, TransferMatrix


def vumps_algorithm(h, guess_umps0, tol, maxruns):
    """Find the uMPS ground state of the Hamiltonian h with an initial guess up to tolerance tol, 
    perform maximally maxruns vumps updates."""
    vumps_engine = VUMPSEngine(guess_umps0, h)
    i = 0
    for _ in range(maxruns):
        i += 1
        vumps_engine.run()
        if vumps_engine.err <= tol:
            print(f"uMPS ground state converged with VUMPS up to tol = {tol} in gradient norm.")
            break
    umps0 = vumps_engine.umps
    e0 = umps0.get_bond_expectation_value(h)
    var0 = vumps_engine.get_energy_variance()
    if vumps_engine.err > tol:
        print(f"uMPS ground state did not converge with VUMPS up to tol = {tol}.")
    print(f"Final error after {i} iterations: {vumps_engine.err}. \n" \
          + f"Ground state energy density: {e0}. \n" \
          + f"Ground state variance density: {var0}.")
    return e0, umps0, var0


class VUMPSEngine:
    """Simple class for the VUMPS engine to perform the variational ground state optimization for h.

    For P_A the tangent-space projector and H = sum_n h_{n,n+1}, the tangent-space gradient reads
    P_A * H|psi(A)> = |psi(G,A)> with G = Heff1(AC) - AL Heff0(C) = Heff1(AC) - Heff0(C) AR.
    The variational ground state optimum (corresponding to G=0) in the uMPS manifold satisfies
        1) AC ground state of Heff1,
        2) C ground state of Heff0,
        3) AC = AL C = C AR.
    
    Parameters
    ----------
    umps, h, tol: Same as attributes.

    Attributes
    ----------
    umps: UniformMPS
         The current state to be iteratively optimized towards the ground state.
    h: np.array[ndim=4]
       The two-site Hamiltonian of which the ground state is searched.
    tol: float
         Tolerance up to which the geometric sum environments Lh and Rh are computed with gmres.
    Lh: np.array[ndim=2]
        Left environment computed from geometric sum of transfer matrix TL.
    Rh: np.array[ndim=2]
        Right environment computed from geometric sum of transfer matrix TR.
    err: float
         The error measure for umps, equal to the gradient norm ||Heff_AC(AC) - AL Heff_C(C)||.
    """
    def __init__(self, umps, h, tol=1.e-14):
        self.umps = umps.copy()
        self.h = subtract_energy_offset(self.umps, h, canonical_form=False)
        self.tol = tol
        self.Lh = get_Lh(self.umps, self.h, canonical_form=False, guess=None, tol=self.tol)
        self.Rh = get_Rh(self.umps, self.h, canonical_form=False, guess=None, tol=self.tol)
        self.err = self.get_gradient_norm()
        
    def run(self):
        """Perform one update of self.umps in the variational ground state optimization for self.h.
        
        1) AC -> ground state of Heff1,
        2) C -> ground state of Heff0,
        3) AL/AR from left/right polar decompositions of AC and C.
        """
        H_eff_1 = Heff1(self.h, self.Lh, self.umps.AL, self.umps.AR, self.Rh)
        H_eff_0 = Heff0(self.h, self.Lh, self.umps.AL, self.umps.AR, self.Rh)
        AC_new = self.get_theta_gs(Heff=H_eff_1, guess=self.umps.AC)  
        C_new = self.get_theta_gs(Heff=H_eff_0, guess=self.umps.C)  
        AL_new, AR_new = get_AL_AR(AC_new, C_new)  
        self.umps = UniformMPS(AL_new, AR_new, AC_new, C_new)
        self.h = subtract_energy_offset(self.umps, self.h, canonical_form=False)
        self.Lh = get_Lh(self.umps, self.h, canonical_form=False, guess=self.Lh, tol=self.tol)
        self.Rh = get_Rh(self.umps, self.h, canonical_form=False, guess=self.Rh, tol=self.tol)
        self.err = self.get_gradient_norm()

    @staticmethod
    def get_theta_gs(Heff, guess):
        """Find the ground state of Heff with an initial guess."""
        guess = np.reshape(guess, Heff.shape[1])
        _, theta_gs = eigsh(Heff, k=1, which="SA", v0=guess)
        theta_gs = np.reshape(theta_gs[:, 0], Heff.shape_theta)
        return theta_gs

    def get_gradient_norm(self):
        """Compute the gradient norm ||Heff1(AC) - AL Heff0(C)|| for self.umps."""
        H_eff_1 = Heff1(self.h, self.Lh, self.umps.AL, self.umps.AR, self.Rh)
        H_eff_0 = Heff0(self.h, self.Lh, self.umps.AL, self.umps.AR, self.Rh)
        AC = H_eff_1._matvec(np.reshape(self.umps.AC, H_eff_1.shape[1]))  # vL.p.vR
        AC = np.reshape(AC, H_eff_1.shape_theta)  # vL p vR
        C = H_eff_0._matvec(np.reshape(self.umps.C, H_eff_0.shape[1]))  # vL.vR
        C = np.reshape(C, H_eff_0.shape_theta)  # vL vR
        ALC = np.tensordot(self.umps.AL, C, axes=(2, 0))  # vL p [vR], [vL] vR
        gradient_norm = np.linalg.norm(AC - ALC)
        return gradient_norm
    
    def get_energy_variance(self):
        """For the Hamiltonian H = sum_n h_{n,n+1}, compute the energy variance density of self.umps.

        var(H) = <umps|H^2|umps> - <umps|H|umps>^2 = <umps|(H-<umps|H|umps>)^2|umps>.

        Diagrammatic representation for the variance density (after h -> h - <umps|h|umps>):

        .----(AC)--(AR)--.     .--(AL)--(AC)--(AR)--.     .--(AC)--(AR)--.     .--(AC)--(AR)--(AR)--.     
        |     |     |    |     |   |     |     |    |     |   |     |    |     |   |     |     |    |     
        |   (----h----)  |     |   |   (----h----)  |     | (----h----)  |     | (----h----)   |    |     
        |     |     |    |  +  |   |     |     |    |  +  |   |     |    |  +  |   |     |     |    |   
       (Lh)   |     |    |     | (----h----)   |    |     | (----h----)  |     |   |   (----h----)  |     
        |     |     |    |     |   |     |     |    |     |   |     |    |     |   |     |     |    |      
        .---(AC*)--(AR*)-.     .-(AL*)-(AC*)--(AR*)-.     .-(AC*)--(AR*)-.     .-(AC*)--(AR*)-(AR*)-.     

            .--(AC)--(AR)----.
            |   |     |      |
            | (----h----)    |
        +   |   |     |      |
            |   |     |    (Rh)
            |   |     |      |
            .-(AC*)--(AR*)---.
        """
        h = self.h
        Lh = self.Lh
        AL = self.umps.AL
        AC = self.umps.AC
        AR = self.umps.AR
        Rh = self.Rh
        ACAR = np.tensordot(AC, AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR  -> vL p1 p2 vR
        ACAR_h = np.tensordot(ACAR, h, axes=((1, 2), (2, 3)))  # vL [p1] [p2] vR, p1 p2 [p1*] [p2*]
        ACAR_h = np.transpose(ACAR_h, (0, 2, 3, 1))  # vL p1 p2 vR
        var1 = np.tensordot(ACAR_h, np.conj(ACAR), axes=((1, 2, 3), (1, 2, 3)))  
        # vL [p1] [p2] [vR], vL* [p1*] [p2*] [vR*]
        var1 = np.tensordot(Lh, var1, axes=((0, 1), (0, 1)))  # [vR] [vR*], [vL] [vL*]
        var2 = np.tensordot(AL, ACAR_h, axes=(2, 0))  # vL p1 [vR], [vL] p2 p3 vR
        var2 = np.tensordot(h, var2, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] p3 vR
        var2 = np.tensordot(var2, np.conj(AL), axes=((0, 2), (1, 0)))  
        # [p1]  p2 [vL] p3 vR, [vL*] [p1*] vR*
        var2 = np.tensordot(var2, np.conj(ACAR), axes=((3, 0, 1, 2), (0, 1, 2, 3)))  
        # [p2] [p3] [vR] [vR*], [vL*] [p2*] [p3*] [vR*]
        var3 = np.tensordot(h, ACAR_h, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        var3 = np.tensordot(var3, np.conj(ACAR), axes=((2, 0, 1, 3), (0, 1, 2, 3)))  
        # [p1] [p2] [vL] [vR], [vL*] [p1*] [p2*] [vR*]
        var4 = np.tensordot(ACAR_h, AR, axes=(3, 0))  # vL p1 p2 [vR], [vL] p3 vR
        var4 = np.tensordot(h, var4, axes=((2, 3), (2, 3)))  # p2 p3 [p2*] [p3*], vL p1 [p2] [p3] vR
        var4 = np.tensordot(var4, np.conj(ACAR), axes=((2, 3, 0), (0, 1, 2)))  
        # [p2] p3 [vL] [p1] vR, [vL*] [p1*] [p2*] vR*
        var4 = np.tensordot(var4, np.conj(AR), axes=((2, 0, 1), (0, 1, 2)))  
        # [p3] [vR] [vR*], [vL*] [p3*] [vR*]
        var5 = np.tensordot(ACAR_h, np.conj(ACAR), axes=((0, 1, 2), (0, 1, 2)))  
        # [vL] [p1] [p2] vR, [vL*] [p1*] [p2*] vR*
        var5 = np.tensordot(var5, Rh, axes=((0, 1), (0, 1)))  # [vR] [vR*], [vL] [vL*]
        var = var1 + var2 + var3 + var4 + var5
        return var
    

class Heff1(LinearOperator):
    """Class for the effective Hamiltonian acting on the center site tensor AC.
    
                            .---(AC)--.     .--(AL)--(AC)--.     .--(AC)--(AR)--.     .--(AC)---.
                            |    |    |     |   |     |    |     |   |     |    |     |   |     |
    matvec:  --(AC)--  ->  (Lh)  |    |  +  | (----h----)  |  +  | (----h----)  |  +  |   |   (Rh)
                |           |    |    |     |   |     |    |     |   |     |    |     |   |     |
                            .---    --.     .-(AL*)--    --.     .--   --(AR*)--.     .--    ---.
    """
    def __init__(self, h, Lh, AL, AR, Rh):
        self.h = h  # p1 p2 p1* p2*
        self.Lh = Lh  # vR vR*
        self.AL = AL  # vL p vR
        self.AR = AR  # vL p vR
        self.Rh = Rh  # vL vL*
        D = np.shape(self.AL)[0]
        d = np.shape(self.AL)[1]
        self.shape = (D * d * D, D * d * D)
        self.shape_theta = (D, d, D)
        self.dtype = reduce(np.promote_types, [h.dtype, Lh.dtype, AL.dtype, AR.dtype, Rh.dtype])
        
    def _matvec(self, AC):
        AC = np.reshape(AC, self.shape_theta)  # vL p vR
        AC1 = np.tensordot(self.Lh, AC, axes=(0, 0))  # [vR] vR*, [vL] p vR -> vL p vR
        AC2 = np.tensordot(self.AL, AC, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        AC2 = np.tensordot(self.h, AC2, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        AC2 = np.tensordot(np.conj(self.AL), AC2, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR -> vL p vR  
        AC3 = np.tensordot(AC, self.AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        AC3 = np.tensordot(self.h, AC3, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        AC3 = np.tensordot(np.conj(self.AR), AC3, axes=((1, 2), (1, 3)))
        # vL* [p2*] [vR*], p1 [p2] vL [vR] -> vR p vL
        AC3 = np.transpose(AC3, (2, 1, 0))  # vL p vR                        
        AC4 = np.tensordot(AC, self.Rh, axes=(2, 0))  # vL p [vR], [vL] vL* -> vL p vR
        AC_new = AC1 + AC2 + AC3 + AC4
        AC_new = np.reshape(AC_new, self.shape[1])  # vL.p.vR
        return AC_new


class Heff0(LinearOperator):
    """Class for the effective Hamiltonian acting on the center matrix C.

                           .---(C)--.     .--(AL)--(C)--(AR)--.     .--(C)---.
                           |        |     |   |          |    |     |        |
    matvec:  --(C)--  ->  (Lh)      |  +  | (------h-------)  |  +  |      (Rh)
                           |        |     |   |          |    |     |        |
                           .---   --.     .-(AL*)--   --(AR*)-.     .--   ---.
    """                    
    def __init__(self, h, Lh, AL, AR, Rh):
        self.h = h  # p1 p2 p1* p2*
        self.Lh = Lh  # vR vR*
        self.AL = AL  # vL p vR
        self.AR = AR  # vL p vR
        self.Rh = Rh  # vL vL*
        D = np.shape(self.AL)[0]
        self.shape = (D * D, D * D)
        self.shape_theta = (D, D)
        self.dtype = reduce(np.promote_types, [h.dtype, Lh.dtype, AL.dtype, AR.dtype, Rh.dtype])
        
    def _matvec(self, C):
        C = np.reshape(C, self.shape_theta)  # vL vR
        C1 = np.tensordot(self.Lh, C, axes=(0, 0))  # [vR] vR*, [vL] vR -> vL vR
        C2 = np.tensordot(self.AL, C, axes=(2, 0))  # vL p1 [vR], [vL] vR
        C2 = np.tensordot(C2, self.AR, axes=(2, 0))  # vL p1 [vR], [vL] p2 vR
        C2 = np.tensordot(self.h, C2, axes=((2, 3), (1, 2)))  # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
        C2 = np.tensordot(np.conj(self.AL), C2, axes=((0, 1), (2, 0)))  
        # [vL*] [p1*] vR*, [p1] p2 [vL] vR
        C2 = np.tensordot(C2, np.conj(self.AR), axes=((1, 2), (1, 2)))  
        # vR* [p2] [vR], vL* [p2*] [vR*] -> vL vR
        C3 = np.tensordot(C, self.Rh, axes=(1, 0))  # vL [vR], [vL] vL* -> vL vR
        C_new = C1 + C2 + C3
        C_new = np.reshape(C_new, self.shape[1])  # vL.vR
        return C_new
    
    
class InverseGeometricSum(LinearOperator):
    """Class for the inverse of the geometric sum of a transfer matrix T_AB with leading right/left
    eigenvector |R>/<L|.
    
    sum_{n=0}^{infty} (alpha * T_AB)^n = [1 - (alpha * T_AB) + (alpha * |R><L|)]^{-1}  (pseudo=True)
                                       = [1 - (alpha * T_AB)]^{-1}  (pseudo=False)

    The case pseudo=True applies for transfer matrices T_AB with leading eigenvalue 1. We assume 
    that the term proportional to |N||R><L| cancels out after multiplication to a vector on the
    right/left. The case pseudo=False applies for transfer matrices T_AB with spectral radius 
    strictly smaller than 1. 
                                                                            
                                 ---.       ---.            ---(A)---.             ---.   .----.    
                                    |          |                |    |                |   |    |      
    matvec for transpose=False:    (X)  ->    (X)  -  alpha     |   (X)  [+  alpha   (R) (L)  (X)]    
                                    |          |                |    |                |   |    | 
                                 ---.       ---.            ---(B*)--.             ---.   .----.  
                                                                                        
                                                                         [iff pseudo=True]

                                 .---        .---           .---(A)---             .----.   .---
                                 |           |              |    |                 |    |   |
    matvec for transpose=True:  (X)    ->   (X)   -  alpha (X)   |      [+  alpha (X)  (R) (L)  ]
                                 |           |              |    |                 |    |   |
                                 .---        .---           .--(B*)---             .----.   .---
    """
    def __init__(self, A, B, R, L, transpose=False, alpha=1., pseudo=True):
        D = np.shape(A)[0]
        self.T_AB = TransferMatrix([A], [B], transpose=transpose)
        if pseudo:
            self.R = np.reshape(R, (D * D))
            self.L = np.reshape(L, (D * D))
        self.alpha = alpha
        self.transpose = transpose
        self.pseudo = pseudo
        self.shape = (D * D, D * D)
        self.shape_X = (D, D)
        self.dtype = reduce(np.promote_types, [A.dtype, B.dtype])

    def _matvec(self, X):
        if not self.transpose:
            X1 = X
            X2 = -self.alpha * self.T_AB._matvec(X)
            if self.pseudo:
                X2 += self.alpha * self.R * np.inner(self.L, X)
            return X1 + X2
        else:
            X1 = X
            X2 = -self.alpha * self.T_AB._matvec(X)
            if self.pseudo:
                X2 += np.inner(X, self.R) * self.L
            return X1 + X2
            
    def multiply_geometric_sum(self, b, guess, tol):
        """Solve the linear equation self|X> = |b>."""
        b = np.reshape(b, self.shape[1])
        if guess is not None:
            guess = np.reshape(guess, self.shape[1])
        X = gmres(self, b, x0=guess, rtol=tol, atol=0)[0]
        X = np.reshape(X, self.shape_X)  # vL vL* or vR vR*
        return X

    
def get_Lh(umps, h, canonical_form, guess, tol):
    """Compute left environment Lh from geometric sum of transfer matrix TL (umps not necessarily in 
    canonical form).

     .---     .--(AL)--(AL)---                   ---(AL)---^n
     |        |   |     |                            |
    (Lh)   =  | (----h----)    sum_{n=0}^{infty}     | 
     |        |   |     |                            |
     .---     .-(AL*)-(AL*)---                   --(AL*)---
    """
    AL = umps.AL  # vL p vR
    D = np.shape(AL)[0]
    C = umps.C
    R = np.matmul(C, np.conj(C).T)  # vL vL*
    if not canonical_form:
        TL = TransferMatrix([AL], [AL])
        _, R = TL.get_leading_eigenpairs(k=1, guess=R)
        R /= np.trace(R)  # vL vL*   
    IGS = InverseGeometricSum(AL, AL, R=R, L=np.eye(D), transpose=True)
    theta2 = np.tensordot(AL, AL, axes=(2, 0))  # vL p1 p2 vR
    lh = np.tensordot(h, theta2, axes=((2, 3), (1, 2)))
    # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
    lh = np.tensordot(lh, np.conj(theta2), axes=((2, 0, 1), (0, 1, 2)))
    # [p1] [p2] [vL] vR, [vL*] [p1*] [p2*] vR*
    Lh = IGS.multiply_geometric_sum(lh, guess, tol)  # vR vR*
    return Lh

def get_Rh(umps, h, canonical_form, guess, tol):
    """Compute right environment Rh from geometric sum of transfer matrix TR (umps not necessarily in 
    canonical form).

    ---.                        ---(AR)---^n  ---(AR)--(AR)--.
       |                            |             |     |    |
     (Rh)  =  sum_{n=0}^{infty}     |           (----h----)  |
       |                            |             |     |    |
    ---.                        --(AR*)---    --(AR*)-(AR*)--.
    """
    AR = umps.AR  # vL p vR
    D = np.shape(AR)[0]
    C = umps.C
    L = np.matmul(C.T, np.conj(C))  # vR vR*
    if not canonical_form:
        TR = TransferMatrix([AR], [AR], transpose=True)
        _, L = TR.get_leading_eigenpairs(k=1, guess=L)
        L /= np.trace(L)  # vR vR*
    IGS = InverseGeometricSum(AR, AR, R=np.eye(D), L=L)
    theta2 = np.tensordot(AR, AR, axes=(2, 0))  # vL p1 p2 vR
    rh = np.tensordot(h, theta2, axes=((2, 3), (1, 2)))
    # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
    rh = np.tensordot(rh, np.conj(theta2), axes=((0, 1, 3), (1, 2, 3)))
    # [p1] [p2] vL [vR], vL* [p1*] [p2*] [vR*]
    Rh = IGS.multiply_geometric_sum(rh, guess, tol)  # vL vL*
    return Rh


def subtract_energy_offset(umps, h, canonical_form):
    """Subtract energy of umps from two site Hamiltonian h (umps not necessarily in canonical form).
    
                                             .--(AL)--(AL)--.
                                             |   |     |    |
    h -> h - e * Id with e = <umps|h|umps> = | (----h----) (R)    
                                             |   |     |    |
                                             .-(AL*)--(AL*)-.  
    """
    AL = umps.AL  # vL p vR
    R = np.matmul(umps.C, np.conj(umps.C).T)  # vL vL*
    if not canonical_form:
        T = TransferMatrix([AL], [AL])
        _, R = T.get_leading_eigenpairs(k=1, guess=R)
        R /= np.trace(R)  # vL vL*
    theta2 = np.tensordot(AL, AL, axes=(2, 0))  # vL p1 p2 vR
    lh = np.tensordot(h, theta2, axes=((2, 3), (1, 2)))
    # p1 p2 [p1*] [p2*], vL [p1] [p2] vR
    lh = np.tensordot(lh, np.conj(theta2), axes=((2, 0, 1), (0, 1, 2)))
    # [p1] [p2] [vL] vR, [vL*] [p1*] [p2*] vR*
    e = np.tensordot(lh, R, axes=((0, 1), (0, 1)))  # [vR] [vR*], [vL] [vL*]
    d = np.shape(h)[0]  
    Id = np.reshape(np.eye(d * d), (d, d, d, d))  
    h = h - e * Id  # p1 p2 p1* p2*
    return h


def get_AL_AR(AC, C):
    """From given AC and C, find AL and AR which minimize ||AC - AL C||, ||AC - C AR||.
    
    Left polar decompositions: AC = UL_AC PL_AC, C = UL_C PL_C  ->  AL = UL_AC UL_C^{dagger},
    Right polar decompositions: AC = PR_AC UR_AC, C = PR_C UR_C  ->  AR = UR_C^{dagger} UR_AC.
    """
    D = np.shape(AC)[0]
    d = np.shape(AC)[1]
    UL_AC, _ = polar_right(np.reshape(AC, (D * d, D)))  # vL.p vR
    UL_C, _ = polar_right(C)  
    AL = np.matmul(UL_AC, np.conj(UL_C).T)  # vL.p [vR], [vL] vR
    AL = np.reshape(AL, (D, d, D))  # vL p vR
    _, UR_AC = polar_left(np.reshape(AC, (D, d * D)))  # vL p.vR
    _, UR_C = polar_left(C)
    AR = np.matmul(np.conj(UR_C).T, UR_AC)  # vL [vR], [vL] p.vR
    AR = np.reshape(AR, (D, d, D))  # vL p vR
    return AL, AR 