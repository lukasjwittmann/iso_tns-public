"""Toy code for various decompositions of a complex matrix in matrices with orthonormal rows/columns."""

import numpy as np
from scipy.linalg import qr, svd, polar, LinAlgError


def qr_positive(A):
    """Compute the unique positive QR decomposition of complex MxN matrix A.

    A = QR with Q: MxK matrix fulfilling Q^{dagger}Q = 1
                R: KxN upper triangular matrix with positive diagonal elements [K=min(M,N)]
    """
    Q, R = qr(A, mode="economic")
    diag_R = np.diag(R).copy()
    diag_R[np.abs(diag_R) < 1.e-15] = 1.
    P = np.diag(diag_R/np.abs(diag_R))
    Q = np.matmul(Q, P)
    R = np.matmul(np.conj(P), R)
    return Q, R

def lq_positive(A):
    """Compute the unique positive LQ decomposition of complex MxN matrix A.

    A = LQ with L: MxK lower triangular matrix with positive diagonal elements 
                Q: KxN matrix fulfilling QQ^{dagger} = 1 [K=min(M,N)]
    """
    Q, R = qr_positive(A.T)
    L = R.T
    Q = Q.T
    return L, Q


def svd_safe(A):
    """Compute singular value decomposition of complex MxN matrix A.
    
    A = USV with U: MxK matrix fulfilling U^{dagger}U = 1
                 S: KxK diagonal matrix with positive elements
                 V: KxN matrix fulfilling VV^{dagger} = 1 [K=min(M,N)]

    We use scipy.linalg.svd and first try the more efficient divide-and-conquer approach "gesdd", 
    and if it doesn't converge we use the general rectangular approach "gesvd".
    """
    try:
        U, S, V = svd(A, full_matrices=False)
    except LinAlgError:
        U, S, V = svd(A, full_matrices=False, lapack_driver="gesvd")
    return U, S, V

def svd_truncation(A, chi_max, eps):
    """Normalize A and perform a singular value decomposition. Keep only singular values larger than 
    eps and maximally chi_max of them."""
    norm = np.linalg.norm(A)
    if norm > 1.e-15:
        A = A / norm
    U, S, V = svd_safe(A)
    chi = min(np.sum(S > eps), chi_max)
    assert chi >= 1, "At least one singular value must be kept."
    ind_sorted = np.argsort(S)[::-1]
    ind_keep = ind_sorted[:chi]
    ind_trunc = ind_sorted[chi:]
    trunc_error = np.sum(S[ind_trunc]**2)
    """
    if trunc_error > 1.e-1:
        print(f"Warning: large svd truncation error of {trunc_error}.")
    """
    U, S, V = U[:, ind_keep], S[ind_keep], V[ind_keep, :]
    norm = np.linalg.norm(S)
    if norm > 1.e-15:
        S = S / norm
    return U, S, V, trunc_error


def polar_right(A):
    """Compute right polar decomposition of complex MxN matrix A.

    A = UP with U: MxN matrix fulfilling U^{dagger}U = 1 if M >= N and UU^{dagger} = 1 if M <= N
                P: NxN positive semidefinite matrix
    """
    U, P = polar(A, side="right")
    return U, P

def polar_left(A):
    """Compute left polar decomposition of complex MxN matrix A.

    A = PU with P: MxM positive semidefinite matrix
                U: MxN matrix fulfilling U^{dagger}U = 1 if M >= N and UU^{dagger} = 1 if M <= N
    """
    U, P = polar(A, side="left")
    return P, U