"""Toy code implementing variational quasiparticle excitations on top of a diagonal isometric PEPS
ground state, optimized directly from the overlap with excited wavefunction or MPS."""

import numpy as np
import opt_einsum as oe
from scipy.linalg import null_space
from copy import deepcopy

from .a_iso_peps.src.isoTPS.square.isoTPS import isoTPS_Square as DiagonalIsometricPEPS
from .b_model import DiagonalSquareLattice
from .c_mps import MPS


def extract_all_isometric_configurations(iso_peps, min_dims=True):
    """Extract all (ALs|CDs,CCs,CUs|ARs) configurations of iso_peps by moving the orthogonality 
    column from left to right with YB-moves and the orthogonality center from down to up with 
    QR-decompositions. If min_dims=True, always take the minimum of the outer dimensions for the 
    inner dimension in the orthogonal matrix decompositions."""
    iso_peps_copy = DiagonalIsometricPEPS(iso_peps.Lx, iso_peps.Ly, D_max=iso_peps.D_max, \
                                          chi_factor=iso_peps.chi_factor, chi_max=iso_peps.chi_max, \
                                          d=iso_peps.d, shifting_options=iso_peps.shifting_options, \
                                          yb_options=iso_peps.yb_options, \
                                          tebd_options=iso_peps.tebd_options)
    iso_peps_copy._init_as_copy(iso_peps)
    Lx = iso_peps_copy.Lx
    Ly = iso_peps_copy.Ly
    Nx = 2 * Lx
    Ny = 2 * Ly - 1
    ALs = [[None] * Ly for _ in range(Nx)]
    ARs = [[None] * Ly for _ in range(Nx)]
    CDs = [[None] * Ny for _ in range(Nx)]
    CCs = [[None] * Ny for _ in range(Nx)]
    CUs = [[None] * Ny for _ in range(Nx)]
    iso_peps_copy.move_orthogonality_column_to(0, min_dims)
    if min_dims:
        Cs = iso_peps_copy.get_Cs(0)
        C = MPS(Cs, norm=1.)
        Us, Vs, Ss, _ = C.get_canonical_form()
        iso_peps_copy.Ws = [np.transpose(U, (1, 3, 2, 0)) for U in Us]
    ARs[0] = iso_peps_copy.get_ARs(0)
    for nx in range(Nx):
        iso_peps_copy.move_orthogonality_column_to(nx+1, min_dims)
        ALs[nx] = iso_peps_copy.get_ALs(nx+1)
        if nx < Nx-1:
            ARs[nx+1] = iso_peps_copy.get_ARs(nx+1)
        Cs = iso_peps_copy.get_Cs(nx+1)
        C = MPS(Cs, norm=1.)
        Us, Vs, Ss, _ = C.get_canonical_form()
        CDs[nx] = Us
        CCs[nx] = [np.tensordot(Ss[ny], Vs[ny], axes=(1, 0)) for ny in range(Ny)]
        CUs[nx] = Vs
        if min_dims:
            if nx%2 == 0:
                iso_peps_copy.Ws = [np.transpose(V, (1, 3, 2, 0)) for V in Vs]
            elif nx%2 == 1:
                iso_peps_copy.Ws = [np.transpose(U, (1, 3, 2, 0)) for U in Us]
    return ALs, ARs, CDs, CCs, CUs  


def extract_single_isometric_configuration(nx, ny, ALs, ARs, CDs, CCs, CUs):
    """From complete configurations (ALs|CDs,CCs,CUs|ARs), extract the one for orthogonality column
    and center at nx and ny."""
    ALs_single = deepcopy(ALs[:nx+1][:])
    ARs_single = deepcopy(ARs[nx+1:][:]) 
    Cs_single = deepcopy(CDs[nx][:ny]) + deepcopy([CCs[nx][ny]]) + deepcopy(CUs[nx][ny+1:])
    return ALs_single, ARs_single, Cs_single


def isometric_peps_from_single_configuration(D_max, chi_max_c, nx, ny, ALs_single, ARs_single, Cs_single):
    """From single configuration (ALs_single|Cs_single|ARs_single) initialize an isometric PEPS with 
    orthogonality column and center at nx and ny, and maximal site/column leg dimension of 
    D_max/chi_max_c."""
    assert len(ALs_single) == nx+1
    Nx_L = len(ALs_single)
    Nx_R = len(ARs_single)
    Nx = Nx_L + Nx_R
    Lx = Nx // 2
    Ly = len(ALs_single[0])
    lattice = DiagonalSquareLattice(Lx, Ly)
    Ts = [None] * Nx * Ly
    for bx in range(Nx_L):
        for y in range(Ly):
            Ts[lattice.get_site_scalar((bx//2, y, bx%2))] = np.transpose(ALs_single[bx][y].copy(), \
                                                                         (0, 4, 3, 1, 2))
    if Nx_R > 0:
        for bx in range(Nx_L, Nx):
            for y in range(Ly):
                Ts[lattice.get_site_scalar((bx//2, y, bx%2))] = np.transpose(ARs_single[bx-nx-1][y].copy(), \
                                                                             (0, 2, 1, 3, 4))
    Ws = [np.transpose(C.copy(), (1, 3, 2, 0)) for C in Cs_single]
    peps_parameters = {
    "Lx": Lx,
    "Ly": Ly,
    "D_max": D_max,
    "chi_max": chi_max_c,
    "d": 2,
    "yb_options" : { 
        "mode" : "svd",
        "disentangle": True,
        "disentangle_options": {
            "mode": "renyi_approx",
            "renyi_alpha": 0.5,
            "method": "trm",
            "N_iters": 100,
        }
    },
    "tebd_options": {
        "mode" : "iterate_polar",
        "N_iters": 100,
    }
    }
    iso_peps = DiagonalIsometricPEPS(**peps_parameters)
    iso_peps.Ts = Ts
    iso_peps.Ws = Ws
    iso_peps.ortho_surface = nx
    iso_peps.ortho_center = ny
    return iso_peps


# excitations AL-VL-X-AR

def get_VLs(ALs):
    """For left orthonormal tensor AL[nx][y], compute tensor VL[nx][y] such that
    .
    |\
    | \
    .  (VL)=== d*Dld*Dlu-Drd*Dru      \     //               \      /
     \/ |  Dru                         \ | //                 \ |  /
     /\ |  /    = 0               ->    (VL)   for even nx,    (VL)   for odd nx
    .  (AL*)                           /    \                 /  \ \
    | /    \                          /      \               /    \ \
    |/     Drd                   
    .         

    If d*Dld*Dru-Drd*Dru <= 0, set the corresponding VL to None.                         
    """
    Nx = len(ALs)
    Ly = len(ALs[0])
    VLs = [[None] * Ly for _ in range(Nx)]
    for nx in range(Nx):
        for y in range(Ly):
            AL = ALs[nx][y].copy()
            d, Dld, Dlu, Drd, Dru = np.shape(AL)
            if (d * Dld * Dlu) - (Drd * Dru) > 0:
                AL = np.reshape(AL, (d * Dld * Dlu, Drd * Dru))
                VL = null_space(np.conj(AL).T)
                Drd_Dru_ortho = (d * Dld * Dlu) - (Drd * Dru)
                if nx%2 == 0:
                    Drd_ortho, Dru_ortho = 1, Drd_Dru_ortho
                elif nx%2 == 1:
                    Drd_ortho, Dru_ortho = Drd_Dru_ortho, 1
                VL = np.reshape(VL, (d, Dld, Dlu, Drd_ortho, Dru_ortho))
                VLs[nx][y] = VL
    return VLs


def get_shape_Xs_vecX(ALs, CDs, CCs, CUs):
    """For left orthonormal tensor AL[nx][y] and orthogonality column tensors CC[nx][2*y] and
    CD[nx][2*y-1]/CU[nx][2*y+1], compute the following shapes of perturbation tensor X[nx][2*y] and
    identites XD[nx][2*y-1]/XU[nx][2*y+1]:

    for even nx:
    
    Dlu     chi_u Druu      Dlu     chi_u Druu              .
      \         |/            \         |/                  |\
       \       (CC)            \       (X)                  | \
        \  d   /|               \  d  //|                   .  (VL)=== d*Dld*Dlu-Drd*Dru
         \ |  / |                \ | // |                    \/ |  Dru
          (AL)  |       ->        (VL)  |chi_d*Drdd  with    /\ |  /    = 0
         /    \ |                /   1\ |                   .  (AL*)
        /      \|               /      \|                   | /    \
       /       (CD)            /       (1)                  |/     Drd
      /         |\            /         |\                  .
    Dld     chi_d Drdd      Dld     chi_d Drdd

    for odd nx:
    
    Dlu     chi_u Druu      Dlu     chi_u Druu              .
      \         |/            \         |/                  |\
       \       (CU)            \       (1)                  | \
        \  d   /|               \  d  / |                   .  (VL)=== d*Dld*Dlu-Drd*Dru
         \ |  / |                \ | /1 |                    \/ |  Dru
          (AL)  |       ->        (VL)  |chi_u*Druu  with    /\ |  /    = 0
         /    \ |                /  \ \ |                   .  (AL*)    
        /      \|               /    \ \|                   | /    \
       /       (CC)            /       (X)                  |/     Drd
      /         |\            /         |\                  .
    Dld     chi_d Drdd      Dld     chi_d Drdd

    If d*Dld*Dlu-Drd*Dru <= 0, set the corresponding shapes to None. Also return the length of the 
    vector vecX containing all X[nx][2*y]s that are not None.
    """
    Nx = len(ALs)
    Ly = len(ALs[0])
    Ny = 2 * Ly - 1
    shape_XDs = [[None] * Ny for _ in range(Nx)]
    shape_Xs = [[None] * Ny for _ in range(Nx)]
    shape_XUs = [[None] * Ny for _ in range(Nx)]
    shape_vecX = 0
    for nx in range(Nx):
        for y in range(Ly):
            AL = ALs[nx][y].copy()
            d, Dld, Dlu, Drd, Dru = np.shape(AL)
            if (d * Dld * Dlu) - (Drd * Dru) > 0:
                Drd_Dru_ortho = (d * Dld * Dlu) - (Drd * Dru)
                if nx%2 == 0:
                    Drd_ortho, Dru_ortho = 1, Drd_Dru_ortho
                    if y == 0:
                        CC = CCs[nx][0].copy()
                        Druu, chi_u = np.shape(CC)[2], np.shape(CC)[3]
                        shape_Xs[nx][0] = (1, Dru_ortho, Druu, chi_u)
                        shape_vecX += 1 * Dru_ortho * Druu * chi_u
                    else:
                        CD = CDs[nx][2*y-1].copy()
                        chi_d, Drdd = np.shape(CD)[0], np.shape(CD)[2]
                        chi = chi_d * Drdd
                        shape_XDs[nx][2*y-1] = (chi_d, Drd_ortho, Drdd, chi)
                        CC = CCs[nx][2*y].copy()
                        Druu, chi_u = np.shape(CC)[2], np.shape(CC)[3]
                        shape_Xs[nx][2*y] = (chi, Dru_ortho, Druu, chi_u)
                        shape_vecX += chi * Dru_ortho * Druu * chi_u
                elif nx%2 == 1:
                    Drd_ortho, Dru_ortho = Drd_Dru_ortho, 1
                    if y == Ly-1:
                        CC = CCs[nx][2*Ly-2].copy()
                        chi_d, Drdd = np.shape(CC)[0], np.shape(CC)[2]
                        shape_Xs[nx][2*Ly-2] = (chi_d, Drd_ortho, Drdd, 1)
                        shape_vecX += chi_d * Drd_ortho * Drdd * 1
                    else:
                        CU = CUs[nx][2*y+1].copy()
                        Druu, chi_u = np.shape(CU)[2], np.shape(CU)[3]
                        chi = Druu * chi_u
                        shape_XUs[nx][2*y+1] = (chi, Dru_ortho, Druu, chi_u)
                        CC = CCs[nx][2*y].copy()
                        chi_d, Drdd = np.shape(CC)[0], np.shape(CC)[2]
                        shape_Xs[nx][2*y] = (chi_d, Drd_ortho, Drdd, chi)
                        shape_vecX += chi_d * Drd_ortho * Drdd * chi
    return shape_XDs, shape_Xs, shape_XUs, shape_vecX


def extract_single_excitation_configuration(nx, y, ALs, ARs, CDs, CCs, CUs):
    """From complete configurations (ALs|CDs,CCs,CUs|ARs), extract the one needed for an excitation
    at site (nx, y). If the dimensions allow, the corresponding AL is replaced by its orthogonal
    complement VL, and the CC which has to be replaced by X is set to None for now."""
    VLs = get_VLs(ALs)
    VL = VLs[nx][y].copy()
    if VL is not None:
        ALs_single, ARs_single, Cs_single = extract_single_isometric_configuration(nx, 2*y, ALs, ARs, \
                                                                                   CDs, CCs, CUs)
        ALs_VL_single = deepcopy(ALs_single)
        ALs_VL_single[nx][y] = VL
        Cs_noX_single = deepcopy(Cs_single)
        Cs_noX_single[2*y] = None
        shape_XDs, _, shape_XUs, _ = get_shape_Xs_vecX(ALs, CDs, CCs, CUs)
        if nx%2 == 0 and y > 0:
            shape_XD = shape_XDs[nx][2*y-1]
            chi_d, Drd_ortho, Drdd, chi = shape_XD
            assert chi_d * Drd_ortho * Drdd == chi
            XD = np.eye(chi)
            XD = np.reshape(XD, shape_XD)
            Cs_noX_single[2*y-1] = XD
        elif nx%2 == 1 and y < len(ALs_single[0]) - 1:
            shape_XU = shape_XUs[nx][2*y+1]
            chi, Dru_ortho, Druu, chi_u = shape_XU
            assert chi == Dru_ortho * Druu * chi_u
            XU = np.eye(chi)
            XU = np.reshape(XU, shape_XU)
            Cs_noX_single[2*y+1] = XU
        return ALs_VL_single, ARs_single, Cs_noX_single
    return None


def vec_to_tensors(vecX, shape_Xs):
    """Reshape a vector vecX into tensors of shapes shape_Xs."""
    Nx = len(shape_Xs)
    Ny = len(shape_Xs[0])
    Xs = [[None] * Ny for _ in range(Nx)]
    vec_ind = 0
    for nx in range(Nx):
        for ny in range(Ny):
            shape_X = shape_Xs[nx][ny]
            if shape_X is not None:
                X = vecX[vec_ind : vec_ind + np.prod(shape_X)]
                X = np.reshape(X, shape_X)
                Xs[nx][ny] = X
                vec_ind += np.prod(shape_X)
    assert vec_ind == len(vecX)
    return Xs


def tensors_to_vec(Xs, shape_vecX):
    """Reshape all tensors in Xs into one vector of length shape_vecX."""
    Nx = len(Xs)
    Ny = len(Xs[0])
    vecX = np.zeros(shape_vecX, dtype=complex)
    vec_ind = 0
    for nx in range(Nx):
        for ny in range(Ny):
            X = Xs[nx][ny]
            if X is not None:
                X = X.copy().flatten()
                vecX[vec_ind : vec_ind + np.size(X)] = X
                vec_ind += np.size(X)
    assert vec_ind == shape_vecX
    return vecX


# excitations AL-AL-AL-X_column

def get_VDs(CDs):
    """For down orthonormal orthogonality column tensor CD[ny], compute tensor VD[ny] such that

            chi_u
              |
              |
         .--(CD*)---.
        /     |    /
       /chi   |   /     =   0   with chi = chi_d * Dl * Dr - chi_u
      /  |    .  /
     /   |   /  /
    .---(VD)---.
         | /
         |/
         .

    If chi_d * Dl * Dr - chi_u <= 0, set the corresponding VD to None.
    """
    Ny = len(CDs)
    VDs = [None] * Ny
    for ny in range(Ny):
        CD = CDs[ny].copy()
        chi_d, Dl, Dr, chi_u = np.shape(CD)
        if chi_d * Dl * Dr - chi_u > 0:
            CD = np.reshape(CD, (chi_d * Dl * Dr, chi_u))
            VD = null_space(np.conj(CD).T)
            VD = np.reshape(VD, (chi_d, Dl, Dr, chi_d * Dl * Dr - chi_u))
            VDs[ny] = VD
    return VDs


def get_shape_Xs_vecX_column(CDs):
    """For down orthonormal orthogonality column tensor CD[ny], compute the shape 
    (chi_d * Dl * Dr - chi_u, chi_u) of the down-gauge excitation parametrization X[ny]. If 
    chi_d * Dl * Dr - chi_u <= 0, set the corresponding shape to None. Also return the length of the
    vector vecX containing all X[ny]s that are not None."""
    Ny = len(CDs)
    shape_Xs = [None] * Ny
    shape_vecX = 0
    for ny in range(Ny):
        CD = CDs[ny].copy()
        chi_d, Dl, Dr, chi_u = np.shape(CD)
        if chi_d * Dl * Dr - chi_u > 0:
            shape_Xs[ny] = (chi_d * Dl * Dr - chi_u, chi_u)
            shape_vecX += (chi_d * Dl * Dr - chi_u) * chi_u
    return shape_Xs, shape_vecX


def extract_single_excitation_configuration_column(ny, ALs, ARs, CDs, CCs, CUs):
    """From complete configurations (ALs|CDs,CCs,CUs|ARs), extract the one needed for an excitation
    on the last column Nx-1 at bond ny. If the dimensions allow, the corresponding CC which has to 
    be replaced by B = VD X is set to None for now."""
    Nx = len(ALs)
    VDs = get_VDs(CDs[Nx-1])
    VD = VDs[ny]
    if VD is not None:
        ALs_single, ARs_single, Cs_single = extract_single_isometric_configuration(Nx-1, ny, ALs, ARs, \
                                                                                   CDs, CCs, CUs)
        Cs_noB_single = deepcopy(Cs_single)
        Cs_noB_single[ny] = None
        return ALs_single, ARs_single, Cs_noB_single
    

def vec_to_tensors_column(vecX_column, shape_Xs_column):
    """Reshape a vector vecX_column into tensors of shapes shape_Xs_column."""
    Ny = len(shape_Xs_column)
    Xs_column = [None] * Ny
    vec_ind = 0
    for ny in range(Ny):
        shape_X_column = shape_Xs_column[ny]
        if shape_X_column is not None:
            X_column = vecX_column[vec_ind : vec_ind + np.prod(shape_X_column)]
            X_column = np.reshape(X_column, shape_X_column)
            Xs_column[ny] = X_column
            vec_ind += np.prod(shape_X_column)
    assert vec_ind == len(vecX_column)
    return Xs_column


def tensors_to_vec_column(Xs_column, shape_vecX_column):
    """Reshape all tensors in Xs_column into one vector of length shape_vecX_column."""
    Ny = len(Xs_column)
    vecX_column = np.zeros(shape_vecX_column, dtype=complex)
    vec_ind = 0
    for ny in range(Ny):
        X_column = Xs_column[ny]
        if X_column is not None:
            X_column = X_column.copy().flatten()
            vecX_column[vec_ind : vec_ind + len(X_column)] = X_column
            vec_ind += len(X_column)
    assert vec_ind == shape_vecX_column
    return vecX_column


def Xs_to_Bs_column(Xs_column, VDs):
    """For X[ny] the down-gauge parametrization, compute the perturbation tensor B[ny] given by

                        |
                       (X)
        |               |
    ---(B)---   =   ---(VD)--- 
        |               |
    
    """
    Ny = len(Xs_column)
    Bs_column = [None] * Ny
    for ny in range(Ny):
        VD = VDs[ny]
        if VD is not None:
            Bs_column[ny] = np.tensordot(VD.copy(), Xs_column[ny], axes=(3, 0))
    return Bs_column


def Bs_to_Xs_column(Bs_column, VDs):
    """For B[ny] the perturbation tensor, compute the down-gauge parametrization given by

              |
              |
         .--(VD*)---.
        /     |    /         |
       /      |   /     =   (X) 
      /  |    .  /           |
     /   |   /  /
    .---(B)----.
         | /
         |/
         .
    """
    Ny = len(Bs_column)
    Xs_column = [None] * Ny
    for ny in range(Ny):
        B = Bs_column[ny]
        if B is not None:
            Xs_column[ny] = np.tensordot(np.conj(VDs[ny]), B.copy(), axes=((0, 1, 2), (0, 1, 2)))
    return Xs_column


def get_overlap_wavefunction_iso_peps(psi, iso_peps, \
                                      nx_c=None, ny_c=None, ALs_single=None, ARs_single=None, Cs_single=None, \
                                      deriv_after_CC=False):
    """Compute the overlap between a full wavefunction psi and an iso_peps, which can be given as a 
    class instance or as single tensor configuration. If deriv_after_CC=True, leave the legs of CC*
    open, effectively optimizing CC for maximal overlap."""
    # iso_peps tensors
    if iso_peps is not None:
        Lx = iso_peps.Lx
        Ly = iso_peps.Ly
        lattice = DiagonalSquareLattice(Lx, Ly)
        Nx = 2 * Lx
        Ny = 2 * Ly - 1
        N = 2 * Lx * Ly
        nx_c = iso_peps.ortho_surface
        ny_c = iso_peps.ortho_center
        ALs = [iso_peps.get_ALs(nx+1) for nx in range(nx_c + 1)] + [None] * (Nx - nx_c - 1)
        ARs = [None] * (nx_c + 1) + [iso_peps.get_ARs(nx+1) for nx in range(nx_c, Nx-1)]
        Cs_single = iso_peps.get_Cs(nx_c+1)
    else:
        Nx = len(ALs_single) + len(ARs_single)
        Ny = len(Cs_single)
        Lx = Nx // 2
        Ly = (Ny + 1) // 2
        lattice = DiagonalSquareLattice(Lx, Ly)
        N = 2 * Lx * Ly
        ALs = ALs_single + [None] * (Nx - nx_c - 1)
        ARs = [None] * (nx_c + 1) + ARs_single
    # wavefunction with legs in same order as physical iso_peps legs
    psi = np.reshape(psi, (2,)*N)
    psi_leg_order = []
    for nx in range(nx_c + 1):
        for y in range(Ly):
            x, p = nx//2, nx%2
            site_scalar = lattice.get_site_scalar((x, y, p))
            psi_leg_order.append(site_scalar)
    for nx in reversed(range(nx_c+1, Nx)):
        for y in range(Ly):
            x, p = nx//2, nx%2
            site_scalar = lattice.get_site_scalar((x, y, p))
            psi_leg_order.append(site_scalar)
    psi = np.transpose(psi, psi_leg_order)
    # function for contracting boundary (with all open wavefunction legs) with one column of As
    def contract_boundary(B, As_array, nx, side="left"):
        if side == "left":
            B_leg_base = 0
        elif side == "right":
            B_leg_base = Ny
        As = As_array[nx]
        B_new = B.copy()
        if nx%2 == 0:
            A = As[0]
            assert np.shape(A)[1] == np.shape(A)[3] == 1
            A = A[:, 0, :, 0, :]
            B_new = np.tensordot(B_new, np.conj(A), axes=((B_leg_base+Ny, B_leg_base), (0, 1)))
            B_new = np.moveaxis(B_new, -1, B_leg_base)
            for y in range(1, Ly):
                B_new = np.tensordot(B_new, np.conj(As[y]), axes=((B_leg_base+Ny, B_leg_base+2*y-1, \
                                                                   B_leg_base+2*y), (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, B_leg_base+2*y-1)
                B_new = np.moveaxis(B_new, -1, B_leg_base+2*y)
        elif nx%2 == 1:
            for y in range(Ly-1):
                B_new = np.tensordot(B_new, np.conj(As[y]), axes=((B_leg_base+Ny, B_leg_base+2*y, \
                                                                   B_leg_base+2*y+1), (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, B_leg_base+2*y)
                B_new = np.moveaxis(B_new, -1, B_leg_base+2*y+1)
            A = As[Ly-1]
            assert np.shape(A)[2] == np.shape(A)[4] == 1
            A = A[:, :, 0, :, 0]
            B_new = np.tensordot(B_new, np.conj(A), axes=((B_leg_base+Ny, B_leg_base+Ny-1), (0, 1)))
            B_new = np.moveaxis(B_new, -1, B_leg_base+Ny-1)
        return B_new
    # left and right part
    LR = np.expand_dims(psi, axis=tuple(range(Ny)))
    for nx in range(nx_c + 1):
        LR = contract_boundary(LR, ALs, nx, side="left")
    LR = np.expand_dims(LR, axis=tuple(range(Ny, 2*Ny)))
    for nx in reversed(range(nx_c+1, Nx)):
        LR = contract_boundary(LR, ARs, nx, side="right")
    # orthogonality column
    LCR = np.expand_dims(LR, axis=0)
    for ny in range(ny_c):
        LCR = np.tensordot(np.conj(Cs_single[ny]), LCR, axes=((0, 1, 2), (0, 1, Ny-ny+1)))
    LCR = np.expand_dims(LCR, axis=0)
    for ny in reversed(range(ny_c+1, Ny)):
        LCR = np.tensordot(np.conj(Cs_single[ny]), LCR, axes=((3, 1, 2), (0, ny-ny_c+2, 2*(ny-ny_c+1)+1)))
    # full overlap without open legs
    if not deriv_after_CC:
        LCR = np.tensordot(np.conj(Cs_single[ny_c]), LCR, axes=((0, 1, 2, 3), (1, 2, 3, 0)))
        return np.real_if_close(LCR)
    # leave legs of CC* open
    elif deriv_after_CC:
        CC = np.transpose(LCR, (1, 2, 3, 0))
        return CC


def get_overlap_mps_iso_peps(mps, iso_peps, \
                             nx_c=None, ny_c=None, ALs_single=None, ARs_single=None, Cs_single=None, \
                             deriv_after_CC=False):
    """Compute the overlap between an mps and an iso_peps, which can be given as a class instance or 
    as single tensor configuration. For Lx > Ly the mps line is assumed to go from down to up on
    every column, and accordingly from left to right for Ly >= Lx. If deriv_after_CC=True, leave the 
    legs of CC* open, effectively optimizing CC for maximal overlap."""
    # iso_peps tensors
    if iso_peps is not None:
        Lx = iso_peps.Lx
        Ly = iso_peps.Ly
        Nx = 2 * Lx
        Ny = 2 * Ly - 1
        N = 2 * Lx * Ly
        nx_c = iso_peps.ortho_surface
        ny_c = iso_peps.ortho_center
        ALs = [iso_peps.get_ALs(nx+1) for nx in range(nx_c + 1)] + [None] * (Nx - nx_c - 1)
        ARs = [None] * (nx_c + 1) + [iso_peps.get_ARs(nx+1) for nx in range(nx_c, Nx-1)]
        Cs_single = iso_peps.get_Cs(nx_c+1)
    else:
        Nx = len(ALs_single) + len(ARs_single)
        Ny = len(Cs_single)
        Lx = Nx // 2
        Ly = (Ny + 1) // 2
        N = 2 * Lx * Ly
        ALs = ALs_single + [None] * (Nx - nx_c - 1)
        ARs = [None] * (nx_c + 1) + ARs_single
    assert len(mps) == N
    if Lx > Ly:
        # mps tensors
        Ms = [[mps[nx * Ly + y].copy() for y in range(Ly)] for nx in range(Nx)]
        # function for contracting boundary with one column of Ms and As
        def contract_boundary(B, Ms_array, As_array, nx, side="left"):
            Ms = Ms_array[nx]
            As = As_array[nx]
            if side == "right":
                Ms = [np.transpose(M, (2, 1, 0)) for M in Ms[::-1]]
                As = [np.transpose(A, (0, 2, 1, 4, 3)) for A in As[::-1]]
                nx += 1
            B_new = B.copy()
            if nx%2 == 0:
                B_new = np.tensordot(Ms[0], B_new, axes=(0, 0))
                A = As[0]
                assert np.shape(A)[1] == np.shape(A)[3] == 1
                A = A[:, 0, :, 0, :]
                B_new = np.tensordot(B_new, np.conj(A), axes=((0, 2), (0, 1)))
                B_new = np.moveaxis(B_new, -1, 1)
                for y in range(1, Ly):
                    B_new = np.tensordot(Ms[y], B_new, axes=(0, 0))
                    B_new = np.tensordot(B_new, np.conj(As[y]), axes=((0, 2*(y+1)-1, 2*(y+1)), \
                                                                      (0, 1, 2)))
                    B_new = np.moveaxis(B_new, -2, 2*y)
                    B_new = np.moveaxis(B_new, -1, 2*y+1)
            elif nx%2 == 1:
                for y in range(Ly-1):
                    B_new = np.tensordot(Ms[y], B_new, axes=(0, 0))
                    B_new = np.tensordot(B_new, np.conj(As[y]), axes=((0, 2*(y+1), 2*(y+1)+1), \
                                                                      (0, 1, 2)))
                    B_new = np.moveaxis(B_new, -2, 2*y+1)
                    B_new = np.moveaxis(B_new, -1, 2*(y+1))
                A = As[Ly-1]
                assert np.shape(A)[2] == np.shape(A)[4] == 1
                A = A[:, :, 0, :, 0]
                B_new = np.tensordot(Ms[Ly-1], B_new, axes=(0, 0))
                B_new = np.tensordot(B_new, np.conj(A), axes=((0, 2*Ly), (0, 1)))
                B_new = np.moveaxis(B_new, -1, 2*Ly-1)
            return B_new
        # left part
        L = np.ones((1,) * (Ny+1))
        for nx in range(nx_c + 1):
            L = contract_boundary(L, Ms, ALs, nx)
        # right part
        R = np.ones((1,) * (Ny+1))
        for nx in reversed(range(nx_c+1, Nx)):
            R = contract_boundary(R, Ms, ARs, nx, side="right")
        inverted_legs = (0,) + tuple(range(R.ndim - 1, 0, -1))
        R = np.transpose(R, inverted_legs)
        # orthogonality column
        LC = np.expand_dims(L, axis=1)
        for ny in range(ny_c):
            LC = np.tensordot(LC, np.conj(Cs_single[ny]), axes=((1, ny+2), (0, 1)))
            LC = np.moveaxis(LC, -2, ny+1)
            LC = np.moveaxis(LC, -1, 1)
        # full overlap without open legs
        if not deriv_after_CC:
            for ny in range(ny_c, Ny):
                LC = np.tensordot(LC, np.conj(Cs_single[ny]), axes=((1, ny+2), (0, 1)))
                LC = np.moveaxis(LC, -2, ny+1)
                LC = np.moveaxis(LC, -1, 1)
            assert np.shape(LC)[1] == 1
            LC = np.squeeze(LC, axis=1)
            LCR = oe.contract("...,...->", LC, R)
            return np.real_if_close(LCR)
        # leave legs of CC* open
        elif deriv_after_CC:
            LC = np.moveaxis(LC, 1, -1)
            LC = np.moveaxis(LC, ny_c+1, -1)
            if ny_c < Ny-1:
                LC = np.tensordot(LC, np.conj(Cs_single[ny_c+1]), axes=(ny_c+1, 1))
                LC = np.moveaxis(LC, -2, ny_c+1)
                LC = np.moveaxis(LC, -1, 1)
                for ny in range(ny_c+2, Ny):
                    LC = np.tensordot(LC, np.conj(Cs_single[ny]), axes=((1, ny+1), (0, 1)))
                    LC = np.moveaxis(LC, -2, ny)
                    LC = np.moveaxis(LC, -1, 1)
                assert np.shape(LC)[1] == 1
                LC = np.squeeze(LC, axis=1)
            else:
                LC = np.expand_dims(LC, axis=-1)
            shape_LC = (np.prod(np.shape(LC)[:-3]),) + np.shape(LC)[-3:]
            LC = np.reshape(LC, shape_LC)
            R = np.moveaxis(R, ny_c+1, -1)
            shape_R = (np.prod(np.shape(R)[:-1]), np.shape(R)[-1])
            R = np.reshape(R, shape_R)
            CC = oe.contract("abcd,ae->bcde", LC, R)
            CC = np.transpose(CC, (0, 1, 3, 2))
            return CC
    elif Ly >= Lx:
        Ny = 2 * Ly
        As = [[None] * Lx for _ in range(Ny)]
        for nx in range(nx_c + 1):
            x = nx // 2
            p = nx % 2
            for y in range(Ly):
                As[2*y+p][x] = np.transpose(ALs[nx][y].copy(), (0, 1, 3, 2, 4))
        for nx in range(nx_c + 1, Nx):
            x = nx // 2
            p = nx % 2
            for y in range(Ly):
                As[2*y+p][x] = np.transpose(ARs[nx][y].copy(), (0, 3, 1, 4, 2))  
        if nx_c%2 == 1:
            Cs_single = [np.transpose(C.copy(), (0, 2, 1, 3)) if C is not None else None \
                                                              for C in Cs_single]
        Nx = 2 * Lx - 1
        # mps tensors
        Ms = [[mps[ny * Lx + x].copy() for x in range(Lx)] for ny in range(Ny)]
        # function for contracting boundary with one line of Ms and As
        def contract_boundary(B, Ms_array, As_array, ny, side="down"):
            Ms = Ms_array[ny]
            As = As_array[ny]
            p = 0
            if side == "down" and ny%2 == 1:
                p = 1
            elif side == "up":
                Ms = [np.transpose(M, (2, 1, 0)) for M in Ms[::-1]]
                As = [np.transpose(A, (0, 4, 3, 2, 1)) for A in As[::-1]]
                if ny%2 == 0:
                    p = 1
            B_new = B.copy()
            for x in range(Lx):
                B_new = np.tensordot(Ms[x], B_new, axes=(0, 0))
                B_new = np.tensordot(B_new, np.conj(As[x]), axes=((0, 2*(x+1)+p, 2*(x+1)+p+1), \
                                                                  (0, 1, 2)))
                B_new = np.moveaxis(B_new, -2, 2*x+1+p)
                B_new = np.moveaxis(B_new, -1, 2*(x+1)+p)
            return B_new
        # down part
        D = np.ones((1,) * (2 * Lx + 3))
        for ny in range(ny_c):
            D = contract_boundary(D, Ms, As, ny)
            if ny%2 == 0:
                D = np.tensordot(D, np.conj(Cs_single[ny]), axes=((-1, nx_c+2), (0, 1)))
            elif ny%2 == 1:
                D = np.tensordot(D, np.conj(Cs_single[ny]), axes=((-1, nx_c+2), (0, 2)))
            D = np.moveaxis(D, -2, nx_c+2)
        D = contract_boundary(D, Ms, As, ny_c)
        # up part
        U = np.ones((1,) * (2 * Lx + 3))
        for ny in reversed(range(ny_c+2, Ny)):
            U = contract_boundary(U, Ms, As, ny, side="up")
            if ny%2 == 1:
                U = np.tensordot(U, np.conj(Cs_single[ny-1]), axes=((-1, 2*Lx-nx_c), (3, 2)))
            elif ny%2 == 0:
                U = np.tensordot(U, np.conj(Cs_single[ny-1]), axes=((-1, 2*Lx-nx_c), (3, 1)))
            U = np.moveaxis(U, -1, 2*Lx-nx_c)
        U = contract_boundary(U, Ms, As, ny_c+1, side="up")
        inverted_legs = (0,) + tuple(range(U.ndim - 2, 0, -1)) + (U.ndim-1,)
        U = np.transpose(U, inverted_legs)
        # full overlap without open legs
        if not deriv_after_CC:
            if ny_c%2 == 0:
                D = np.tensordot(D, np.conj(Cs_single[ny_c]), axes=((-1, nx_c+2), (0, 1)))
            elif ny_c%2 == 1:
                D = np.tensordot(D, np.conj(Cs_single[ny_c]), axes=((-1, nx_c+2), (0, 2)))
            D = np.moveaxis(D, -2, nx_c+2)
            DU = oe.contract("...,...->", D, U)
            return np.real_if_close(DU)
        # leave legs of CC* open
        elif deriv_after_CC:
            D = np.moveaxis(D, nx_c+2, -1)
            shape_D = (np.prod(np.shape(D)[:-2]),) + np.shape(D)[-2:]
            D = np.reshape(D, shape_D)
            U = np.moveaxis(U, nx_c+2, -2)
            shape_U = (np.prod(np.shape(U)[:-2]),) + np.shape(U)[-2:]
            U = np.reshape(U, shape_U)
            CC = oe.contract("abc,ade->bcde", D, U)
            if nx_c%2 == 1:
                CC = np.transpose(CC, (0, 2, 1, 3))
            if ny_c%2 == 1:
                CC = np.transpose(CC, (0, 2, 1, 3))
            return CC


def get_wavefunction(iso_peps):
    """Convert an iso_peps into its corresponding wavefunction, by contracting all virtual legs 
    column-wise."""
    Lx = iso_peps.Lx
    Ly = iso_peps.Ly
    lattice = DiagonalSquareLattice(Ly, Ly)
    Nx = 2 * Lx
    Ny = 2 * Ly - 1
    N = 2 * Lx * Ly
    nx_c = iso_peps.ortho_surface
    ALs = [iso_peps.get_ALs(nx+1) for nx in range(nx_c + 1)] + [None] * (Nx - nx_c - 1)
    ARs = [None] * (nx_c + 1) + [iso_peps.get_ARs(nx+1) for nx in range(nx_c, Nx-1)]
    Cs = iso_peps.get_Cs(nx_c+1)
    def contract_boundary_with_As(B, As, nx):
        if nx%2 == 0:
            A = As[nx][0]
            assert np.shape(A)[1] == np.shape(A)[3] == 1
            A = A[:, 0, :, 0, :]
            B = np.tensordot(B, A, axes=(0, 1))
            B = np.moveaxis(B, -1, 0)
            for y in range(1, Ly):
                A = As[nx][y]
                B = np.tensordot(B, A, axes=((2*y-1, 2*y), (1, 2)))
                B = np.moveaxis(B, -2, 2*y-1)
                B = np.moveaxis(B, -1, 2*y)
        elif nx%2 == 1:
            for y in range(Ly-1):
                A = As[nx][y]
                B = np.tensordot(B, A, axes=((2*y, 2*y+1), (1, 2)))
                B = np.moveaxis(B, -2, 2*y)
                B = np.moveaxis(B, -1, 2*y+1)
            A = As[nx][Ly-1]
            assert np.shape(A)[2] == np.shape(A)[4] == 1
            A = A[:, :, 0, :, 0]
            B = np.tensordot(B, A, axes=(Ny-1, 1))
            B = np.moveaxis(B, -1, Ny-1)
        return B
    L = np.ones(Ny * (1,))
    for nx in range(nx_c + 1):
        L = contract_boundary_with_As(L, ALs, nx)
    LC = np.expand_dims(L, axis=0)
    for ny in range(Ny):
        LC = np.tensordot(LC, Cs[ny], axes=((0, ny+1), (0, 1)))
        LC = np.moveaxis(LC, -2, ny)
        LC = np.moveaxis(LC, -1, 0)
    assert np.shape(LC)[0] == 1
    LC = np.squeeze(LC, axis=0)
    R = np.ones(Ny * (1,))
    for nx in reversed(range(nx_c+1, Nx)):
        R = contract_boundary_with_As(R, ARs, nx)
    psi = np.tensordot(LC, R, axes=(tuple(range(Ny)), tuple(range(Ny))))
    psi_leg_order = []
    for nx in range(nx_c + 1):
        for y in range(Ly):
            x, p = nx//2, nx%2
            site_scalar = lattice.get_site_scalar((x, y, p))
            psi_leg_order.append(site_scalar)
    for nx in reversed(range(nx_c+1, Nx)):
        for y in range(Ly):
            x, p = nx//2, nx%2
            site_scalar = lattice.get_site_scalar((x, y, p))
            psi_leg_order.append(site_scalar)
    inverse_psi_leg_order = np.argsort(psi_leg_order)
    psi = np.transpose(psi, inverse_psi_leg_order)
    psi = np.reshape(psi, (2**N,))
    return psi 


class ExcitedIsometricPEPS:
    """Simple class for an excited isometric PEPS.
    
    isoPEPS(X,X_column) = sum_nx AL(1)...AL(nx-1) [sum_y VL(nx,y)X(nx,2y)] AR(nx+1)...AR(Nx) 
                          + AL(1)...AL(Nx) [sum_ny VD(Nx,ny)X_column(Nx,ny)].
    """
    def __init__(self, D_max, chi_max_c, ALs, ARs, CDs, CCs, CUs, vecX, vecX_column):
        # ground state iso_peps
        self.d = np.shape(ALs[0][0])[0]
        self.D_max = D_max
        self.chi_max_c = chi_max_c
        self.Nx = len(ALs)
        self.Lx = self.Nx // 2
        self.Ly = len(ALs[0])
        self.Ny = 2 * self.Ly - 1
        self.N = 2 * self.Lx * self.Ly
        self.ALs = ALs
        self.ARs = ARs
        self.CDs = CDs
        self.CCs = CCs
        self.CUs = CUs
        # excitations AL-VL-X-AR
        self.VLs = get_VLs(ALs)
        self.vecX = vecX
        self.shape_XDs, self.shape_Xs, self.shape_XUs, self.shape_vecX = get_shape_Xs_vecX(ALs, CDs, CCs, CUs)
        self.Xs = vec_to_tensors(vecX, self.shape_Xs)
        # excitations AL-AL-AL-X_column
        self.VDs = get_VDs(CDs[self.Nx-1])
        self.vecX_column = vecX_column
        self.shape_Xs_column, self.shape_vecX_column = get_shape_Xs_vecX_column(CDs[self.Nx-1])
        self.Xs_column = vec_to_tensors_column(self.vecX_column, self.shape_Xs_column)
        self.Bs_column = Xs_to_Bs_column(self.Xs_column, self.VDs)

    def print_all_excitation_norms(self):
        print("excitations AL-VL-X-AR:")
        for nx in range(self.Nx):
            for y in range(self.Ly):
                X = self.Xs[nx][2*y]
                if X is not None:
                    print(f"> {np.shape(X)} excitation parameters at site {nx,y} " \
                          + f"with ||X_{nx,y}||^2 = {np.linalg.norm(X)**2}.")
        print("excitations AL-AL-AL-X_column:")
        for ny in range(self.Ny):
            X_column = self.Xs_column[ny]
            if X_column is not None:
                print(f"> {np.shape(X_column)} excitation parameters on bond {self.Nx-1,ny} " \
                      + f"with ||X_column_{ny}||^2 = {np.linalg.norm(X_column)**2}.")
        X2 = np.linalg.norm(self.vecX)**2 + np.linalg.norm(self.vecX_column)**2
        print(f"-> {self.shape_vecX} + {self.shape_vecX_column} = {self.shape_vecX + self.shape_vecX_column} " \
              + f"excitation parameters with ||X||^2 + ||X_column||^2 = {X2}.")
        return

    def get_all_single_iso_peps(self):
        """Initialize the iso_peps for all excitation summands."""
        iso_peps_single_array = [([None] * self.Ly) for _ in range(self.Nx)] + [([None] * self.Ny)]
        # excitations AL-VL-X-AR
        for nx in range(self.Nx):
            for y in range(self.Ly):
                VL = self.VLs[nx][y]
                if VL is not None:
                    ALs_VL_single, ARs_single, Cs_noX_single = extract_single_excitation_configuration(nx, y, \
                                                               self.ALs, self.ARs, self.CDs, self.CCs, self.CUs)
                    Cs_X_single = deepcopy(Cs_noX_single)
                    Cs_X_single[2*y] = self.Xs[nx][2*y].copy()
                    iso_peps_single = isometric_peps_from_single_configuration(self.D_max, self.chi_max_c, \
                                      nx, 2*y, ALs_VL_single, ARs_single, Cs_X_single) 
                    iso_peps_single_array[nx][y] = iso_peps_single
        # excitations AL-AL-AL-X_column
        for ny in range(self.Ny):
            B = self.Bs_column[ny]
            if B is not None:
                ALs_single, ARs_single, Cs_noB_single = extract_single_excitation_configuration_column(ny, \
                                                        self.ALs, self.ARs, self.CDs, self.CCs, self.CUs)
                Cs_B_single = deepcopy(Cs_noB_single)
                Cs_B_single[ny] = B
                iso_peps_single = isometric_peps_from_single_configuration(self.D_max, self.chi_max_c, \
                                  self.Nx-1, ny, ALs_single, ARs_single, Cs_B_single)
                iso_peps_single_array[self.Nx][ny] = iso_peps_single
        return iso_peps_single_array
    
    @classmethod
    def optimized_from_excited_wavefunction(cls, iso_peps0, psi):
        """Excite the ground state iso_peps0 by optimizing each perturbation from the overlap with 
        a full wavefunction."""
        print("Optimize excited isoPEPS from full wavefunction psi.")
        # lattice parameters
        Lx = iso_peps0.Lx
        Ly = iso_peps0.Ly
        Nx = 2 * Lx
        Ny = 2 * Ly - 1
        # ground state iso_peps -> excited iso_peps
        ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
        print("excitations AL-VL-X-AR:")
        _, shape_Xs, _, shape_vecX = get_shape_Xs_vecX(ALs, CDs, CCs, CUs)
        Xs = [[None] * Ny for _ in range(Nx)]
        for nx in range(Nx):
            for y in range(Ly):
                shape_X = shape_Xs[nx][2*y]
                if shape_X is not None:
                    ALs_VL_single, ARs_single, Cs_noX_single = extract_single_excitation_configuration(nx, y, \
                                                               ALs, ARs, CDs, CCs, CUs)
                    X = get_overlap_wavefunction_iso_peps(psi, None, nx, 2*y, \
                                                          ALs_VL_single, ARs_single, Cs_noX_single, \
                                                          deriv_after_CC=True)
                    Xs[nx][2*y] = X
                    print(f"> optimized {shape_X} excitation parameters at site {nx,y} " \
                          + f"with ||X_{nx,y}||^2 = {np.linalg.norm(X)**2}.")
        vecX = tensors_to_vec(Xs, shape_vecX)
        print(f"-> optimized {shape_vecX} excitation parameters " \
              + f"with ||X||^2 = sum_(nx,y) ||X_(nx,y)||^2 = {np.linalg.norm(vecX)**2}.")
        print("excitations AL-AL-AL-X_column:")
        VDs = get_VDs(CDs[Nx-1])
        shape_Xs_column, shape_vecX_column = get_shape_Xs_vecX_column(CDs[Nx-1])
        Bs_column = [None] * Ny
        for ny in range(Ny):
            shape_X_column = shape_Xs_column[ny]
            if shape_X_column is not None:
                ALs_single, ARs_single, Cs_noB_single = extract_single_excitation_configuration_column(ny, \
                                                        ALs, ARs, CDs, CCs, CUs)
                B = get_overlap_wavefunction_iso_peps(psi, None, Nx-1, ny, \
                                                      ALs_single, ARs_single, Cs_noB_single, \
                                                      deriv_after_CC=True)  
                Bs_column[ny] = B
                X = np.tensordot(np.conj(VDs[ny]), B, axes=((0, 1, 2), (0, 1, 2)))
                print(f"> optimized {shape_X_column} excitation parameters at bond {Nx-1,ny} " \
                      + f"with ||X_column_{ny}||^2 = {np.linalg.norm(X)**2}.")
        Xs_column = Bs_to_Xs_column(Bs_column, VDs)
        vecX_column = tensors_to_vec_column(Xs_column, shape_vecX_column)     
        print(f"-> optimized {shape_vecX_column} excitation parameters " \
              + f"with ||X_column||^2 = sum_ny ||X_column_ny||^2 = {np.linalg.norm(vecX_column)**2}.") 
        norm = np.sqrt(np.linalg.norm(vecX)**2 + np.linalg.norm(vecX_column)**2)
        print(f"=> |<psi|iso_peps(X,X_column)>| = (||X||^2 + ||X_column||^2)^1/2 = {norm}.") 
        vecX /= norm
        vecX_column /= norm    
        excited_iso_peps = cls(iso_peps0.D_max, iso_peps0.chi_max, ALs, ARs, CDs, CCs, CUs, \
                               vecX, vecX_column)
        return excited_iso_peps
    
    @classmethod
    def optimized_from_excited_mps(cls, iso_peps0, emps):
        """Excite the ground state iso_peps0 by optimizing each perturbation from the overlap with 
        an excited mps, itself being a sum over local perturbations."""
        print("Optimize excited isoPEPS from MPS.")
        # lattice parameters
        Lx = iso_peps0.Lx
        Ly = iso_peps0.Ly
        Nx = 2 * Lx
        Ny = 2 * Ly - 1
        N = 2 * Lx * Ly
        # excited mps
        assert emps.N == N
        Ms_single_list = emps.get_all_single_canonical_configurations()
        # ground state iso_peps -> excited iso_peps
        ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
        print("excitations AL-VL-X-AR:")
        _, shape_Xs, _, shape_vecX = get_shape_Xs_vecX(ALs, CDs, CCs, CUs)
        Xs = [[None] * Ny for _ in range(Nx)]
        for nx in range(Nx):
            for y in range(Ly):
                shape_X = shape_Xs[nx][2*y]
                if shape_X is not None:
                    ALs_VL_single, ARs_single, Cs_noX_single = extract_single_excitation_configuration(nx, y, \
                                                               ALs, ARs, CDs, CCs, CUs)
                    X = np.zeros(shape_X, dtype=np.complex128) 
                    for n in range(N):
                        Ms_single = Ms_single_list[n]
                        if Ms_single is not None:
                            X += get_overlap_mps_iso_peps(Ms_single, None, nx, 2*y, \
                                                          ALs_VL_single, ARs_single, Cs_noX_single, \
                                                          deriv_after_CC=True)
                    Xs[nx][2*y] = X
                    print(f"> optimized {shape_X} excitation parameters at site {nx,y} " \
                          + f"with ||X_{nx,y}||^2 = {np.linalg.norm(X)**2}.")
        vecX = tensors_to_vec(Xs, shape_vecX)
        print(f"-> optimized {shape_vecX} excitation parameters " \
              + f"with ||X||^2 = sum_(nx,y) ||X_(nx,y)||^2 = {np.linalg.norm(vecX)**2}.")
        print("excitations AL-AL-AL-X_column:")
        VDs = get_VDs(CDs[Nx-1])
        shape_Xs_column, shape_vecX_column = get_shape_Xs_vecX_column(CDs[Nx-1])
        Bs_column = [None] * Ny
        for ny in range(Ny):
            shape_X_column = shape_Xs_column[ny]
            if shape_X_column is not None:
                ALs_single, ARs_single, Cs_noB_single = extract_single_excitation_configuration_column(ny, \
                                                        ALs, ARs, CDs, CCs, CUs)
                shape_VD = np.shape(VDs[ny])
                shape_X = shape_Xs_column[ny]
                shape_B = (shape_VD[0], shape_VD[1], shape_VD[2], shape_X[1])
                B = np.zeros(shape_B, dtype=np.complex128) 
                for n in range(N):
                    Ms_single = Ms_single_list[n]
                    if Ms_single is not None:
                        B += get_overlap_mps_iso_peps(Ms_single, None, Nx-1, ny, \
                                                      ALs_single, ARs_single, Cs_noB_single, \
                                                      deriv_after_CC=True)  
                Bs_column[ny] = B  
                X = np.tensordot(np.conj(VDs[ny]), B, axes=((0, 1, 2), (0, 1, 2)))
                print(f"> optimized {shape_X_column} excitation parameters at bond {Nx-1,ny} " \
                      + f"with ||X_column_{ny}||^2 = {np.linalg.norm(X)**2}.")
        Xs_column = Bs_to_Xs_column(Bs_column, VDs)
        vecX_column = tensors_to_vec_column(Xs_column, shape_vecX_column)     
        print(f"-> optimized {shape_vecX_column} excitation parameters " \
              + f"with ||X_column||^2 = sum_ny ||X_column_ny||^2 = {np.linalg.norm(vecX_column)**2}.")
        norm = np.sqrt(np.linalg.norm(vecX)**2 + np.linalg.norm(vecX_column)**2)
        print(f"=> |<mps|iso_peps(X,X_column)>| = (||X||^2 + ||X_column||^2)^1/2 = {norm}") 
        vecX /= norm
        vecX_column /= norm    
        excited_iso_peps = cls(iso_peps0.D_max, iso_peps0.chi_max, ALs, ARs, CDs, CCs, CUs, \
                               vecX, vecX_column)
        return excited_iso_peps, norm
    
    def get_overlap_with_excited_wavefunction(self, psi):
        """Compute the overlap of the excited iso_peps with a full wavefunction psi."""
        single_iso_peps_array = self.get_all_single_iso_peps()
        overlap = 0.
        # excitations AL-VL-X-AR
        for nx in range(self.Nx):
            for y in range(self.Ly):
                single_iso_peps = single_iso_peps_array[nx][y]
                if single_iso_peps is not None:  
                    overlap += get_overlap_wavefunction_iso_peps(psi, single_iso_peps)
        # excitations AL-AL-AL-X_column
        for ny in range(self.Ny):
            single_iso_peps = single_iso_peps_array[self.Nx][ny]
            if single_iso_peps is not None:  
                overlap += get_overlap_wavefunction_iso_peps(psi, single_iso_peps)
        return overlap 
    
    def get_overlap_with_excited_mps(self, emps):
        """Compute the overlap of the excited iso_peps with an excited mps, itself being a sum of
        local perturbations."""
        assert emps.N == self.N
        Ms_single_list = emps.get_all_single_canonical_configurations()
        single_iso_peps_array = self.get_all_single_iso_peps()
        overlap = 0.
        # excitations AL-VL-X-AR
        for nx in range(self.Nx):
            for y in range(self.Ly):
                single_iso_peps = single_iso_peps_array[nx][y]
                if single_iso_peps is not None:  
                    for n in range(self.N):
                        Ms_single = Ms_single_list[n]
                        if Ms_single is not None:
                            overlap += get_overlap_mps_iso_peps(Ms_single, single_iso_peps) 
                    print(f"({nx}, {y}) done, overlap = {overlap}.")
        # excitations AL-AL-AL-X_column
        for ny in range(self.Ny):
            single_iso_peps = single_iso_peps_array[self.Nx][ny]
            if single_iso_peps is not None:  
                for n in range(self.N):
                    Ms_single = Ms_single_list[n]
                    if Ms_single is not None:
                        overlap += get_overlap_mps_iso_peps(Ms_single, single_iso_peps)
        return overlap 
    
    def transform_into_wavefunction(self):
        iso_peps_single_array = self.get_all_single_iso_peps()
        psis_single = []
        # excitations AL-VL-X-AR
        for nx in range(self.Nx):
            for y in range(self.Ly):
                if iso_peps_single_array[nx][y] is not None:
                    psis_single.append(get_wavefunction(iso_peps_single_array[nx][y]))
        # excitations AL-AL-AL-X_column
        for ny in range(self.Ny):
            if iso_peps_single_array[self.Nx][ny] is not None:
                psis_single.append(get_wavefunction(iso_peps_single_array[self.Nx][ny]))
        psi = sum(psis_single)
        return psi        
    
    def get_energy(self, H):
        """Compute all single iso_peps configurations, convert them into wavefunctions, compute
        their pairwise Hamiltonian overlaps <psi|H|phi> and sum them up to the total energy."""
        psi = self.transform_into_wavefunction()
        E = np.inner(np.conj(psi), H @ psi)
        return np.real_if_close(E) 

    def get_bond_energies(self, H_bonds):
        psi = self.transform_into_wavefunction()
        es_bond = [[None] * (2*self.Ly-1) for _ in range(2*self.Lx-1)]
        for bx in range(2*self.Lx-1):
            for by in range(2*self.Ly-1):
                es_bond[bx][by] = np.inner(np.conj(psi), H_bonds[bx][by] @ psi)
        return np.real_if_close(es_bond)

    def extract_Xs_form2(self):
        """Reshape all parameterizations Xs of the excitations AL-VL-X-AR into the form 2 given by

           chi_u Druu
              | /
              |/
        Dr---(X)       with order (chi_d, chi_u, Dr, Drdd, Druu).
              |\
              | \
           chi_d Drdd

        The shapes (chi_d, chi_u) of the parameterization Xs_column of the excitations AL-AL-AL-X
        are left unchanged.
        """
        Xs_form2 = [[None] * self.Ly for _ in range(self.Nx)]
        for nx in range(self.Nx):
            for ny in range(self.Ny):
                if self.Xs[nx][ny] is not None:
                    X_form1 = self.Xs[nx][ny].copy()
                    if nx%2 == 0:
                        if ny == 0:
                            chi_d, Drdd = 1, 1
                        elif ny > 0:
                            chi_d, _, Drdd, _ = self.shape_XDs[nx][ny-1]
                        _, Dr, Druu, chi_u = np.shape(X_form1)
                        X_form2 = np.reshape(X_form1, (chi_d, Drdd, Dr, Druu, chi_u))
                        X_form2 = np.transpose(X_form2, (0, 4, 2, 1, 3))
                    elif nx%2 == 1:
                        if ny < self.Ny-1:
                            _, _, Druu, chi_u = self.shape_XUs[nx][ny+1]
                        elif ny == self.Ny-1:
                            Druu, chi_u = 1, 1
                        chi_d, Dr, Drdd, _ = np.shape(X_form1)
                        X_form2 = np.reshape(X_form1, (chi_d, Dr, Drdd, Druu, chi_u))
                        X_form2 = np.transpose(X_form2, (0, 4, 1, 2, 3))
                    Xs_form2[nx][ny//2] = X_form2
        Xs_column_form2 = deepcopy(self.Xs_column)
        return Xs_form2, Xs_column_form2
    
    @classmethod
    def from_Xs_form2(cls, iso_peps0, Xs_form2, Xs_column_form2):
        """For ground state iso_peps0 and excitation parameterizatins Xs, Xs_column given in form 2,
        initialize a class instance by inversion of the reshaping done in extract_Xs_form2."""
        Nx = len(Xs_form2)
        Ly = len(Xs_form2[0])
        Ny = 2 * Ly -1
        # Xs
        Xs_form1 = [[None] * Ny for _ in range(Nx)]
        shape_vecX_form1 = 0
        for nx in range(Nx):
            for y in range(Ly):
                if Xs_form2[nx][y] is not None:
                    X_form2 = Xs_form2[nx][y].copy()
                    chi_d, chi_u, Dr, Drdd, Druu = np.shape(X_form2)
                    if nx%2 == 0:
                        X_form1 = np.transpose(X_form2, (0, 3, 2, 4, 1))
                        X_form1 = np.reshape(X_form1, (chi_d * Drdd, Dr, Druu, chi_u))
                    elif nx%2 == 1:
                        X_form1 = np.transpose(X_form2, (0, 2, 3, 4, 1))
                        X_form1 = np.reshape(X_form1, (chi_d, Dr, Drdd, Druu * chi_u))
                    Xs_form1[nx][2*y] = X_form1
                    shape_vecX_form1 += np.prod(np.shape(X_form1))
        vecX_form1 = tensors_to_vec(Xs_form1, shape_vecX_form1)
        # Xs_column
        Xs_column_form1 = deepcopy(Xs_column_form2)
        shape_vecX_column_form1 = 0
        for ny in range(Ny):
            if Xs_column_form1[ny] is not None:
                shape_vecX_column_form1 += np.prod(np.shape(Xs_column_form1[ny]))
        vecX_column_form1 = tensors_to_vec_column(Xs_column_form1, shape_vecX_column_form1)
        ALs, ARs, CDs, CCs, CUs = extract_all_isometric_configurations(iso_peps0)
        e_iso_peps = cls(iso_peps0.D_max, iso_peps0.chi_max, ALs, ARs, CDs, CCs, CUs, vecX_form1, vecX_column_form1)
        return e_iso_peps        