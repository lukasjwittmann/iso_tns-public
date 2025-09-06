"""Toy code implementing the expectation value of an MPO acting on one column of a diagonal 
isometric PEPS, in different ways with and without compression."""

import numpy as np
import opt_einsum as oe

from .c_mps import MPS


def get_expectation_value_center(Cs, ALs, ARs, hs):
    """For orthogonality column tensors Cs between left and right isometric site tensors ALs and 
    ARs, compute the expectation value of the mpo hs without compression.
    
    e = <(AL)(h)(AL*)|(C)(C*)|(AR)(h)(AR*)>.

    We choose the following convention (shown for the case L=2):

                              .. (AR[1]h[3]AR[1]*)
                              ||/      / /        
                        ==(C[2]C[2]*)==           
                     / /     /||                   
            (AL[1]h[2]AL[1]*) ||                   
                     \ \     \||                  
    e   =               ==(C[1]C[1]*)==           
                              ||\      \ \        
                              || (AR[0]h[1]AR[0]*)
                              ||/      / /        
                        ==(C[0]C[0]*)==           
                     / /     /||                   
            (AL[0]h[0]AL[0]*) ..                  
    """
    assert len(ALs) == len(ARs) == (len(Cs)+1)//2 == len(hs)//2
    assert np.shape(ALs[0])[1] == np.shape(ALs[0])[3] \
           == np.shape(hs[0])[0] == np.shape(Cs[0])[0] == 1
    AL = ALs[0][:, 0, :, 0, :]
    h = hs[0][0, :, :, :]
    C = Cs[0][0, :, :, :]
    DP = oe.contract("abc,dea,ebf,cgh,fij->gdihj", \
                     AL, h, np.conj(AL), C, np.conj(C))
    assert np.shape(ARs[-1])[2] == np.shape(ARs[-1])[4] \
           == np.shape(hs[-1])[1] == np.shape(Cs[-1])[3] == 1
    AR = ARs[-1][:, :, 0, :, 0] 
    h = hs[-1][:, 0, :, :]
    UP = oe.contract("abc,dea,ebf->cdf", \
                     AR, h, np.conj(AR))
    for n in range(len(ALs)-1):
        DP = oe.contract("abcde,fghai,bjkf,kghcl,dmin,eolp->mjonp", \
                         DP, ARs[n], hs[(2*n)+1], np.conj(ARs[n]), \
                         Cs[(2*n)+1], np.conj(Cs[(2*n)+1]))
        DP = oe.contract("abcde,fghai,bjkf,kghcl,dimn,elop->mjonp", \
                         DP, ALs[n+1], hs[2*(n+1)], np.conj(ALs[n+1]), \
                         Cs[2*(n+1)], np.conj(Cs[2*(n+1)]))
    e = oe.contract("abcde,abc->de", \
                    DP, UP)
    assert np.shape(e) == (1, 1)
    e = e[0, 0]
    return np.real_if_close(e)

def get_expectation_value_side(Cs, A1s, A2s, hs):
    """For orthogonality column tensors Cs right of left isometric site tensors A1s, A2s, compute 
    the expectation value of the mpo hs without compression.
    
    e = <(A1A2)(h)(A1A2*)|(C)(C*)>.

    We choose the following convention (shown for the case L=2):

                        (A2[1]h[3]A2[1]*)     ..
                         / / /       \ \      ||       
                        / / /           ==(C[2]C[2]*)==           
                       / / /                  ||                   
            (A1[1]h[2]A1[1]*)                 ||                   
                       \ \ \                  ||                  
    e   =               \ \ \            ==(C[1]C[1]*)==           
                         \ \ \       / /      ||        
                        (A2[0]h[1]A2[0]*)     ||
                         / / /       \ \      ||       
                        / / /            ==(C[0]C[0]*)==           
                       / / /                  ||                   
            (A1[0]h[0]A1[0]*)                 ..    
    """  
    assert len(A1s) == len(A2s) == (len(Cs)+1)//2 == len(hs)//2
    assert np.shape(A1s[0])[1] == np.shape(A1s[0])[3] \
           == np.shape(hs[0])[0] == np.shape(Cs[0])[0] == 1
    A1 = A1s[0][:, 0, :, 0, :]
    h = hs[0][0, :, :, :]
    DP = oe.contract("abc,dea,ebf->cdf", \
                     A1, h, np.conj(A1))
    DP = DP[:, :, :, np.newaxis, np.newaxis]
    assert np.shape(A2s[-1])[2] == np.shape(A2s[-1])[4] \
           == np.shape(hs[-1])[1] == np.shape(Cs[-1])[3] == 1
    A2 = A2s[-1][:, :, 0, :, 0] 
    h = hs[-1][:, 0, :, :]
    C = Cs[-1][:, :, :, 0]
    UP = oe.contract("abc,dea,efg,hci,jgi->bdfhj", \
                     A2, h, np.conj(A2), C, np.conj(C))
    for n in range(len(A1s)-1):
        DP = oe.contract("abcde,faghi,bjkf,kclmn,dhop,emoq,pirs,qnrt->gjlst", \
                         DP, A2s[n], hs[(2*n)+1], np.conj(A2s[n]), \
                         Cs[2*n], np.conj(Cs[2*n]), Cs[2*n+1], np.conj(Cs[2*n+1])) 
        DP = oe.contract("abcde,fghai,bjkf,kghcl->ijlde", \
                         DP, A1s[n+1], hs[2*(n+1)], np.conj(A1s[n+1])) 
    e = oe.contract("abcde,abcde->", \
                    DP, UP)  
    return np.real_if_close(e)  

def get_expectation_value_boundary(Cs, Bh, side):
    """For orthogonality column tensors Cs, compute the expectation value of mpos contained in 
    compressed boundary mps Bh on side "left" or "right".
    
    e = <(Bh)|(C)(C*)>.

               .         ..                 ..          .
               |         ||                 ||          |
            (Lh[2])==(C[2]C[2]*)        (C[2]C[2]*)==(Rh[2])
               |         ||                 ||          |
               |         ||                 ||          |
               |         ||                 ||          |
    e   =   (Lh[1])==(C[1]C[1]*)   or   (C[1]C[1]*)==(Rh[1])   (for the case L=2)
               |         ||                 ||          |
               |         ||                 ||          |
               |         ||                 ||          |
            (Lh[0])==(C[0]C[0]*)        (C[0]C[0]*)==(Rh[0])
               |         ||                 ||          |
               .         ..                 ..          .    

               side="left"                   side="right"
    """
    Ms = Bh.Ms
    assert len(Ms) == len(Cs)
    assert np.shape(Ms[0])[0] == np.shape(Cs[0])[0] == 1
    if side == "right":
        Cs = [np.transpose(C, (0, 2, 1, 3)) for C in Cs]
    e = oe.contract("abc,ade,bdf->cef", \
                    Ms[0][0, :, :, :], Cs[0][0, :, :, :], np.conj(Cs[0][0, :, :, :]))
    for n in range(1, len(Ms)):
        e = oe.contract("abc,adef,bdgh,cegi->fhi", \
                        e, Ms[n], Cs[n], np.conj(Cs[n]))
    assert np.shape(e) == (1, 1, 1)
    e = Bh.norm * e[0, 0, 0]
    return np.real_if_close(e)

def get_expectation_value_sum(Cs, Lh, Rh, ALs, ARs, hs):
    """For orthogonality column tensors Cs, compute the expectation value from the (possibly) three 
    contributions left, right and center."""
    E = 0.
    EL = None
    if Lh is not None:
        EL = get_expectation_value_boundary(Cs, Lh, side="left")
        E += EL
    EC = None
    if hs is not None:
        EC = get_expectation_value_center(Cs, ALs, ARs, hs)
        E += EC
    ER = None
    if Rh is not None:
        ER = get_expectation_value_boundary(Cs, Rh, side="right")
        E += ER
    return E, EL, EC, ER


def subtract_energy_offset_mpo(hs, e):
    """From mpo hs, subtract the energy e evenly over all sites."""
    N = len(hs)
    Id = np.eye(2)
    for n in range(N-1):
        hs[n][0, 2] -= (e/N) * Id
    hs[N-1][0, 0] -= (e/N) * Id
    return 

def subtract_energy_offset_mpos(h_mpos, es):
    """From column mpos h_mpos, subtract the column energies es evenly over all sites."""
    assert len(h_mpos) == len(es)
    Nx = len(h_mpos)
    for nx in range(Nx):
        subtract_energy_offset_mpo(h_mpos[nx], es[nx])
    return


def get_flipped_As(As):
    """For a list of site tensors As, reverse the order and flip every tensor A vertically.

    2       4      1       3
     \  0  /        \  0  /
      \ | /          \ | / 
       (A)     ->     (A)
      /   \          /   \
     /     \        /     \
    1       3      2       4

    p ld lu rd ru -> p lu ld ru rd
    """
    if As is not None:
        return [np.transpose(A, (0, 2, 1, 4, 3)) for A in As[::-1]]
    else:
        return None

def get_flipped_hs(hs):
    """For a list of mpo tensors hs, reverse the order and flip every tensor h vertically.

         1           0
     2  /        2  /
     | /         | /
    (h)     ->  (h)
     | \         | \
     3  \        3  \
         0           1

    rd ru p p* -> ru rd p p*
    """
    if hs is not None:
        return [np.transpose(h, (1, 0, 2, 3)) for h in hs[::-1]]
    else:
        return None

def get_flipped_Cs(Cs):
    """For a list of orthogonality column or boundary mps tensors Cs, reverse the order and flip 
    every tensor C vertically.

         3                0
         |                |
         |                |
    1---(C)---2  ->  1---(C)---2
         |                |
         |                |
         0                3

    d l r u -> u l r d
    """
    if Cs is not None:
        return [np.transpose(C, (3, 1, 2, 0)) for C in Cs[::-1]]
    else:
        return None

def get_flipped_mps(mps):
    """Initialize a new MPS instance with the tensors of mps, but in reversed order and flipped
    vertically."""
    if mps is not None:
        Ms_flipped = get_flipped_Cs(mps.Ms)
        return MPS(Ms_flipped, mps.norm)
    else:
        return None