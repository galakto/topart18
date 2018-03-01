from scipy import integrate
from scipy.linalg import expm
import scipy as sp
from pylab import *
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp

""" Conventions:
B[i,a,b] has axes (physical, left virtual, right virtual),
W[a,b,i,j] has axes (virtual left, virtual right, physical out, physical in)
S[i] are schmidt values between sites (i, i+1),
H_bond[i] is the bond hamiltonian between (i,i+1) with (only physical)
axes (out left, out right, in left, in right)"""

def init_fm_mps(L):
    """ Return FM Ising MPS"""
    d = 2
    B = []
    s = []
    for i in range(L):
        B.append(np.zeros([2,1,1])); B[-1][0,0,0]=1
        s.append(np.ones([1]))
    s.append(np.ones([1]))
    return B,s

def init_ising_H_mpo(g,J,L):
    """ Returns hamiltonian in MPO form"""
    s0 = np.eye(2)
    sx = np.array([[0.,1.],[1.,0.]])
    sy = np.array([[0.,-1j],[1j,0.]])
    sz = np.array([[1.,0.],[0.,-1.]])
    d = 2

    w_list = []
    for i in range(L):
        w = np.zeros((3,3,d,d),dtype=np.float)
        w[0,:2] = [s0,sz]
        w[0:,2] = [g*sx, -J*sz, s0]
        w_list.append(np.real(w))
    return w_list

def init_ising_H_bond(g,J,L):
    """ Returns bond hamiltonian"""
    sx = np.array([[0.,1.],[1.,0.]])
    sy = np.array([[0.,-1j],[1j,0.]])
    sz = np.array([[1.,0.],[0.,-1.]])
    d = 2

    H_bond = []
    for i in range(L):
        H = -J*np.kron(sz,sz) + g*np.kron(sx,np.eye(2))
        H_bond.append(np.reshape(H,(d,d,d,d)))
    return H_bond

def bond_expectation(B,s,O_list):
    " Expectation value for a bond operator "
    E=[]
    L = len(B)
    for i_bond in range(L):
        BB = np.tensordot(B[i_bond],B[np.mod(i_bond+1,L)],axes=(2,1))
        sBB = np.tensordot(np.diag(s[np.mod(i_bond-1,L)]),BB,axes=(1,1))
        C = np.tensordot(sBB,O_list[i_bond],axes=([1,2],[2,3]))
        sBB=np.conj(sBB)
        E.append(np.squeeze(np.tensordot(sBB,C,axes=([0,3,1,2],[0,1,2,3]))).item())
    return E

def site_expectation(B,s,O_list):
    " Expectation value for a site operator "
    E=[]
    L = len(B)
    for isite in range(0,L):
        sB = np.tensordot(np.diag(s[np.mod(isite-1,L)]),B[isite],axes=(1,1))
        C = np.tensordot(sB,O_list[isite],axes=(1,0))
        sB=sB.conj()
        E.append(np.squeeze(np.tensordot(sB,C,axes=([0,1,2],[0,2,1]))).item())
    return(E)

def entanglement_entropy(s):
    " Returns the half chain entanglement entropy "
    S=[]
    L = len(s)
    for i_bond in range(L):
        x=s[i_bond][s[i_bond]>10**(-20)]**2
        S.append(-np.inner(np.log(x),x))
    return(S)

def correlation_length(B):
    " Constructs the mixed transfermatrix and returns correlation length"
    chi = B[0].shape[1]
    L = len(B)

    T = np.tensordot(B[0],np.conj(B[0]),axes=(0,0)) #a,b,a',b'
    T = T.transpose(0,2,1,3) #a,a',b,b'
    for i in range(1,L):
        T = np.tensordot(T,B[i],axes=(2,1)) #a,a',b',i,b
        T = np.tensordot(T,np.conj(B[i]),axes=([2,3],[1,0])) #a,a',b,b'
    T = np.reshape(T,(chi**2,chi**2))

    # Obtain the 2nd largest eigenvalue
    eta = arp.eigs(T,k=2,which='LM',return_eigenvectors=False,ncv=20)
    return -L/np.log(np.min(np.abs(eta)))

class H_mixed(object):
    def __init__(self,Lp,Rp,M1,M2,dtype=float):
        self.Lp = Lp
        self.Rp = Rp
        self.M1 = M1
        self.M2 = M2
        self.d = M1.shape[3]
        self.chi1 = Lp.shape[0]
        self.chi2 = Rp.shape[0]
        self.shape = np.array([self.d**2*self.chi1*self.chi2,self.d**2*self.chi1*self.chi2])
        self.dtype = dtype

    def matvec(self,x):
        x=np.reshape(x,(self.d,self.chi1,self.d,self.chi2)) # i a j b
        x=np.tensordot(self.Lp,x,axes=(0,1))                # ap m i j b
        x=np.tensordot(x,self.M1,axes=([1,2],[0,2]))        # ap j b mp ip
        x=np.tensordot(x,self.M2,axes=([3,1],[0,2]))        # ap b ip m jp
        x=np.tensordot(x,self.Rp,axes=([1,3],[0,2]))        # ap ip jp bp
        x=np.transpose(x,(1,0,2,3))
        x=np.reshape(x,((self.d*self.d)*(self.chi1*self.chi2)))
        if(self.dtype==float):
            return np.real(x)
        else:
            return(x)

def diag(B,s,H,ia,ib,ic,chia,chic):
    """ Diagonalizes the mixed hamiltonian """
    # Get a guess for the ground state based on the old MPS
    d = B[0].shape[0]
    theta0 = np.tensordot(np.diag(s[ia]),np.tensordot(B[ib],B[ic],axes=(2,1)),axes=(1,1))
    theta0 = np.reshape(np.transpose(theta0,(1,0,2,3)),((chia*chic)*(d**2)))

    # Diagonalize Hamiltonian
    e0,v0 = arp.eigsh(H,k=1,which='SA',return_eigenvectors=True,v0=theta0,ncv=20)
    return np.reshape(v0.squeeze(),(d*chia,d*chic))

def sweep(B,s,chi,H_mpo,Lp,Rp):
    """ One iDMRG sweep through unit cell"""
    d = B[0].shape[0]
    for i_bond in [0,1]:
        ia = np.mod(i_bond-1,2); ib = np.mod(i_bond,2); ic = np.mod(i_bond+1,2)
        chia = B[ib].shape[1]; chic = B[ic].shape[2]

        # Construct theta matrix #
        H = H_mixed(Lp,Rp,H_mpo[ib],H_mpo[ic])
        theta = diag(B,s,H,ia,ib,ic,chia,chic)

        # Schmidt deomposition #
        X, Y, Z = sp.linalg.svd(theta,lapack_driver='gesvd'); Z = Z.T

        chib = np.min([np.sum(Y>10.**(-8)), chi])
        X=np.reshape(X[:d*chia,:chib],(d,chia,chib))
        Z=np.transpose(np.reshape(Z[:d*chic,:chib],(d,chic,chib)),(0,2,1))

        # Update Environment #
        Lp = np.tensordot(Lp, H_mpo[ib], axes=(2,0))
        Lp = np.tensordot(Lp, X, axes=([0,3],[1,0]))
        Lp = np.tensordot(Lp, np.conj(X), axes=([0,2],[1,0]))
        Lp = np.transpose(Lp,(1,2,0))

        Rp = np.tensordot(H_mpo[ic], Rp, axes=(1,2))
        Rp = np.tensordot(np.conj(Z),Rp, axes=([0,2],[2,4]))
        Rp = np.tensordot(Z,Rp, axes=([0,2],[2,3]))

        # Obtain the new values for B and s #
        s[ib] = Y[:chib]/np.sqrt(sum(Y[:chib]**2))
        B[ib] = np.transpose(np.tensordot(np.diag(s[ia]**(-1)),X,axes=(1,1)),(1,0,2))
        B[ib] = np.tensordot(B[ib], np.diag(s[ib]),axes=(2,1))

        B[ic] = Z
    return Lp,Rp

def dmrg_ising(J,g,N,chi):
    B,s = init_fm_mps(2)
    H_bond = init_ising_H_bond(g,J,2)
    H_mpo = init_ising_H_mpo(g,J,2)
    sz = np.array([[1.,0.],[0.,-1.]])

    Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
    Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.

    for step in range(N):
        Lp,Rp = sweep(B,s,chi,H_mpo,Lp,Rp)

        E = np.mean(bond_expectation(B,s,H_bond))
        m = np.mean(site_expectation(B,s,[sz,sz]))
        S = np.mean(entanglement_entropy(s))
        xi = np.mean(correlation_length(B))

        print E,m,S,xi,"(DMRG)"
    
    return E,m,S,xi
    
if __name__ == "__main__":
    J = 1
    g = 0.5
    chi = 10
    N = 20
    
    E,m,S,xi = dmrg_ising(J,g,N,chi)
    
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
    print E0_exact,"(EXACT)"