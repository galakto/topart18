from scipy import integrate
from scipy.linalg import expm
import pylab as pl
import numpy as np
import scipy as sp
import scipy.sparse.linalg.eigen.arpack as arp
from scipy.sparse.linalg import eigs
np.set_printoptions(linewidth=2000, precision=3,threshold=4000)

""" Conventions:
B[i,a,b] has axes (physical, left virtual, right virtual),
W[a,b,i,j] has axes (virtual left, virtual right, physical out, physical in)
S[i] are schmidt values between sites (i-1, i+1),
H_bond[i] is the bond hamiltonian between (i,i+1) with (only physical)
axes (out left, out right, in left, in right)"""

def init_cdw_mps(L):
    """ Returns FM Ising MPS"""
    d = 2
    B = []
    s = []
    for i in range(L):
        B.append(np.zeros([2,1,1])); B[-1][i%L,0,0]=1
        s.append(np.ones([1]))
    s.append(np.ones([1]))
    return B,s

def init_rice_mele_U_bond(t,u,V,delta):
    """ Returns bond hamiltonian and bond time-evolution"""
    bp = np.array([[0.,0.],[1.,0.]])
    bm = bp.T
    n = np.dot(bp,bm)
    d = 2

    U_bond = []
    H_bond = []
    for i in range(2):
        H = t[i]*(np.kron(bp,bm) + np.kron(bm,bp)) + u[i]*np.kron(n,np.eye(2)) + V*np.kron(n,n)
        H_bond.append(np.reshape(H,(d,d,d,d)))
        U_bond.append(np.reshape(expm(-delta*H),(d,d,d,d)))

    return U_bond,H_bond
    
def sweep(B,s,U_bond,chi):
    """ Perform the imaginary time evolution of the MPS """
    L = len(B)
    d = B[0].shape[0]
    for k in [0,1]:
        for i_bond in range(k, L,2):
            ia = i_bond
            ib = np.mod(i_bond+1,L)
            ic = np.mod(i_bond+2,L)
            chia = B[ia].shape[1]
            chic = B[ib].shape[2]

            # Construct theta matrix and time evolution #
            theta = np.tensordot(B[ia],B[ib],axes=(2,1)) # i a j b
            theta = np.tensordot(U_bond[i_bond],theta,axes=([2,3],[0,2])) # ip jp a b
            theta = np.tensordot(np.diag(s[ia]),theta,axes=([1,2])) # a ip jp b
            theta = np.reshape(np.transpose(theta,(1,0,2,3)),(d*chia,d*chic)) # ip a jp b

            # Schmidt deomposition #
            X, Y, Z = sp.linalg.svd(theta,full_matrices=0,lapack_driver='gesvd')
            chi2 = np.min([np.sum(Y>10.**(-15)), chi])

            piv = np.zeros(len(Y), np.bool)
            piv[(np.argsort(Y)[::-1])[:chi2]] = True

            Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
            X = X[:,piv]
            Z = Z[piv,:]

            # Obtain the new values for B and s #
            s[ib] = Y/invsq

            X=np.reshape(X,(d,chia,chi2))
            X = np.transpose(np.tensordot(np.diag(s[ia]**(-1)),X,axes=(1,1)),(1,0,2))
            B[ia] = np.tensordot(X, np.diag(s[ib]),axes=(2,0))

            B[ib] = np.transpose(np.reshape(Z,(chi2,d,chic)),(1,0,2))

def n_right(B,eps=0.1,N=101):
    s0 = np.eye(2)
    sz = np.array([[0.5, 0.0],
                   [ 0.0, -0.5]])
                
    Rp = np.tensordot(np.eye(B[1].shape[2]),np.array([0,1]),axes=0) # a a' m
    
    for i in range(1,N):
        M = np.zeros((2,2,2,2))
        M[0,0] = s0
        M[0:,1] = [sz - np.exp(-eps*i)*sz,s0]
        Rp = np.tensordot(M, Rp, axes=(1,2)) #m i i' a a'
        Rp = np.tensordot(np.conj(B[i%2]),Rp, axes=([0,2],[2,4])) # a' m i a
        Rp = np.tensordot(B[i%2],Rp, axes=([0,2],[2,3])) # a a' m
    
    return np.diag(Rp[:,:,0])
    
def tebd_rice_mele(t,u,V,N,B,s,chi):
    for delta in [0.1,0.01,0.001,0]:
        U_bond,H_bond = init_rice_mele_U_bond(t,u,V,delta)
        for i in range(int(N)):
            sweep(B,s,U_bond,chi)
                
    return B,s
    
if __name__ == "__main__":
    chi = 10
    N = 10    
    V = 0.
    t0 = 1
    
    B,s = init_cdw_mps(2)
    phi_list = np.arange(0,2*np.pi,0.025)
    n = []
    
    du = []
    dt = []
    
    for phi in phi_list:
        u = [np.sin(phi),-np.sin(phi)]
        t = [1.0, t0 + np.cos(phi)]

        du.append(u[0]-u[1])
        dt.append(t[0]-t[1])
        
        B,s = tebd_rice_mele(t,u,V,N,B,s,chi)
        nr = n_right(B); nr = nr-nr[0]
        n.append(np.sum((s[0]**2)*nr))
        print "phi=",phi,"n_right=",n[-1]-n[0]

    pl.figure(figsize=(9,3))
    pl.subplot(121)
    pl.plot(np.array(phi_list)/2./np.pi,n-n[0])
    pl.xlabel('$\\phi/2\\pi$')
    pl.ylabel('$\\Delta Q$')
    pl.subplot(122)
    pl.scatter(du, dt)
    pl.scatter(0,0)
    pl.xlabel('$\Delta u$')
    pl.ylabel('$\Delta t$')
    pl.show()    
