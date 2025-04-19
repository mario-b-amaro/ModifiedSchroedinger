"""
SOLVER FOR TIME-INDEPENDENT DIRAC EQUATION
Method: Finite Difference Method 
Author: Mario B. Amaro
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
import time

beg=time.time()

"""
TODO: Verfiy matrix construction - are potentials in the right spot?
"""

# -------------------------------------------------------------------------
#         PARAMETER DEFINITION (EDIT AT WILL)
# -------------------------------------------------------------------------  

# Solve Domain

r_min,r_max= 1, 10000 # fm/(hbar c)
N=1000
r=np.linspace(r_min,r_max,N)
dr=r[1]-r[0]

# Eigenvalues

eigs=int(N) # No. of Eigenvalues to solve for
eigs_int=2*N # No. of Eigenvalues of interest

# ------------------- VECTOR POTENTIAL ---------------------

m=0.51099895069 # Electron

l=0
s=0.5 # OBS: 0.5 or -0.5
j=l+s

if s==0.5:
    k=(l+1)
elif s==-0.5:
    k=-l
else:
    sys.exit("Invalid value of s (must be 0.5 or -0.5)")
    
Z=1
N_=1
A=Z+N_

# Just a Coulomb Potential - adapt as needed
    
e=0.30282212088
hbar=1
epsilon_0=1
c=1

# Soft-core Coulomb defintion

V=-e**2/(4*np.pi*epsilon_0*hbar*c)*(Z/r) # Typical Coulomb

# ------------------- SCALAR POTENTIAL ---------------------

S=0 # Scalar Potential

# ------------------- POTENTIALS ---------------------

Sigma=V+S
Delta=V-S

dDelta=np.gradient(Delta,dr)

# -----------------------------------------------------------------------------
#              POLYNOMIAL EIGENVALUE MATRICES (DON'T EDIT)
# -----------------------------------------------------------------------------   

print("Constructing Matrices...")

M=np.eye(N-2)

C=np.zeros((N-2)**2).reshape((N-2),(N-2))
for i in range((N-2)):
    if i==0:
        C[i,i]=2*m-Sigma[i+1]-Delta[i+1]-(k/(4*m**2))*dDelta[i+1]
        C[i,i+1]=dDelta[i+1]/(8*m**2*dr)
    elif i==(N-2)-1:
        C[i,i]=2*m-Sigma[i+1]-Delta[i+1]-(k/(4*m**2))*dDelta[i+1]
        C[i,i-1]=-dDelta[i+1]/(8*m**2*dr)
    else:
        C[i,i-1]=-dDelta[i+1]/(8*m**2*dr)
        C[i,i]=2*m-Sigma[i+1]-Delta[i+1]-(k/(4*m**2))*dDelta[i+1]
        C[i,i+1]=dDelta[i+1]/(8*m**2*dr)      
        

K=np.zeros((N-2)**2).reshape((N-2),(N-2))
for i in range((N-2)):
    if i==0:
        K[i,i]=-2*m*Sigma[i+1]+Delta[i+1]*Sigma[i+1]-(k/(2*m))*dDelta[i+1]+(k*Delta[i+1]/(4*m**2))*dDelta[i+1]-(l*(l+1))/r[i+1]**2-2/dr**2
        K[i,i+1]=dDelta[i+1]/(4*m*dr)-(Delta[i+1]*dDelta[i+1])/(8*m**2*dr)+1/dr**2
    elif i==(N-2)-1:
        K[i,i]=-2*m*Sigma[i+1]+Delta[i+1]*Sigma[i+1]-(k/(2*m))*dDelta[i+1]+(k*Delta[i+1]/(4*m**2))*dDelta[i+1]-(l*(l+1))/r[i+1]**2-2/dr**2
        K[i,i-1]=-dDelta[i+1]/(4*m*dr)+(Delta[i+1]*dDelta[i+1])/(8*m**2*dr)+1/dr**2
    else:
        K[i,i-1]=-dDelta[i+1]/(4*m*dr)+(Delta[i+1]*dDelta[i+1])/(8*m**2*dr)+1/dr**2
        K[i,i]=-2*m*Sigma[i+1]+Delta[i+1]*Sigma[i+1]-(k/(2*m))*dDelta[i+1]+(k*Delta[i+1]/(4*m**2))*dDelta[i+1]-(l*(l+1))/r[i+1]**2-2/dr**2
        K[i,i+1]=dDelta[i+1]/(4*m*dr)-(Delta[i+1]*dDelta[i+1])/(8*m**2*dr)+1/dr**2   

print("Matrices Constructed.")

# ----------------------------------------------------------------------------
#            SOLVE POLYNOMIAL EIGENVALUE PROBLEM (DON'T EDIT)
# ----------------------------------------------------------------------------    

def polyeig(*A):

    # Adapted from https://stackoverflow.com/questions/8252428/how-to-solve-the-polynomial-eigenvalue-in-python/65516134#65516134

    """
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x=0 

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X,e = polyeig(A0,A1,..,Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X,e = polyeig(K,C,M)

    """
    
    if len(A)<=0:
        raise Exception('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise Exception('All matrices must have the same shapes');

    n = A[0].shape[0]
    l = len(A)-1 

    C = np.block([
        [np.zeros((n*(l-1),n)), np.eye(n*(l-1))],
        [-np.column_stack( A[0:-1])]
        ])
    D = np.block([
        [np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
        [np.zeros((n, n*(l-1))), A[-1]          ]
        ]);

    e, X = sp.linalg.eig(C, D);
    if np.all(np.isreal(e)):
        e=np.real(e)
    X=X[:n,:]

    I = np.argsort(e)
    X = X[:,I]
    e = e[I]

    # Normalize wavefunctions
    
    #X /= np.tile(np.max(np.abs(X),axis=0), (n,1))

    return X, e

vec,val=polyeig(K,C,M)

print("Eigenproblem Solved.")

print("Runtime:",time.time()-beg)

# -----------------------------------------------------------------------
#       PRINT EIGENVALUES OF INTEREST IN ORDER - FOR TESTING
# -----------------------------------------------------------------------
"""
val,vec=np.array(val),np.array(vec)
z = np.argsort(val)
z = z[0:eigs_int]

plt.figure(figsize=(12,10))
for i in range(eigs_int):
    y = []
    y = np.append(y,vec[:,z[i]])
    y = np.append(y,0)
    y = np.insert(y,0,0)
    rho=np.real(y*np.conj(y))
    plt.plot(r,rho,lw=3)
    plt.xlabel('x', size=20)
    plt.ylabel('$\psi\psi*$',size=20)
plt.title('"Standard" Probability Density',size=20)
plt.grid()
plt.show()

plt.figure(figsize=(12,10))
for i in range(eigs_int):
    y = []
    y = np.append(y,vec[:,z[i]])
    y = np.append(y,0)
    y = np.insert(y,0,0)
    rho=np.real(y*np.conj(y)+(1/(4*m**2))*np.gradient(y,dr)*np.conj(np.gradient(y,dr)))
    plt.plot(r,rho,lw=3)
    plt.xlabel('x', size=20)
    plt.ylabel(r'$\psi\psi*+\frac{1}{4m^2}\nabla\psi\nabla\psi*$',size=20)
plt.title('Corrected Probability Density',size=20)
plt.grid()
plt.show()

plt.figure(figsize=(12,10))
for i in range(eigs_int):
    y = []
    y = np.append(y,vec[:,z[i]])
    y = np.append(y,0)
    y = np.insert(y,0,0)
    rho=np.real(y*np.conj(y))
    rho_c=np.real(y*np.conj(y)+(1/(4*m**2))*np.gradient(y,dr)*np.conj(np.gradient(y,dr)))
    plt.plot(r,rho,'--k',lw=1)
    plt.plot(r,rho_c,'-k',lw=1)
    plt.xlabel('x', size=20)
    plt.ylabel('Probability Density',size=20)
    plt.xscale('log')
    plt.grid()
    plt.show()

    plt.plot(r,rho_c-rho,'-k',lw=1)
    plt.xlabel('x', size=20)
    plt.ylabel('Correction',size=20)
    plt.xscale('log')
    plt.grid()
    plt.show()

"""
# -----------------------------------------------------------------------
#       PLOT AGAINST STANDARD SCHRÖDINGER
# -----------------------------------------------------------------------

H=np.zeros((N-2)**2).reshape(N-2,N-2)
for i in range(N-2):
    for j in range(N-2):
        if i==j:
            H[i,j]=V[i+1]+(hbar**2/(m*dr**2)+(hbar**2*l*(l+1))/(2*m*r[i+1]**2))
        elif np.abs(i-j)==1:
            H[i,j]=-(hbar**2/(2*m*dr**2))
        else:
            H[i,j]=0

print("Hamiltonian Matrix Constructed.")

val_std,vec_std=sp.sparse.linalg.eigsh(H,k=eigs,which='SM')

val_std,vec_std=np.array(val_std),np.array(vec_std)
z_std = np.argsort(val_std)
#z_std = z_std[0:eigs_int]

val,vec=np.array(val),np.array(vec)
z = np.argsort(val)
#z = z[0:eigs_int]

"""
plt.figure(figsize=(12,10))
for i in range(eigs_int):
    y_std = []
    y_std = np.append(y_std,vec_std[:,z_std[i]])
    y_std = np.append(y_std,0)
    y_std = np.insert(y_std,0,0)
    rho=np.real(y_std*np.conj(y_std))
    rho=rho/np.trapz(rho)
    
    y = []
    y = np.append(y,vec_std[:,z[i]])
    y = np.append(y,0)
    y = np.insert(y,0,0)
    rho_c=np.real(y*np.conj(y)+(1/(4*m**2))*np.gradient(y,dr)*np.conj(np.gradient(y,dr)))
    rho_c=rho_c/np.trapz(rho_c)
    
    plt.plot(r,rho,'--k',lw=1,label='Schrödinger')
    plt.plot(r,rho_c,'-k',lw=1,label='Corrected')
    plt.xlabel('r (fm/(ℏc))', size=10)
    plt.ylabel('Probability Density',size=10)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(r,rho_c-rho,'-k',lw=1)
    plt.xlabel('r (fm/(ℏc))', size=10)
    plt.ylabel('Correction',size=10)
    plt.grid()
    plt.show()
"""
# -----------------------------------------

B=-0.1013367
E=B

minim=0
maxim=750

def find_nearest(array, value):
    array = np.asarray(np.real(array))
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

i=find_nearest(val[z],E)[0]
print(find_nearest(val[z],E))
i_std=find_nearest(val_std[z_std],E)[0]
print(find_nearest(val_std[z_std],E))

psi = []
psi = np.append(psi,vec[:,z[i]])
psi = np.append(psi,0)
psi = np.insert(psi,0,0)
psi = psi

psi_std = []
psi_std = np.append(psi_std,vec_std[:,z_std[i_std]])
psi_std = np.append(psi_std,0)
psi_std = np.insert(psi_std,0,0)
psi_std = psi_std

rho=np.real(psi*np.conj(psi)+(1/(4*m**2))*np.gradient(psi,dr)*np.conj(np.gradient(psi,dr)))
rho_std=np.real(psi_std*np.conj(psi_std))

rho=rho/np.trapz(rho,x=r)
rho_std=rho_std/np.trapz(rho_std,x=r)

plt.plot(r[minim:maxim],psi_std[minim:maxim]/np.trapz(psi_std,x=r),'k--',lw=2, label="Standard")
plt.plot(r[minim:maxim],psi[minim:maxim]/np.trapz(psi,x=r),'k-',lw=2, label="Modified")
plt.xlabel('x (fm/ℏc)')
plt.legend()
plt.ylabel("u=rR(r)")
plt.grid()
plt.show()

plt.plot(r[minim:maxim],(psi/np.trapz(psi_std,x=r)-psi_std/np.trapz(psi,x=r))[minim:maxim],'-k',lw=1)
plt.xlabel('r (fm/ℏc)', size=10)
plt.ylabel(r'Correction $u(r)$',size=10)
plt.grid()
plt.show()

plt.plot(r[minim:maxim],rho_std[minim:maxim],'k--',lw=2, label="Standard")
plt.plot(r[minim:maxim],rho[minim:maxim],'k-',lw=2, label="Modified")
plt.xlabel('x (fm/ℏc)')
plt.legend()
plt.ylabel(r"$\rho(r)$")
plt.grid()
plt.show()

plt.plot(r[minim:maxim],(rho-rho_std)[minim:maxim],'-k',lw=1)
plt.xlabel('r (fm/ℏc)', size=10)
plt.ylabel(r'Correction $\rho(r)$',size=10)
plt.grid()
plt.show()
