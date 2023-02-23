#%% defining the problem
import numpy as np
from scipy.sparse import diags, linalg
import matplotlib.pylab as plt
import plotly.express as px
import microstructure as ms
import pyvista as pv

# create the entire domain
N = [80,70,60] # number of points in each direction
dx = 0.001 # grid spacing (m)
domain = ms.create_phase_data(
    voxels = N, 
    vol_frac = [0.33, 0.33, 0.34],
    sigma = 5, 
    seed = [70,30], 
    display = False)

# conductivity
K = 1.0 

# boundary conditions
bc = {
    'Top':      ['Neumann', 0],
    'Bottom':   ['Neumann', 0],
    'West':     ['Dirichlet', 2],
    'East':    ['Dirichlet', 1],
    'North':    ['Neumann', 0],
    'South':     ['Neumann', 0]
}

# source function
from sympy.abc import x
from sympy import exp, log
source_func = exp(-x**2)

#%% solver pre-processing
# extract the domain that should be solved. ds is short for Domain for Solver.
domain, exitcode, exitcode = ms.percolation_analysis(domain)
ds = np.zeros(domain.shape, dtype=bool)
ds[domain==1] = True

# measure the triple phase boundary and create a mask for source term
TPB_mask, TPB_density, vertices, lines = ms.measure_TPB_vec(domain)
TPB_mask[np.logical_and(ds==False, TPB_mask)] = False

# removing thin boundaries
ds = ms.remove_thin_boundaries(ds)

# source term treatment
from sympy import diff
from sympy import lambdify
f = lambdify(x, source_func)
fp = lambdify(x, diff(source_func, x))

# allocating the neighboring arrays (aP, aW, aE, aS, aN, aB, aT)
L = N[0]*N[1]*N[2]  # total number of points in diagonal of A (matrix of coefficients)

aP = np.ones(shape = L, dtype = float)
aT = np.ones(shape = L, dtype = float)
aB = np.ones(shape = L, dtype = float)
aN = np.ones(shape = L, dtype = float)
aS = np.ones(shape = L, dtype = float)
aE = np.ones(shape = L, dtype = float)
aW = np.ones(shape = L, dtype = float)
rhs = np.ones(shape = L, dtype = float) # right hand side vector

# creating the matrix of coefficients (A) and right hand side vector (rhs)
index = -1  # index of the point in the diagonal of A
for i in range(0,N[0]):
    for j in range(0,N[1]):
        for k in range(0,N[2]):
            index += 1
            
            # check if the point is in the solver domain
            if ds[i,j,k] == False:
                aP[index], aT[index], aB[index], aN[index], aS[index], aE[index], aW[index], rhs[index] =\
                    1, 0, 0, 0, 0, 0, 0, 0
                continue

            # west side
            if i==0:
                aW[index] = 0
                if bc['West'][0] == 'Dirichlet':
                    aP[index] = -1
                    aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
                    rhs[index] = -bc['West'][1]
                elif bc['West'][0] == 'Neumann':
                    aP[index] = -1
                    aE[index] = 1
                    aN[index], aS[index], aT[index], aB[index] = 0,0,0,0
                    rhs[index] = -bc['West'][1]*dx
                continue

            # east side
            if i==N[0]-1:
                aE[index] = 0
                if bc['East'][0] == 'Dirichlet':
                    aP[index] = -1
                    aW[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
                    rhs[index] = -bc['East'][1]
                elif bc['East'][0] == 'Neumann':
                    aP[index] = -1
                    aW[index] = 1
                    aN[index], aS[index], aT[index], aB[index] = 0,0,0,0
                    rhs[index] = -bc['East'][1]*dx
                continue

            # south side
            if j==0:
                aS[index] = 0
                if bc['South'][0] == 'Dirichlet':
                    aP[index] = -1
                    aW[index], aE[index], aN[index], aT[index], aB[index] = 0,0,0,0,0
                    rhs[index] = -bc['South'][1]
                elif bc['South'][0] == 'Neumann':
                    aP[index] = -1
                    aN[index] = 1
                    aW[index], aE[index], aT[index], aB[index] = 0,0,0,0
                    rhs[index] = -bc['South'][1]*dx
                continue

            # north side
            if j==N[1]-1:
                aN[index] = 0
                if bc['North'][0] == 'Dirichlet':
                    aP[index] = -1
                    aW[index], aE[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
                    rhs[index] = -bc['North'][1]
                elif bc['North'][0] == 'Neumann':
                    aP[index] = -1
                    aS[index] = 1
                    aW[index], aE[index], aT[index], aB[index] = 0,0,0,0
                    rhs[index] = -bc['North'][1]*dx
                continue

            # bottom side
            if k==0:
                aB[index] = 0
                if bc['Bottom'][0] == 'Dirichlet':
                    aP[index] = -1
                    aW[index], aE[index], aS[index], aT[index], aN[index] = 0,0,0,0,0
                    rhs[index] = -bc['Bottom'][1]
                elif bc['Bottom'][0] == 'Neumann':
                    aP[index] = -1
                    aT[index] = 1
                    aW[index], aE[index], aS[index], aN[index] = 0,0,0,0
                    rhs[index] = -bc['Bottom'][1]*dx
                continue

            # top side
            if k==N[2]-1:   
                aT[index] = 0
                if bc['Top'][0] == 'Dirichlet':
                    aP[index] = -1
                    aW[index], aE[index], aS[index], aB[index], aN[index] = 0,0,0,0,0
                    rhs[index] = -bc['Top'][1]
                elif bc['Top'][0] == 'Neumann':
                    aP[index] = -1
                    aB[index] = 1
                    aW[index], aE[index], aS[index], aN[index] = 0,0,0,0
                    rhs[index] = -bc['Top'][1]*dx
                continue

            # interior points
            aW[index], aE[index], aS[index], aN[index], aB[index], aT[index] = \
            K*ds[i-1,j,k], K*ds[i+1,j,k], K*ds[i,j-1,k], K*ds[i,j+1,k], K*ds[i,j,k-1], K*ds[i,j,k+1]
            
            aP[index] = -(
                + aW[index] + aE[index] 
                + aS[index] + aN[index] 
                + aB[index] + aT[index]
                # + fp(phi.reshape(N)[i,j,k])*TPB_mask[i,j,k]*dx**3
                )
                
            rhs[index] = 0 #+ (
                # + (f(phi.reshape(N)[i,j,k])
                # - fp(phi.reshape(N)[i,j,k])*phi.reshape(N)[i,j,k])
                # * TPB_mask[i,j,k]*dx**3
                #)

# trim the neighboring arrays to the correct size
aT = np.delete(aT, L-1)
aB = np.delete(aB, 0)
aN = np.delete(aN, np.arange(L-N[2],L))
aS = np.delete(aS, np.arange(0,N[2]))
aE = np.delete(aE, np.arange(L-N[1]*N[2],L))
aW = np.delete(aW, np.arange(0,N[1]*N[2]))

# create the sparse matrix of coefficients
A = diags([aP,aT,aB,aN,aS,aE,aW], [0,1,-1,N[2],-N[2],N[1]*N[2],-N[1]*N[2]], shape=(L,L))

#%% Solving the system of equations
# class gmres_counter(object):
#     def __init__(self, disp=True):
#         self._disp = disp
#         self.niter = 0
#     def __call__(self, rk=None):
#         self.niter += 1
#         if self._disp:
#             print('iter %3i\trk = %s' % (self.niter, str(rk)))

# counter = gmres_counter()
x0 = 1.5 # initial guess
max_iter = 100
tol = 1e-6
import time

t = time.time()
phi, exitCode = linalg.gmres(A, rhs, tol=tol, x0=np.ones(L)*1.5, maxiter=max_iter)
print('Single GMRES took %f seconds' % (time.time()-t))

t = time.time()
for i in range(max_iter):
    if i==0:
        phi2, exitcode = linalg.gmres(A, rhs, tol=tol, x0=np.ones(L)*1.5, maxiter=1)
    else:
        phi2, exitcode = linalg.gmres(A, rhs, tol=tol, x0=phi2, maxiter=1)
print('Consequetive GMRES took %f seconds' % (time.time()-t))
#%% visualize the solution
pv.set_plot_theme("document")

# removing the phi points outside the domain for visualization purposes
phi = phi.reshape(N)
phi[ds==False] = np.nan

TPB_mesh = pv.PolyData(vertices, lines=lines)
ms.visualize_mesh(
    mat = [phi], 
    thd = [()], 
    clip_widget = False, 
    TPB_mesh = TPB_mesh)
