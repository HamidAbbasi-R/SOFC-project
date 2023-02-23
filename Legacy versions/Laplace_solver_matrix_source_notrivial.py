#%% defining the problem
import numpy as np
from scipy.sparse import diags, linalg, lil_matrix
# import matplotlib.pylab as plt
import plotly.express as px
import microstructure as ms
import pyvista as pv
import time
import compute_matrices as cm


# create the entire domain
# number of grid points
N = [50,    # x direction
     50,    # y direction
     50]    # z direction

dx = 1 # grid spacing (m)
domain = ms.create_phase_data(
    voxels = N,
    vol_frac = [0.33, 0.33, 0.34],
    sigma = 4,
    seed = [70,30],
    display = False)

# conductivity
K = 1.0 

# maximum iterations
max_iter = 50

# initial condition
init_cond = 0

# boundary conditions
bc = {
    'West':     ['Dirichlet', 0],    # i = 0
    'East':     ['Dirichlet', 0],    # i = N[0]-1
    'South':    ['Dirichlet', 0],    # j = 0
    'North':    ['Dirichlet', 0],    # j = N[1]-1
    'Bottom':   ['Dirichlet', 0],    # k = 0
    'Top':      ['Dirichlet', 0],    # k = N[2]-1
}

# source function
from sympy.abc import x
from sympy import exp, log
source_func = -exp(-x**2)


#%% solver pre-processing

# domain, _, _ = ms.percolation_analysis(domain)
# if bc['West'][0] == 'Periodic' and np.sum(ds[0,:,:] == ds[-1,:,:]) != N[1]*N[2]:
#     raise Exception('The percolating domain is not periodic in the West-East direction.') 
# if bc['South'][0] == 'Periodic' and np.sum(ds[:,0,:] == ds[:,-1,:]) != N[0]*N[2]:
#     raise Exception('The percolating domain is not periodic in the South-North direction.')
# if bc['Bottom'][0] == 'Periodic' and np.sum(ds[:,:,0] == ds[:,:,-1]) != N[0]*N[1]:
#     raise Exception('The percolating domain is not periodic in the Bottom-Top direction.')

# measure the triple phase boundary and create a mask for source term
TPB_mask, TPB_density, vertices, lines = ms.measure_TPB(domain)


# source term treatment
from sympy import diff
from sympy import lambdify
f = lambdify(x, source_func)
fp = lambdify(x, diff(source_func, x))

# removing thin boundaries
# ds = ms.remove_thin_boundaries(ds)

# get indices of elements in the domain that should be solved
ds, indices = ms.get_indices(domain, 1, TPB_mask)
L = len(indices['all_points'])


# initial guess
phi = np.ones(shape = L, dtype = float) * init_cond

# create the matrix A and vector b for the linear system Ax = b

A, rhs = cm.boundaries(bc, indices, dx)     # boundaries
A = cm.interior(A, indices, K, ds)          # interior points [source==False]

# interior points (source==True) - in each iteration, linear solver should be called for just one iteration
t = time.time()
# for iter in range(1,max_iter+1):    
tol = 1e-6
res = 1
iter = 0
while res > tol and iter < max_iter:
    
    # update A[n,n] and rhs[b] for interior points (source==True)
    for n in indices['source']:
        aW = A[n,indices['west_nb'][n]]
        aE = A[n,indices['east_nb'][n]]
        aS = A[n,indices['south_nb'][n]]
        aN = A[n,indices['north_nb'][n]]
        aB = A[n,indices['bottom_nb'][n]]
        aT = A[n,indices['top_nb'][n]]
        sigma_anb = aW + aE + aS + aN + aB + aT

        A[n,n] = -(sigma_anb - fp(phi[n])*dx**3)    # aP
        rhs[n] = -((f(phi[n]) - fp(phi[n])*phi[n]) * dx**3) # RHS

    # A_show = A.todense()
    # A_show[A_show!=0] = 1
    # A_show[A_show==0] = np.nan
    # fig = px.imshow(A_show)
    # fig.show()

    phi_new, _ = linalg.gmres(A, rhs, x0=phi, maxiter=1)
    
    res = rhs - A.tocsr()@phi_new
    res = np.linalg.norm(res)
    phi = phi_new
    iter += 1
    print('iter = {}, error = {:.3e}'.format(iter, res))

print('elapsed time: ', time.time()-t)

#%% visualize the solution
sol = np.zeros(N, dtype=float)
sol[ds==False] = np.nan
sol[ds==True] = phi

# phi = phi.reshape(N)
# if np.sum(phi[0,:,:] == phi[-1,:,:]) == np.sum(ds[0,:,:]):
#     print('Periodic boundary condition is satisfied.')
    
pv.set_plot_theme("document")

# removing the phi points outside the domain for visualization purposes
# phi[ds==False] = np.nan

TPB_mesh = pv.PolyData(vertices, lines=lines)
ms.visualize_mesh(
    mat = [sol],
    thd = [()],
    clip_widget = False,
    TPB_mesh = TPB_mesh)
