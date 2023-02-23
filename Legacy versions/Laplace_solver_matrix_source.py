#%% defining the problem
import numpy as np
from scipy.sparse import diags, linalg
# import matplotlib.pylab as plt
import plotly.express as px
import microstructure as ms
import pyvista as pv
import time
from scipy.sparse import lil_matrix

# create the entire domain
N = [50,
     50,
     50] # number of points in each direction
dx = 1 # grid spacing (m)
domain = ms.create_phase_data(
    voxels = N,
    vol_frac = [0.35, 0.35, 0.3],
    sigma =4,
    seed = [30,40],
    display = False)

# conductivity
K = 1.0 

# maximum iterations
max_iter = 50

# initial condition
init_cond = 5

# boundary conditions
# when two perpendicular pair of boundaries are set to periodic (for instance: south/north and east/west)
# the edges of these boundaries will not be solved (aP will be zero at those nodes). 
# this problem can be resolved with some extra work, but for now, try not to use two perpendicular 
# periodic boundaries. in the case of SOFC microstructure, two perpendicular periodic boundaries are
# not realistic in the first place.
bc = {
    'West':     ['Dirichlet', 1],    # i = 0
    'East':     ['Dirichlet', 3],    # i = N[0]-1
    'South':    ['Dirichlet', 1],    # j = 0
    'North':    ['Dirichlet', 3],    # j = N[1]-1
    'Bottom':   ['Periodic', 1],    # k = 0
    'Top':      ['Periodic', 3],    # k = N[2]-1
}

# source function
from sympy.abc import x
from sympy import exp, log
source_func = exp(-x**2)


#%% solver pre-processing

# extract the domain that should be solved. ds is short for Domain for Solver.
# domain_P, _, _ = ms.percolation_analysis(domain)
domain_P = domain
ds = np.zeros(domain_P.shape, dtype=bool)
ds[domain_P==1] = True
if bc['West'][0] == 'Periodic' and np.sum(ds[0,:,:] == ds[-1,:,:]) != N[1]*N[2]:
    raise Exception('The percolating domain is not periodic in the West-East direction.')
if bc['South'][0] == 'Periodic' and np.sum(ds[:,0,:] == ds[:,-1,:]) != N[0]*N[2]:
    raise Exception('The percolating domain is not periodic in the South-North direction.')
if bc['Bottom'][0] == 'Periodic' and np.sum(ds[:,:,0] == ds[:,:,-1]) != N[0]*N[1]:
    raise Exception('The percolating domain is not periodic in the Bottom-Top direction.')

# measure the triple phase boundary and create a mask for source term
TPB_mask, TPB_density, vertices, lines = ms.measure_TPB(domain_P)
TPB_mask[np.logical_and(ds==False, TPB_mask)] = False

# source term treatment
from sympy import diff
from sympy import lambdify
f = lambdify(x, source_func)
fp = lambdify(x, diff(source_func, x))

# removing thin boundaries
# don't remove the thin boundaries for periodic boundary conditions. it will cause problems.
# ds = ms.remove_thin_boundaries(ds)

# allocating the neighboring arrays (aP, aW, aE, aS, aN, aB, aT)
L = N[0]*N[1]*N[2]  # total number of points in diagonal of A (matrix of coefficients)

# initial guess
phi = np.ones(shape = L, dtype = float) * init_cond

# initializing left hand side and right hand side of the system of equaitons
# aP = np.zeros(shape = L, dtype = float)
# aT = np.zeros(shape = L, dtype = float)
# aB = np.zeros(shape = L, dtype = float)
# aW = np.zeros(shape = L, dtype = float)
# aE = np.zeros(shape = L, dtype = float)
# aS = np.zeros(shape = L, dtype = float)
# aN = np.zeros(shape = L, dtype = float)
rhs = np.zeros(shape = L, dtype = float) # right hand side vector
sum_nb = np.zeros(shape = L, dtype = float) # vector to collect the sum of the coefficients of the neighbors

# creating the matrix of coefficients (A) and right hand side vector (rhs)
# check if the point is in the solver domain

ip1, jp1, kp1 = np.where(ds)    # indices of points in the solver domain
ip1_F, jp1_F, kp1_F = np.where(np.logical_not(ds)) # indices of points outside the solver domain

# allocating A matrix
A = lil_matrix((L,L), dtype = float) # sparse matrix

# creating the matrix of coefficients (A) and right hand side vector (rhs) for the points that lie outside the solver domain
for n in range(len(ip1_F)):
    i,j,k = ip1_F[n], jp1_F[n], kp1_F[n]
    index = i*(N[1]*N[2]) + j*N[2] + k
    A[index,index] = -1
    # rhs[index] = 0  # no need to set the rhs to zero, it is already zero
    # a_nb[index] = 0 # no need to set the rhs to zero, it is already zero

# west side (i=0)
i = 0
if bc['West'][0] == 'Dirichlet':
    for n in np.array(np.where(ip1==i)).T:
        j,k = jp1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aW[index], aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0,0
        rhs[index] = -bc['West'][1]
elif bc['West'][0] == 'Neumann':
    for n in np.array(np.where(ip1==i)).T:
        j,k = jp1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        A[index, index+N[1]*N[2]] = 1   # aE[index]
        # aW[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
        rhs[index] = -bc['West'][1]*dx
elif bc['West'][0] == 'Periodic':
    for n in np.array(np.where(ip1==i)).T:
        j,k = jp1[n], kp1[n]
        if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
        index = i*(N[1]*N[2]) + j*N[2] + k
        aW, aE, aS, aN, aB, aT = \
        K*ds[-2,j,k], K*ds[i+1,j,k], K*ds[i,j-1,k], K*ds[i,j+1,k], K*ds[i,j,k-1], K*ds[i,j,k+1]

        A[index, (N[0]-2)*(N[1]*N[2]) + j*N[2] + k] = aW
        A[index, index+N[2]*N[1]] = aE
        A[index, index-N[2]] = aS
        A[index, index+N[2]] = aN
        A[index, index-1] = aB
        A[index, index+1] = aT
        A[index, index] = -(aW + aE + aS + aN + aB + aT)
        # rhs[index] = 0  # no need to set the rhs to zero, it is already zero

# east side (i=N[0]-1)
i = N[0]-1
if bc['East'][0] == 'Dirichlet':
    for n in np.array(np.where(ip1==i)).T:
        j,k = jp1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aW[index], aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0,0
        rhs[index] = -bc['East'][1]
elif bc['East'][0] == 'Neumann':
    for n in np.array(np.where(ip1==i)).T:
        j,k = jp1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aW[index] = 1
        A[index, index-N[1]*N[2]] = 1   # aW[index]
        # aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
        rhs[index] = -bc['East'][1]*dx


# south side (j=0)
j = 0
if bc['South'][0] == 'Dirichlet':
    for n in np.array(np.where(jp1==j)).T:
        i,k = ip1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        # aS[index] = 0
        A[index, index] = -1
        # aW[index], aE[index], aN[index], aT[index], aB[index] = 0,0,0,0,0
        rhs[index] = -bc['South'][1]
elif bc['South'][0] == 'Neumann':
    for n in np.array(np.where(jp1==j)).T:
        i,k = ip1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aN[index] = 1
        A[index, index+N[2]] = 1   # aN[index]
        # aW[index], aE[index], aT[index], aB[index] = 0,0,0,0
        rhs[index] = -bc['South'][1]*dx
elif bc['South'][0] == 'Periodic':
    for n in np.array(np.where(jp1==j)).T:
        i,k = ip1[n], kp1[n]
        if i==0 or i==N[0]-1 or k==0 or k==N[2]-1: continue
        index = i*(N[1]*N[2]) + j*N[2] + k
        aW, aE, aS, aN, aB, aT = \
        K*ds[i-1,j,k], K*ds[i+1,j,k], K*ds[i,-2,k], K*ds[i,j+1,k], K*ds[i,j,k-1], K*ds[i,j,k+1]
        
        A[index, index-N[2]*N[1]] = aW
        A[index, index+N[2]*N[1]] = aE
        A[index, i*(N[1]*N[2]) + (N[1]-2)*N[2] + k] = aS
        A[index, index+N[2]] = aN
        A[index, index-1] = aB
        A[index, index+1] = aT
        A[index, index] = -(aW + aE + aS + aN + aB + aT)
        # rhs[index] = 0  # no need to set the rhs to zero, it is already zero


# north side (j=N[1]-1)
j = N[1]-1
if bc['North'][0] == 'Dirichlet':
    for n in np.array(np.where(jp1==j)).T:
        i,k = ip1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        # aN[index] = 0
        A[index, index] = -1
        # aW[index], aE[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
        rhs[index] = -bc['North'][1]
elif bc['North'][0] == 'Neumann':   
    for n in np.array(np.where(jp1==j)).T:
        i,k = ip1[n], kp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aS[index] = 1
        A[index, index-N[2]] = 1   # aS[index]
        # aW[index], aE[index], aT[index], aB[index] = 0,0,0,0
        rhs[index] = -bc['North'][1]*dx

# bottom side (k=0)
k = 0
if bc['Bottom'][0] == 'Dirichlet':
    for n in np.array(np.where(kp1==k)).T:
        i,j = ip1[n], jp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        # aB[index] = 0
        A[index, index] = -1
        # aW[index], aE[index], aS[index], aN[index], aT[index] = 0,0,0,0,0
        rhs[index] = -bc['Bottom'][1]
elif bc['Bottom'][0] == 'Neumann':
    for n in np.array(np.where(kp1==k)).T:
        i,j = ip1[n], jp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aT[index] = 1
        A[index, index+1] = 1   # aT[index]
        # aW[index], aE[index], aS[index], aN[index] = 0,0,0,0
        rhs[index] = -bc['Bottom'][1]*dx
elif bc['Bottom'][0] == 'Periodic':
    for n in np.array(np.where(kp1==k)).T:
        i,j = ip1[n], jp1[n]
        if i==0 or i==N[0]-1 or j==0 or j==N[1]-1: continue
        index = i*(N[1]*N[2]) + j*N[2] + k
        aW, aE, aS, aN, aB, aT = \
        K*ds[i-1,j,k], K*ds[i+1,j,k], K*ds[i,j-1,k], K*ds[i,j+1,k], K*ds[i,j,-2], K*ds[i,j,k+1]

        A[index, index-N[2]*N[1]] = aW
        A[index, index+N[2]*N[1]] = aE
        A[index, index-N[2]] = aS
        A[index, index+N[2]] = aN
        A[index, i*(N[1]*N[2]) + j*N[2] + (N[2]-2)] = aB
        A[index, index+1] = aT
        A[index, index] = -(aW + aE + aS + aN + aB + aT)
        # rhs[index] = 0  # no need to set the rhs to zero, it is already zero

# top side (k=N[2]-1)
k = N[2]-1
if bc['Top'][0] == 'Dirichlet':
    for n in np.array(np.where(kp1==k)).T:
        i,j = ip1[n], jp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        # aT[index] = 0
        A[index, index] = -1
        # aW[index], aE[index], aS[index], aN[index], aB[index] = 0,0,0,0,0
        rhs[index] = -bc['Top'][1]
elif bc['Top'][0] == 'Neumann':
    for n in np.array(np.where(kp1==k)).T:
        i,j = ip1[n], jp1[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        A[index, index] = -1
        # aB[index] = 1
        A[index, index-1] = 1   # aB[index]
        # aW[index], aE[index], aS[index], aN[index] = 0,0,0,0
        rhs[index] = -bc['Top'][1]*dx

# interior points inside the solver domain (only neighbors coefficients)
b_index = np.array(np.where(np.logical_or.reduce((
    ip1==0, 
    ip1==N[0]-1, 
    jp1==0, 
    jp1==N[1]-1, 
    kp1==0, 
    kp1==N[2]-1
    ))))

ip1_T_int = np.delete(ip1, b_index)
jp1_T_int = np.delete(jp1, b_index)
kp1_T_int = np.delete(kp1, b_index)

for n in range(len(ip1_T_int)):
    i,j,k = ip1_T_int[n], jp1_T_int[n], kp1_T_int[n]
    index = i*(N[1]*N[2]) + j*N[2] + k
    A[index, index - N[1]*N[2]] = K*ds[i-1,j,k]
    A[index, index + N[1]*N[2]] = K*ds[i+1,j,k] 
    A[index, index - N[2]] = K*ds[i,j-1,k] 
    A[index, index + N[2]] = K*ds[i,j+1,k] 
    A[index, index - 1] = K*ds[i,j,k-1] 
    A[index, index + 1] = K*ds[i,j,k+1]

# create index arrays where source_mask==True and source_mask==False
L_int_s = np.sum(np.logical_and(TPB_mask[1:-1,1:-1,1:-1], ds[1:-1,1:-1,1:-1]))
L_int_ns = np.sum(np.logical_and(np.logical_not(TPB_mask[1:-1,1:-1,1:-1]), ds[1:-1,1:-1,1:-1]))
    
    # interior points (source==True)
ip1_int_s = np.zeros(shape = L_int_s, dtype = int) 
jp1_int_s = np.zeros(shape = L_int_s, dtype = int)
kp1_int_s = np.zeros(shape = L_int_s, dtype = int)
    # interior points (source==False)
ip1_int_ns = np.zeros(shape = L_int_ns, dtype = int)  
jp1_int_ns = np.zeros(shape = L_int_ns, dtype = int)
kp1_int_ns = np.zeros(shape = L_int_ns, dtype = int)

cntr_s = 0
cntr_ns = 0
for n in range(len(ip1_T_int)):
    i,j,k = ip1_T_int[n], jp1_T_int[n], kp1_T_int[n]
    if TPB_mask[i,j,k]:
        ip1_int_s[cntr_s], jp1_int_s[cntr_s], kp1_int_s[cntr_s] = i,j,k
        cntr_s += 1
    else:
        ip1_int_ns[cntr_ns], jp1_int_ns[cntr_ns], kp1_int_ns[cntr_ns] = i,j,k
        cntr_ns += 1

# interior points (source==False)
for n in range(len(ip1_int_ns)):
    i,j,k = ip1_int_ns[n], jp1_int_ns[n], kp1_int_ns[n]
    index = i*(N[1]*N[2]) + j*N[2] + k
    aW = A[index, index - N[1]*N[2]]# = K*ds[i-1,j,k]
    aE = A[index, index + N[1]*N[2]]# = K*ds[i+1,j,k] 
    aS = A[index, index - N[2]]# = K*ds[i,j-1,k] 
    aN = A[index, index + N[2]]# = K*ds[i,j+1,k] 
    aB = A[index, index - 1]# = K*ds[i,j,k-1] 
    aT = A[index, index + 1]# = K*ds[i,j,k+1]
    A[index, index] = -(aW + aE + aS + aN + aB + aT)
    # rhs[index] = 0    # no need to change rhs here since it is initialized to zero

# interior points (source==True) - in each iteration solver should be called here
t = time.time()
for iter in range(1,max_iter+1):

    #periodic boundary conditions
    if bc['East'][0] == 'Periodic':
        i_E, i_W = N[0]-1, 0
        for n in np.array(np.where(ip1==i_E)).T:
            j,k = jp1[n], kp1[n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
            index_E = i_E*(N[1]*N[2]) + j*N[2] + k
            index_W = i_W*(N[1]*N[2]) + j*N[2] + k
            A[index_E, index_E] = -1
            # aW[index_E], aE[index_E], aS[index_E], aN[index_E], aB[index_E], aT[index_E] = 0,0,0,0,0,0
            rhs[index_E] = -phi[index_W]
    if bc['North'][0] == 'Periodic':
        j_N, j_S = N[1]-1, 0
        for n in np.array(np.where(jp1==j_N)).T:
            i,k = ip1[n], kp1[n]
            index_N = i*(N[1]*N[2]) + j_N*N[2] + k
            index_S = i*(N[1]*N[2]) + j_S*N[2] + k
            A[index_N, index_N] = -1
            # aW[index_N], aE[index_N], aS[index_N], aN[index_N], aB[index_N], aT[index_N] = 0,0,0,0,0,0
            rhs[index_N] = -phi[index_S]
    if bc['Top'][0] == 'Periodic':
        k_T, k_B = N[2]-1, 0
        for n in np.array(np.where(kp1==k_T)).T:
            i,j = ip1[n], jp1[n]
            index_T = i*(N[1]*N[2]) + j*N[2] + k_T
            index_B = i*(N[1]*N[2]) + j*N[2] + k_B
            A[index_T, index_T] = -1
            # aW[index_T], aE[index_T], aS[index_T], aN[index_T], aB[index_T], aT[index_T] = 0,0,0,0,0,0
            rhs[index_T] = -phi[index_B]
    
    # interior points (source==True)
    for n in range(len(ip1_int_s)):
        i,j,k = ip1_int_s[n], jp1_int_s[n], kp1_int_s[n]
        index = i*(N[1]*N[2]) + j*N[2] + k
        if iter==1:
            aW = A[index, index - N[1]*N[2]]
            aE = A[index, index + N[1]*N[2]] 
            aS = A[index, index - N[2]] 
            aN = A[index, index + N[2]] 
            aB = A[index, index - 1] 
            aT = A[index, index + 1]
            sum_nb[index] = aW + aE + aS + aN + aB + aT
        A[index, index] = -(sum_nb[index] - fp(phi[index])*dx**3)

        rhs[index] = -((f(phi[index]) - fp(phi[index])*phi[index]) * dx**3)

    # trim the neighboring arrays to the correct size
    # np.sum([aW[i] for i in np.arange(0,N[1]*N[2])])
    # aT_R = np.delete(aT, L-1)
    # aB_R = np.delete(aB, 0)
    # aN_R = np.delete(aN, np.arange(L-N[2],L))
    # aS_R = np.delete(aS, np.arange(0,N[2]))
    # aE_R = np.delete(aE, np.arange(L-N[1]*N[2],L))
    # aW_R = np.delete(aW, np.arange(0,N[1]*N[2]))

    # create the sparse matrix of coefficients
    # A.setdiag(aP, k=0)
    # A.setdiag(aT_R, k=1)
    # A.setdiag(aB_R, k=-1)
    # A.setdiag(aN_R, k=N[2])
    # A.setdiag(aS_R, k=-N[2])
    # A.setdiag(aE_R, k=N[1]*N[2])
    # A.setdiag(aW_R, k=-N[1]*N[2])

    # A = diags([aP,aT_R,aB_R,aN_R,aS_R,aE_R,aW_R], [0,1,-1,N[2],-N[2],N[1]*N[2],-N[1]*N[2]], shape=(L,L))
    # A_show = A.todense()
    # A_show[A_show!=0] = 1
    # A_show[A_show==0] = np.nan
    # fig = px.imshow(A_show)
    # fig.show()
    phi, _ = linalg.gmres(A, rhs, x0=phi, maxiter=1)

    # phi_r = np.copy(phi.reshape(N))
    # phi_r[ds==False] = np.nan

    # ms.visualize_mesh(
    #     mat = [phi_r], 
    #     thd = [()])


    # if np.min(phi_r[-1,:,:][ds[-1,:,:]] == phi_r[0 ,:,:][ds[0 ,:,:]])==False:
    #     print('Periodic BCs not satisfied east-west')
    # if np.min(phi_r[:,-1,:][ds[:,-1,:]] == phi_r[:,0 ,:][ds[:,0 ,:]])==False:
    #     print('Periodic BCs not satisfied north-south')
    # if np.min(phi_r[:,:,-1][ds[:,:,-1]] == phi_r[:,:,0 ][ds[:,:,0 ]])==False:
    #     raise Exception('Periodic BCs not satisfied top-bottom')

    print(f"iteration: {iter}")

print(f'elapsed time: {time.time()-t}')

#%% visualize the solution
phi = phi.reshape(N)
if np.sum(phi[0,:,:] == phi[-1,:,:]) == np.sum(ds[0,:,:]):
    print('Periodic boundary condition is satisfied.')
    
pv.set_plot_theme("document")

# removing the phi points outside the domain for visualization purposes
phi[ds==False] = np.nan

TPB_mesh = pv.PolyData(vertices, lines=lines)
ms.visualize_mesh(
    mat = [phi], 
    thd = [()], 
    clip_widget = False, 
    TPB_mesh = TPB_mesh)
