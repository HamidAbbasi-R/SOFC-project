#%%
import microstructure as ms
import numpy as np
import time

# create the entire domain
N = [50,50,50]
domain = ms.create_phase_data(
    voxels = N,
    vol_frac = [0.33, 0.33, 0.34],
    sigma = 4,
    seed = [70,30],
    display = False)

# domain, _, _ = ms.percolation_analysis(domain)

# construct the domain that should be solved
solver_domain = np.zeros(domain.shape, dtype=bool)
solver_domain[domain==1] = True

if np.sum(solver_domain[:,:,0] == solver_domain[:,:,-1]) != N[0]*N[1] or \
    np.sum(solver_domain[:,0,:] == solver_domain[:,-1,:]) != N[0]*N[2] or \
    np.sum(solver_domain[0,:,:] == solver_domain[-1,:,:]) != N[1]*N[2]:
    raise Exception('The percolating domain is not periodic in at least one direction.\nNot suitable for periodic boundary condition.')


# measure the triple phase boundary 
TPB_mask, TPB_density, vertices, lines = ms.measure_TPB_vec(domain)
TPB_mask[np.logical_and(solver_domain==False, TPB_mask==True)] = False


# source function
from sympy.abc import x
from sympy import exp, log
source_func = exp(-x**2)

# defining the boundary conditions
bc = {
    'West and East':     ['Periodic'],    # i = 0
    # 'East':     ['Periodic',3],    # i = N[0]-1
    'South':    ['Dirichlet',1],    # j = 0
    'North':    ['Dirichlet',3],    # j = N[1]-1
    'Bottom':   ['Dirichlet', 1],    # k = 0
    'Top':      ['Dirichlet', 3],    # k = N[2]-1
}

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

# calling the solver and solve the problem
from Laplace_solver import Laplace_3D as m3d
iters=1000
with Timer('Solve the problem'):
    solution, err = m3d(
        domain = solver_domain,
        conductivity = 1,
        source_mask = TPB_mask,
        source_func = source_func, 
        bc_dict = bc,
        ic = 5, 
        delx = 0.001, 
        iters = iters,
        orf = 1)

if np.sum(solution[0,:,:] == solution[-1,:,:]) == np.sum(solver_domain[0,:,:]):
    print('Periodic boundary condition is satisfied.')
# visualize the TPB
# import pyvista as pv
# pv.set_plot_theme("document")
# TPB_mesh = pv.PolyData(vertices, lines=lines)
# ms.visualize_mesh(
#     mat = [solution], 
#     thd = [()], 
#     clip_widget = False, 
#     TPB_mesh = TPB_mesh)

# visualize the contour
# ms.visualize_contour(solution,10)

# visualize the error
# import plotly.express as px
# fig = px.line(x=np.arange(10,iters+10,10), y=err, log_y=True)
# fig.update_layout(
#     xaxis = dict(
#         title = 'Iterations'),
#     yaxis = dict(
#         title = 'Normalized residual',
#         showexponent = 'all',
#         exponentformat = 'e'
#     )
# )
# fig.show()
