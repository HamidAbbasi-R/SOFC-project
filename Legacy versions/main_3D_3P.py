#%%
import microstructure as ms
import numpy as np

# create the entire domain
domain = ms.create_phase_data(
    voxels = [50,50,50], 
    vol_frac = [0.33, 0.33, 0.34],
    sigma = 5, 
    seed = [70,30], 
    display = False)

# remove the non-percolating clusters
domain, _, _ = ms.percolation_analysis(domain)

# measure the triple phase boundary 
TPB_mask, TPB_density, vertices, lines = ms.measure_TPB_vec(domain)

# define the source functions
from sympy import symbols
from sympy import exp, log
phi0 = symbols('phi0')
phi1 = symbols('phi1')
phi2 = symbols('phi2')
source_func = [0]*3     # initialize the source function list
source_func[0] = exp(-(phi0**2+phi1**2))
source_func[1] = exp(-(phi1**2+phi2**2))
source_func[2] = exp(-(phi2**2+phi0**2))

# defining the boundary conditions of all three phases 
bc_dict = [{
    # the first phase [pores] ??
    'Top':      ['Dirichlet', [0]],
    'Bottom':   ['Dirichlet', [0]],
    'Left':     ['Dirichlet', [0]],
    'Right':    ['Dirichlet', [0]],
    'Front':    ['Dirichlet', [0]],
    'Back':     ['Dirichlet', [0]]
    },{
    # the second phase [Ni] ??
    'Top':      ['Dirichlet', [0]],
    'Bottom':   ['Dirichlet', [0]],
    'Left':     ['Dirichlet', [0]],
    'Right':    ['Dirichlet', [0]],
    'Front':    ['Dirichlet', [0]],
    'Back':     ['Dirichlet', [0]]
    },{
    # the third phase [YSZ] ??
    'Top':      ['Dirichlet', [0]],
    'Bottom':   ['Dirichlet', [0]],
    'Left':     ['Dirichlet', [0]],
    'Right':    ['Dirichlet', [0]],
    'Front':    ['Dirichlet', [0]],
    'Back':     ['Dirichlet', [0]]
    }]

# calling the solver and solve the problem
from Laplace_solver import Laplace_3D_3phase as m3d3p
iters = 100
solution, err = m3d3p(
    domain = domain, 
    conductivity = [1,1,1],
    source_mask = TPB_mask,
    source_func = source_func, 
    bc_dict = bc_dict, 
    ic = [0.0, 0.0, 0.0], 
    delx = 0.1, 
    iters = iters, 
    orf = 1)

# visualize the error
# import plotly.express as px
# fig = px.line(x=np.arange(10,iters+10,10), y=err, log_y=True)
# fig.show()


# visualize the TPB
import pyvista as pv
pv.set_plot_theme("document")
TPB_mesh = pv.PolyData(vertices, lines=lines)
ms.visualize_mesh(
    mat = [solution[:,:,:,0]], 
    thd = [()], 
    clip_widget = True, 
    TPB_mesh = TPB_mesh)
