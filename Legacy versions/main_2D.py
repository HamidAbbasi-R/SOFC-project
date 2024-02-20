#%% test a 2D problem
import numpy as np
import plotly.express as px
from modules.topology import create_microstructure_plurigaussian as cmp
from modules.topology import remove_thin_boundaries as rtb

# ms = imd('microstructure')
# create the entire domain
domain = cmp(
    voxels = [200,100], 
    vol_frac = [0.2,0.8],
    d_ave = 5,
    dx = 1,  
    seed = [70,30], 
    display = False)

# convert to float
domain = domain.astype(float)

# remvoe thin boundaries
domain = rtb(domain)

# extract the domain that should be solved
domain -= 1
solver_domain = domain.astype(bool)
solver_domain[domain==True] = True

# construct the 2D source matrix
# since in 2D models, TPB is not a boundary (instead it's a cluster of point), 
# I defined a sample source matrix on the boundaries of the domain. 
# It's just for debugging purposes.
from scipy import ndimage as ndi
A = 1-np.copy(domain)
B = ndi.binary_dilation(A)
C = B - A
source_mask = C.astype(bool)

# defining the boundary conditions
bc_dict = {
    'North':      ['Neumann', 0],
    'South':   ['Neumann', 0],
    # 'Top and Bottom': ['Periodic'],
    'West':     ['Neumann', 0],
    'East':    ['Dirichlet', 0]
    }

# source term
from sympy.abc import x
from sympy import exp
# source_func = exp(-x**2)
source_func = 1

# calling the solver and solve the problem
from Laplace_solver import Laplace_2D as m2d
iters=1000
solution, err = m2d(
    domain = solver_domain,
    source_mask = source_mask, 
    source_func = source_func,
    bc_dict = bc_dict, 
    ic = 0.5, 
    delx = 0.1, 
    iters = iters, 
    orf = 0.9)

# visualize the solution
fig = px.imshow(np.rot90(solution))
fig.show()

# visualize the contour
# import plotly.graph_objects as go
# fig = go.Figure(data = go.Contour(z=solution.transpose()))
# fig.update_yaxes(
#     scaleanchor = "x",
#     scaleratio = 1,
#   )
# fig.show()

# visualize the error
fig = px.line(x=np.arange(10,iters+10,10), y=err, log_y=True)
fig.update_layout(
    xaxis = dict(
        title = 'Iterations'),
    yaxis = dict(
        title = 'Normalized residual',
        showexponent = 'all',
        exponentformat = 'e'
    )
)
fig.show()