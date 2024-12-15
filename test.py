# from main import solve_anode as sis
# if __name__ == '__main__':
#     sis(999)

from main import solve_entire_domain as ed
if __name__=='__main__':
    ed(1)

# from modules.topology import create_microstructure_plurigaussian as cmp
# from modules.topology import create_ideal_microstructre_straight_bars as cimsb
# from modules.topology import create_vertices_in_uniform_grid as cvug
# from modules.topology import measure_TPB as mtpb
# import pyvista as pv
# from modules.postprocess import visualize_mesh as vm

# # create a microstructure
# a = cmp(
#     [3025,100,100],
#     [0.4,0.3,0.3], 
#     500e-9, 
#     50e-9,
#     save_matrix=True)

# TPB_mask, TPB_density, lines, _ = mtpb(a,50e-9)
# TPB_mesh_a = pv.PolyData(
#     cvug(a.shape), 
#     lines=lines,
#     )

# b = cimsb([150,40,40])
# TPB_mask, TPB_density, lines, _ = mtpb(b,50e-9)
# TPB_mesh_b = pv.PolyData(
#     cvug(b.shape), 
#     lines=lines,
#     )

# vm(
#     [a,b],
#     [[2,3],[2,3]],
#     TPB_mesh=[TPB_mesh_a, TPB_mesh_b], 
#     # elevation=30, 
#     # azimuth=150,
#     link_views=True,
#     )

#%%
from modules import postprocess as post
import scipy.io
import numpy as np
vm = post.visualize_mesh
from modules.topology import create_vertices_in_uniform_grid as cvug
from modules.topology import measure_TPB as mtpb
import pyvista as pv


mat = scipy.io.loadmat('hadi microstructures\\17.mat')

values = mat.values()
type(values)

# get the values from dict_values
values = list(values)
mat = values[-1]

mat = mat[0:256//4, 0:256//4, 0:256//4]

# measure TPB
TPB_mask, TPB_density, lines, _ = mtpb(mat,50e-9)
TPB_mesh_a = pv.PolyData(
    cvug(mat.shape), 
    lines=lines,
    )


#%%
vm(
    [mat],
    # [[2,3]],
    TPB_mesh=[TPB_mesh_a],
    cmap = 'plasma',
    onlyTPB=True,
    isometric=True,
    save_graphics=True,
    )