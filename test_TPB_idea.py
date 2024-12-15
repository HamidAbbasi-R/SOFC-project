#%%
import numpy as np
from modules.postprocess import visualize_mesh as vm
from modules.topology import create_vertices_in_uniform_grid as cvug
from modules.topology import measure_TPB as mTPB
import pyvista as pv

#%%
a = np.array([[1, 1], [2, 2]])
b = np.array([[1, 1], [1, 3]])
c = np.stack((a, b), axis=0)

mask, density, lines, dist = mTPB(c,1)
TPB_mesh = pv.PolyData(
    cvug(c.shape), 
    lines=lines,
    )

vm(
    [c],
    [[]],
    TPB_mesh=[TPB_mesh],
    entire_domain=[False],
    edge_width=5,
    isometric=True,
    # clip_widget=True,
    cmap='Accent',
    link_colorbar=True,
    # opacity=0.8,
    # save_graphics=True,
    # link_views=True,
    notebook = True,
    )
