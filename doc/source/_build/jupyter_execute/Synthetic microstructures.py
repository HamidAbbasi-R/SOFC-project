#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../..")
from modules.topology import create_microstructure_plurigaussian
domain = create_microstructure_plurigaussian(
    [200,50,50],
    [0.3,0.3,0.4],
    300e-9,
    50e-9,
    display=True,
    notebook=True,
)

domain = create_microstructure_plurigaussian(
    [200,50,50],
    [0.33,0.33,0.34],
    300e-9,
    50e-9,
    gradient_factor = 1,
    display=True,
    notebook=True,
)


# In[2]:


import sys
sys.path.append("../..")
from modules.topology import create_fibre
from modules.topology import create_twisted_fibre
from modules.topology import create_twisted_multifibre
from modules.topology import create_fibrous_bed
from modules.topology import bend_fibre

from modules.postprocess import visualize_mesh as vm
from modules.topology import measure_TPB
from modules.topology import create_vertices_in_uniform_grid

fibre = create_fibre(
    radius=10,
    length=100,
)

twisted_fibre = create_twisted_fibre(
    radius=10,
    length=100,
    amp=1,
    freq=1,
)

twisted_multifibre = create_twisted_multifibre(
    radius=10,
    fibre_length=100,
    amp=1,
    freq=1,
    n_fibres=2,
)

import pyvista as pv
_, _, lines, _ = measure_TPB(twisted_multifibre+1,50e-9)
TPB_mesh_twist = pv.PolyData(create_vertices_in_uniform_grid(twisted_multifibre.shape), lines=lines)

bended_fibre = bend_fibre(
    twisted_multifibre,
    bending_factor = 0.5
)

_, _, lines, _ = measure_TPB(bended_fibre+1,50e-9)
TPB_mesh_bend = pv.PolyData(create_vertices_in_uniform_grid(bended_fibre.shape), lines=lines)


bed = create_fibrous_bed(
    voxels = [150,80,80],
    radius = 5,
    fibre_length = 80,
    target_porosity = 0.7,
    rotation_max = (90,0,0),
)

vm(
    [fibre],
    [(1,1)],
    notebook=True,
    )

vm(
    [twisted_fibre],
    [(1,1)],
    notebook=True,
    )

vm(
    [twisted_multifibre],
    [(1,2)],
    TPB_mesh=[TPB_mesh_twist],
    notebook=True,
    )

vm(
    [bended_fibre],
    [(1,2)],
    TPB_mesh=[TPB_mesh_bend],
    notebook=True,
)

vm(
    [bed],
    [(2,3)],
    notebook=True,
    )


# In[3]:


import sys
sys.path.append("../..")
from modules.topology import create_microstructure_lattice
from modules.postprocess import visualize_mesh as vm

lattice, _ = create_microstructure_lattice(
    vol_frac=[0.3,0.35,0.35],
    dx=50e-9,
    voxels=[100,50,50],
    d_particle =500e-9,
    offset = False,
    smallest_lattice = False,
)

vm(
    [lattice],
    [(2,3)],
    notebook=True,
    )

