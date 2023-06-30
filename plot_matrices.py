id = 1001
ch2_flag        = False
Vel_flag        = True
Vio_flag        = True
Ia_flag         = False
eta_act_flag    = False
eta_con_flag    = False


# Import libraries
import numpy as np

matrices = np.load(f'Binary files/arrays/matrices_{id}.npz')

phi = matrices['phi']
cH2 = matrices['cH2']
Vel = matrices['Vel']
Vio = matrices['Vio']
Ia = matrices['Ia']
eta_act = matrices['eta_act']
eta_con = matrices['eta_con']
vertices = matrices['vertices']
lines = matrices['lines']

# Plotting
mats = []
thds = []
log_scale = []
titles = []

import pyvista as pv
pv.set_plot_theme("document")
TPB_mesh = pv.PolyData(vertices, lines=lines)

if ch2_flag:
    mats.append(cH2)
    thds.append([])
    log_scale.append(False)
    titles.append('cH2')

if Vel_flag:
    mats.append(Vel)
    thds.append([])
    log_scale.append(False)
    titles.append('Vel')

if Vio_flag:
    mats.append(Vio)
    thds.append([])
    log_scale.append(False)
    titles.append('Vio')

if Ia_flag:
    mats.append(Ia)
    thds.append([])
    log_scale.append(True)
    titles.append('Ia')

if eta_act_flag:
    mats.append(eta_act)
    thds.append([])
    log_scale.append(False)
    titles.append('eta_act')

if eta_con_flag:
    mats.append(eta_con)
    thds.append([])
    log_scale.append(False)
    titles.append('eta_con')

from modules.postprocess import visualize_mesh
visualize_mesh(
    mat = mats,
    thd = thds,
    # titles = titles,
    # clip_widget = False, 
    # TPB_mesh = TPB_mesh,
    log_scale = log_scale,
    )