#%%
from modules import microstructure as ms
N = 50
vol_frac = [0.4,0.3]
sigma = 5
phase_mat = ms.create_phase_data(N=N, vol_frac=vol_frac, dim=2, sigma=sigma)

#%% upscaling
# ups_factor = 4
# N_ups = int(N/ups_factor)
# ups_mat = np.zeros(shape=(N_ups,N_ups))
# for i in np.arange(0,N,ups_factor):
#     for j in np.arange(0,N,ups_factor):
#         i_ups = int(i/ups_factor)
#         j_ups = int(j/ups_factor)
#         ups_mat[i_ups,j_ups] = np.average(phase_mat[i:i+ups_factor,j:j+ups_factor])

#%% Specific surface [image dilation]
from scipy import ndimage as ndi

A1 = phase_mat == 1
A2 = phase_mat == 2
A3 = phase_mat == 3

B1 = ndi.binary_dilation(A1).astype(A1.dtype)
B2 = ndi.binary_dilation(A2).astype(A2.dtype)
B3 = ndi.binary_dilation(A3).astype(A3.dtype)

C1 = B1 - A1.astype(int)
C2 = B2 - A2.astype(int)
C3 = B3 - A3.astype(int)

C12 = C1 * A2.astype(int)
C23 = C2 * A3.astype(int)
C13 = C1 * A3.astype(int)


#%% Percolation analysis
import numpy as np
from scipy import ndimage as ndi
import copy

phase_1= copy.copy(phase_mat)
phase_2= copy.copy(phase_mat)
phase_3= copy.copy(phase_mat)

phase_1[phase_mat==3] = 0
phase_1[phase_mat==2] = 0

phase_2[phase_mat==1] = 0
phase_2[phase_mat==3] = 0

phase_3[phase_mat==1] = 0
phase_3[phase_mat==2] = 0

lw1, num1 = ndi.label(phase_1)
lw2, num2 = ndi.label(phase_2)
lw3, num3 = ndi.label(phase_3)

lw1 = lw1.astype(float)
lw1[lw1==0] = np.nan

lw2 = lw2.astype(float)
lw2[lw2==0] = np.nan

lw3 = lw3.astype(float)
lw3[lw3==0] = np.nan

#%% visualization
import plotly.express as px
import plotly.io as po
from plotly.subplots import make_subplots

# po.renderers.default = "browser"

fig = make_subplots(rows=1, cols=2, shared_yaxes='all', shared_xaxes='all',
                    subplot_titles=("First smooth field", "Second smooth field"))

fig.add_heatmap(z=phase_mat,row=1,col=1, showscale=False,)
fig.add_heatmap(z=smooth_mat_2,row=1,col=2, showscale=False)
# fig.add_heatmap(z=C23.astype(int),row=2,col=1, showscale=False)
# fig.add_heatmap(z=C13.astype(int),row=2,col=2, showscale=False)

fig['layout']['yaxis1']['scaleanchor']='x1'
fig['layout']['yaxis2']['scaleanchor']='x2'
# fig['layout']['yaxis3']['scaleanchor']='x3'
# fig['layout']['yaxis4']['scaleanchor']='x4'
fig.show()

# px.imshow(phase_mat)
# px.imshow(lw1)
# px.imshow(dist_mat)
# px.imshow(labels)
