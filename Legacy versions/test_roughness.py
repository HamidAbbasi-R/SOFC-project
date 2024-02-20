# this file is a script to test the surface area of differet roughness and different resolutions

from modules import topology as tpl
import numpy as np

AR_rough = 0.04     # aspect ratio of roughness (AR = d_rough / D_circle)
D_circle = 1000
roughness_iteration = 4
dx = 1

domain_smooth = tpl.create_circle(D_circle//dx,'quarter')
domain_rough = tpl.add_roughness_all_phases(
    domain_smooth, 
    iteration=roughness_iteration, 
    d_rough=int(D_circle*AR_rough))

# surf_rough = tpl.measure_interface(domain_rough)

# SCALE = [1,2,4,8,16,32,64,128]
SCALE = [1,8,64]
length = [None] * len(SCALE)
domain_down = [None] * len(SCALE)

for i,scale in enumerate(SCALE):
    domain_down[i] = tpl.downscale_domain(domain_rough, scale)
    surf_down = tpl.measure_interface(domain_down[i])
    length[i] = np.sum(surf_down)*dx*scale

# write length to file
with open('length.txt','w') as f:
    for i in range(len(SCALE)):
        f.write(str(SCALE[i]) + ' ' + str(length[i]) + '\n')
        
# domain_rough = tpl.add_roughness_all_phases(domain_smooth, factor=1, d_rough=int(D_circle*AR_rough))
# domain_down = tpl.downscale_domain(domain_smooth, 16)


# surf_rough = tpl.measure_interface(domain_rough)

# print([np.sum(surf_smooth)*dx])

from modules.postprocess import plot_domain as pd
# plots = [domain_smooth]
# for i in range(len(DX)):
#     plots.append(down_domain[i])
# pd(domain_down, gap=1)


