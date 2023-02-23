import json
from modules import topology as tpl
# from modules import file_options as fop
from modules import preprocess as prep
from modules import postprocess as post
from modules import solve as slv

f = open('inputs_entire_domain.json')
inputs = json.load(f)
file_options = inputs['file_options']
domain = tpl.create_microstructure_entire_cell(inputs)
domain, TPB_dict = tpl.topological_operations_entire_cell(inputs, domain)
field_functions, bc_dict = prep.sourcefunc_calc_entire_cell(inputs, TPB_dict)
masks_dict, indices = prep.get_indices_entire_cell(inputs, domain, TPB_dict)
J, rhs, sum_nb = prep.create_SOLE_individual_entire_cell(inputs, bc_dict, indices, masks_dict)
phi, residuals = prep.initilize_field_variables_individual_entire_cell(inputs, indices, inputs['is_multiple_instances'], inputs['M_instances'], inputs['scaling_factor'])
phi, residuals = slv.Newton_loop_individual_entire_cell(inputs, J, rhs, phi, indices, field_functions, masks_dict, sum_nb, residuals)

if inputs['show_residuals']:
    post.visualize_residuals(residuals)

if inputs['show_3D_plots'] or inputs['show_1D_plots']:
    import numpy as np
    N = [inputs['Nx_a']+inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    phi_dense = np.zeros(N)
    phi_dense[:inputs['Nx_a'],:,:][masks_dict['ds'][0]] = phi[0]
    phi_dense[:inputs['Nx_a'],:,:][masks_dict['ds'][1]] = phi[1]
    phi_dense[masks_dict['ds'][2]] = phi[2]
    phi_dense[inputs['Nx_a']+inputs['Nx_e']:,:,:][masks_dict['ds'][3]] = phi[3]
    phi_dense[inputs['Nx_a']+inputs['Nx_e']:,:,:][masks_dict['ds'][4]] = phi[4]

    post.visualize_3D_matrix_entire_cell(
        inputs, phi_dense, masks_dict, TPB_dict,
        titles = ['Vio'],
        Vio = True
        )