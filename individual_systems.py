def solve_individual_systems():
    """
    This function does the following:
    1. Reads the input file
    2. Creates the microstructure
    3. Perform the topological operations
    4. Creates the field functions and assign the boundary conditions
    5. Creates the indices for the cells in each of three phases
    6. Creates the system of linear equations
    8. Solves the system of linear equations
    9. Creates the dense matrices
    10. Visualizes the results
    11. returns the current density
    """
    import json
    # from modules.file_options import file_options as fo
    from modules import topology as tpl
    from modules import file_options as fop
    from modules import preprocess as prep
    from modules import postprocess as post
    from modules import solve as slv

    f = open('inputs.json')
    inputs = json.load(f)
    file_options = inputs['file_options']

    # scale the ion potential if it is not zero to avoid numerical difficulties
    scale_ion = inputs['boundary_conditions']['Vio_b'] 
    inputs['boundary_conditions']['Vel_b'] = inputs['boundary_conditions']['Vel_b'] - inputs['boundary_conditions']['Vio_b']
    inputs['boundary_conditions']['Vio_b'] = 0

    if file_options['new_case']:
        domain = tpl.create_microstructure(inputs)
        domain, TPB_dict = tpl.topological_operations(inputs, domain)
        field_functions, _, bc_dict = prep.sourcefunc_calc(inputs, TPB_dict)
        masks_dict, indices =prep.get_indices_all(inputs, domain, TPB_dict)
        J, rhs, sum_nb = prep.create_SOLE_individual(inputs, bc_dict, indices, masks_dict)
        if file_options['save_case']:
            fop.save_case_individual(file_options['case_id'],
                inputs, indices, J, rhs, field_functions, masks_dict, sum_nb, TPB_dict, bc_dict)
        phi, residuals = prep.initilize_field_variables_individual(inputs, masks_dict, indices, inputs['is_multiple_instances'], inputs['M_instances'], inputs['scaling_factor'])

    elif file_options['load_case']:
        inputs_old, indices, J, rhs, field_functions, masks_dict, sum_nb, TPB_dict, bc_dict = fop.load_case_individual(file_options['case_id'])
        if inputs['microstructure'] != inputs_old['microstructure']: raise Exception('Microstructure does not match!')
        if (inputs['boundary_conditions'] != inputs_old['boundary_conditions'] or 
            inputs['operating_conditions'] != inputs_old['operating_conditions']):
            field_functions, _, bc_dict = prep.sourcefunc_calc(inputs, TPB_dict)
            J, rhs, sum_nb = prep.create_SOLE_individual(inputs, bc_dict, indices, masks_dict)
            if inputs['file_options']['save_case']:
                fop.save_case_individual(file_options['case_id'],
                    inputs, indices, J, rhs, field_functions, masks_dict, sum_nb, TPB_dict, bc_dict)
        phi, residuals = prep.initilize_field_variables_individual(inputs, masks_dict, indices, inputs['is_multiple_instances'], inputs['M_instances'], inputs['scaling_factor'])

    elif file_options['load_case_data']:
        inputs_old, indices, J, rhs, field_functions, masks_dict, sum_nb, TPB_dict, bc_dict, phi, residuals = fop.load_case_data_individual(file_options['case_id'], file_options['data_id'])
        if (inputs['microstructure'] != inputs_old['microstructure'] or 
            inputs['boundary_conditions'] != inputs_old['boundary_conditions'] or 
            inputs['operating_conditions'] != inputs_old['operating_conditions']): 
            raise Exception('Microstructure, boundary conditions, or operating conditions do not match!')


    phi, residuals = slv.Newton_loop_individual(inputs, J, rhs, phi, indices, field_functions, masks_dict, sum_nb, residuals)
    if file_options['save_data']:
        fop.save_data(file_options['data_id'], phi, residuals)
    

    if inputs['output_options']['show_residuals']:
        post.visualize_residuals(inputs, residuals)

    
    # scaling the electron and ion potentials to the original values
    phi[1] = phi[1] + scale_ion
    phi[2] = phi[2] + scale_ion
    dense_m = post.create_dense_matrices(phi, inputs, masks_dict, indices, field_functions)
    
    if inputs['output_options']['show_3D_plots'] or inputs['output_options']['show_1D_plots']:
        post.visualize_3D_matrix(
            inputs, dense_m, masks_dict, TPB_dict,
            plots = {
                'cH2_1D':       False,
                'Vel_1D':       False,
                'Vio_1D':       True,
                'Ia_1D':        True,
                'eta_act_1D':   True,
                'eta_con_1D':   True,
                'cH2_3D':       False,
                'Vel_3D':       False,
                'Vio_3D':       False,
                'Ia_3D':        False,
                'eta_act_3D':   False,
                'eta_con_3D':   False,
            })
    return dense_m['Ia']
            
