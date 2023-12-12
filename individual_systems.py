# if __name__ == '__main__':
def solve_individual_systems(id):
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
    import time
    import json
    # from modules.file_options import file_options as fo
    from modules import topology as tpl
    from modules import file_options as fop
    from modules import preprocess as prep
    from modules import postprocess as post
    from modules import solve as slv

    start = time.time()
    id_str = str(id).zfill(3)
    input_str = 'input files/inputs_' + id_str + '.json'
    f = open(input_str)
    inputs = json.load(f)
    file_options = inputs['file_options']

    # scale the ion potential if it is not zero to avoid numerical difficulties
    scale_ion = 0
    if inputs['boundary_conditions']['Vio_b'][0] != 0:
        Vel_b = inputs['boundary_conditions']['Vel_b']
        Vio_b = inputs['boundary_conditions']['Vio_b']
        inputs['boundary_conditions']['Vel_b'] = [Vel_b[0] - Vio_b[0], Vel_b[1] - Vio_b[0]]
        inputs['boundary_conditions']['Vio_b'] = [0, Vio_b[1] - Vio_b[0]]
        scale_ion = Vio_b[0]

    if file_options['new_case']:
        domain, domain_org = tpl.create_microstructure(
            inputs, 
            display=True,
            )
        
        domain, TPB_dict = tpl.topological_operations(
            inputs, 
            [domain, domain_org], 
            show_TPB=False, 
            show_TPB_variations=False,
            )
        
        if inputs['solver_options']['image_analysis_only']: return
        
        field_functions, _, bc_dict, K = prep.sourcefunc_calc(
            inputs, 
            TPB_dict, 
            plot_source_term=False,
            )
        
        masks_dict, indices = prep.get_indices_all(
            inputs, 
            domain, 
            TPB_dict,
            )
        
        J, rhs, sum_nb = prep.create_SOLE_individual(
            inputs, 
            bc_dict, 
            indices, 
            masks_dict, 
            K,
            )
        
        if file_options['save_case']:
            fop.save_case_individual(
                f'case_{file_options["id"]}',
                inputs, 
                indices, 
                J, 
                rhs, 
                field_functions, 
                masks_dict, 
                sum_nb, 
                TPB_dict, 
                bc_dict,
                )
            
        phi, residuals = prep.initilize_field_variables_individual(
            inputs, 
            masks_dict, 
            indices, 
            bc_dict,
            )

    elif file_options['load_case']:
        
        [
            inputs_old, 
            indices, 
            J, 
            rhs, 
            field_functions, 
            masks_dict, 
            sum_nb, 
            TPB_dict, 
            bc_dict,
            ] = fop.load_case_individual(file_options['case_id'])

        if inputs['microstructure'] != inputs_old['microstructure']: 
            raise Exception('Microstructure does not match!')

        if (inputs['boundary_conditions'] != inputs_old['boundary_conditions'] or 
            inputs['operating_conditions'] != inputs_old['operating_conditions']):
            field_functions, _, bc_dict = prep.sourcefunc_calc(
                inputs, 
                TPB_dict,
                )
            
            J, rhs, sum_nb = prep.create_SOLE_individual(
                inputs, 
                bc_dict, 
                indices, 
                masks_dict,
                )
            
            if inputs['file_options']['save_case']:
                fop.save_case_individual(
                    f'case_{file_options["id"]}',
                    inputs, 
                    indices, 
                    J, 
                    rhs, 
                    field_functions, 
                    masks_dict, 
                    sum_nb, 
                    TPB_dict, 
                    bc_dict,
                    )
                
        phi, residuals = prep.initilize_field_variables_individual(
            inputs, 
            masks_dict, 
            indices, 
            bc_dict,
            )

    elif file_options['load_case_data']:
        [
            inputs_old, 
            indices, 
            J, 
            rhs, 
            field_functions, 
            masks_dict, 
            sum_nb, 
            TPB_dict, 
            bc_dict, 
            phi, 
            residuals,
            ] = fop.load_case_data_individual(
                f'case_{file_options["id"]}', 
                f'data_{file_options["id"]}',)
        
        if inputs['microstructure'] != inputs_old['microstructure']: 
            raise Exception('Microstructure does not match!')
        if (inputs['boundary_conditions'] != inputs_old['boundary_conditions'] or 
            inputs['operating_conditions'] != inputs_old['operating_conditions']):
            field_functions, _, bc_dict = prep.sourcefunc_calc(
                inputs, 
                TPB_dict,
                )
            
            J, rhs, sum_nb = prep.create_SOLE_individual(
                inputs, 
                bc_dict, 
                indices, 
                masks_dict,
                )
            
            if inputs['file_options']['save_case']:
                fop.save_case_individual(
                    f'case_{file_options["id"]}',
                    inputs, 
                    indices, 
                    J, 
                    rhs, 
                    field_functions, 
                    masks_dict, 
                    sum_nb, 
                    TPB_dict, 
                    bc_dict,
                    )


    phi, residuals = slv.Newton_loop_individual(
        inputs, 
        J, 
        rhs, 
        phi, 
        indices, 
        field_functions, 
        masks_dict, 
        sum_nb, 
        residuals,
        )
    
    if file_options['save_data']:
        fop.save_data(
            f'data_{file_options["id"]}', 
            phi, 
            residuals,
            )


    if inputs['output_options']['show_residuals']:
        post.visualize_residuals(inputs, residuals)


    # scaling the electron and ion potentials to the original values
    phi[1] = phi[1] + scale_ion
    phi[2] = phi[2] + scale_ion
    dense_m = post.create_dense_matrices(
        inputs, 
        phi, 
        masks_dict, 
        indices, 
        field_functions, 
        TPB_dict,
        determine_gradients = True,
        )
        
    _ = post.visualize_3D_matrix(
        inputs, 
        dense_m, 
        TPB_dict,
        plots = {
            'cH2_1D':       False,
            'Vel_1D':       False,
            'Vio_1D':       True,
            'Ia_1D':        True,
            'eta_act_1D':   False,
            'eta_con_1D':   False,
            'cH2_3D':       False,
            'Vel_3D':       False,
            'Vio_3D':       True,
            'Ia_3D':        False,
            'eta_act_3D':   False,
            'eta_con_3D':   False,
        })
        
    end = time.time()
    print('Time elapsed: ', end - start)
    # return Ia_A_m[-2]