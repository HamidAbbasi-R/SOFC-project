file_options = {
    # Case and data management (only one of the following should be True)
    'new_case':             True,       # True: create a new case; False: load an existing case
    'load_case':            [False, 'test'],   # name of the case to load if new_case = False
    'load_case_data':       [False, 'test', 'test'],   # name of the case and the data to load if new_case = False
    
    # Save case and data
    'save_case':            [False, 'test'],       # True: save the case; False: don't save the case
    'save_data':            [False, 'test'],       # True: save the data; False: don't save 

    # modify input dictionary
    'modify_input':         [
                            # ['uf', [0.9,0.9,0.9]], 
                            # ['max_iter_non', 50],
                            # ['few_iters', 1000],  
                            # ['scl_fac', 1e5],    
                            # ['show_3D_plots', True],
                            # ['show_1D_plots', True],
                            # ['show_residuals', True],
                            # ['Vel_in', 1],
                            # ['cH2_offset', 0],
                            # ['Vel_offset', 0.0],
                            # ['Vio_offset', 0.0],
                            ]     
}

inputs = {
    # Domain generation
    'Nx':           20,         # Number of nodes in x-direction
    'Ny':           20,         # Number of nodes in y-direction
    'Nz':           20,         # Number of nodes in z-direction
    'dx':           1,#50e-9,      # Node spacing [m] 25nm
    'sig_gen':      2,          # Standard deviation of the Gaussian distribution used for generating the microstructure
    'seed':         [2,1],      # Seed for the random number generator
    
    # Volume fractions
    'vf_pores':     0.4,        # Volume fraction of the pores
    'vf_Ni':        0.3,        # Volume fraction of the Ni phase
    'vf_YSZ':       0.3,        # Volume fraction of the YSZ phase
    
    # Conductivity
    'K_pores':      1,#4.11e-5,    # Diffusivity of hydrogen [m2/s] - Shearing et al. 2010
    'K_Ni':         1,#2.17e6,     # Electronic conductivity [A/V/m] - Shearing et al. 2010  
    'K_YSZ':        1,#8.4,        # Ionic conductivity [A/V/m] - Shearing et al. 2010 
    
    # Operating conditions
    'T':            900+273.15, # Temperature [K]
    'P':            1,          # Pressure [atm]
    # 'I0a_l':        2.14e-4,    # Exchange current per TPB length, anode [A/m] - Shearing et al. 2010 
    
    # Boundary conditions
    'pH2_in':       0.4,        # Inlet hydrogen partial pressure [atm]
    'Vio_in':       0,          # Ionic voltage at electrolyte [V]
    'Vel_in':       1,          # Electronic voltage at anode [V]

    # Initial condition offset
    'cH2_offset':   0,          # Offset of the initial hydrogen concentration [kg/m3]
    'Vel_offset':   0,        # Offset of the initial electronic voltage [V]
    'Vio_offset':   0,        # Offset of the initial ionic voltage [V]

    # Solver
    'max_iter_non': 50,         # Maximum number of iterations for the nonlinear solver
    'max_iter_lin': 20,          # Maximum number of iterations for the linear solver
    'tol':          1e-20,       # Tolerance for the nonlinear solver
    'uf':   [0.9, 0.9, 0.9],  # Underrelaxation factor for the nonlinear solver [cH2, Vel, Vio]
    'few_iters':    1000,         # Number of iterations for the first few iterations after which the underrelaxation factor is increased and the max_iter_lin is set to infinity.
    'iter_swap':    10,         # Number of iterations after which the scaling of the Jacobian matrix is changed.
    'scl_fac':      1e5,       # Scaling factor for the Jacobian matrix

    # Output
    'show_residuals':    True,       # display convergence history
    'show_3D_plots':     True,       # display 3D plots
    'show_1D_plots':     True,       # display 1D plots
}


# import the necessary modules
import various_functions as vf


# Create/Load the case and data
if file_options['new_case']:
    domain = vf.create_microstructure(inputs)
    domain, TPB_dict = vf.topological_operations(inputs, domain)
    field_functions, thd, bc_dict = vf.sourcefunc_calc(inputs, TPB_dict)
    indices_dict, masks_dict = vf.masks_and_vectors(inputs, domain, TPB_dict)
    J, rhs, sum_nb = vf.create_SOLE(inputs, indices_dict, masks_dict, bc_dict)
    if file_options['save_case'][0]:
        vf.save_case(file_options['save_case'][1], 
            inputs, TPB_dict, field_functions, thd, 
            indices_dict, masks_dict, J, rhs, sum_nb, bc_dict)
    phi, residuals = vf.initilize_field_variables(inputs, masks_dict)

elif file_options['load_case'][0]:
    inputs, TPB_dict, \
        field_functions, thd, \
        indices_dict, masks_dict, \
        J, rhs, sum_nb, bc_dict = vf.load_case(file_options['load_case'][1])
    if file_options['modify_input'] != []:
        for i in range(len(file_options['modify_input'])):
            inputs[file_options['modify_input'][i][0]] = file_options['modify_input'][i][1]
    phi, residuals = vf.initilize_field_variables(inputs, masks_dict)

elif file_options['load_case_data'][0]:
    inputs, TPB_dict, \
        field_functions, thd, \
        indices_dict, masks_dict, \
        J, rhs, sum_nb, bc_dict, \
        phi, residuals = vf.load_case_data(file_options['load_case_data'][1], file_options['load_case_data'][2])
    if file_options['modify_input'] != []:
        for i in range(len(file_options['modify_input'])):
            inputs[file_options['modify_input'][i][0]] = file_options['modify_input'][i][1]


# Solving the case
phi, residuals = vf.Newton_loop(inputs, J, rhs, phi, indices_dict, masks_dict, bc_dict, field_functions, sum_nb, residuals, thd)


# Visualize the results
if file_options['save_data'][0]:
    vf.save_data(file_options['save_data'][1], phi, residuals)

if inputs['show_residuals']:
    vf.visualize_residuals(residuals)

if inputs['show_3D_plots'] or inputs['show_1D_plots']:
    eta_conc_mat = vf.create_field_variable(inputs, phi, indices_dict, field_functions['eta_con'])
    Ia_mat = vf.create_TPB_field_variable(inputs, phi, indices_dict, masks_dict, field_functions['Ia'])
    eta_act_mat = vf.create_TPB_field_variable(inputs, phi, indices_dict, masks_dict, field_functions['eta_act'])
    vf.visualize_3D_matrix(
        inputs, phi, masks_dict, TPB_dict,
        titles = ['cH2', 'Vel', 'Vio', 'Ia_TPB'],
        cH2 = True, 
        Vel = True, 
        Vio = True, 
        # field_mat = eta_conc_mat, 
        TPB_mat = Ia_mat
        )
