def create_domain(inputs):
    # create the entire domain
    import microstructure as ms
    domain = ms.create_phase_data(
        voxels = [inputs['Nx'],inputs['Ny'],inputs['Nz']],
        vol_frac = [inputs['vf_pores'],inputs['vf_Ni'],inputs['vf_YSZ']],
        sigma = inputs['sig_gen'],
        seed = inputs['seed'],
        display = False,
        )
    return domain

def topological_operations(inputs, domain):
    """
    domain topological operations
    """
    import microstructure as ms
    print("Domain topological operations...", end='')
    # removing thin boundaries
    # remove thin boundaries to avoid numerical error for the case of Neumann BCs.
    # don't remove thin boundaries for periodic boundary conditions. it will cause problems.
    domain = ms.remove_thin_boundaries(domain.astype(float))
    # extract the domain that should be solved. ds is short for Domain for Solver.
    # when periodic boundary condition is used, percolation analysis should not be done.
    domain, _, _ = ms.percolation_analysis(domain)

    # measure the triple phase boundary and create a mask for source term
    TPB_mask, TPB_density, vertices, lines = ms.measure_TPB(domain, inputs['dx'])
    print("Done!")

    TPB_dict = {
        'TPB_mask': TPB_mask,
        'TPB_density': TPB_density,
        'vertices': vertices,
        'lines': lines
    }
    return domain, TPB_dict

def sourcefunc_calc(inputs, TPB_dict):
    print("Source function calculations...", end='')
    from sympy import diff, exp, lambdify, log, simplify, symbols
    import numpy as np
    # Constants
    MH2 = 2.01588e-3        # Molar weight of hydrogen [kg/mol]
    MH2O = 18.01528e-3      # Molar weight of water [kg/mol]
    rhoH2 = 0.0251          # Density of hydrogen [kg/m^3]
    rhoH2O = 997            # Density of water [kg/m^3]
    R = 8.31446261815324    # Universal gas constant [J/mol/K]
    F = 96485.33289         # Faraday's constant [C/mol]
    aa = 0.5                # anode transfer coefficient []

    T = inputs['T']
    # I0a_l = inputs['I0a_l']

    # boundary conditions
    pH2_b = inputs['pH2_in']

    # other variables
    pH2O_b = inputs['P'] - pH2_b          # partial pressure of H2O [atm]
    ptot = pH2_b + pH2O_b       # total pressure [atm]
    nH2_b = pH2_b/ptot          # mole fraction of hydrogen @ anode boundary, bulk []
    nH2O_b = pH2O_b/ptot        # mole fraction of water @ anode boundary, bulk []
    cH2_b = rhoH2*nH2_b         # Hydrogen concentration @ anode boundary, bulk [kg/m^3] - boundary condition
    cH2O_b = rhoH2O*nH2O_b      # Water concentration @ anode boundary, bulk [kg/m^3]

    # symbolic expressions
    cH2 = symbols('cH2')    # hydrogen mass fraction [pores]
    Vel = symbols('Vel')    # electron potential [Nickel]
    Vio = symbols('Vio')    # ion potential [YSZ]

    nH2 = cH2/rhoH2             # mole fraction of hydrogen, symbolic []
    pH2 = ptot*nH2         # partial pressure of hydrogen, symbolic [atm]
    pH2O = ptot - pH2           # partial pressure of water, symbolic [atm]
    I0a_l = 31.4 * (pH2*1e5)**(-0.03) * (pH2O*1e5)**(0.4) * np.exp(-152155/R/T)     # Exchange current per TPB length, anode [A/m], Prokop et al. 2018
    I0a = I0a_l*TPB_dict['TPB_density']      # Exchange current density, anode [A/m^3]
    nH2O = pH2O/ptot       # mole fraction of water, symbolic []

    # Tseronis et al. 2012 model for anode current density
    eta_conc_a = R*T/2/F*log(nH2_b/nH2*nH2O/nH2O_b)         # ??? (make sure, reverse?) anode concentration overpotential [V]
    eta_act_a = Vel - Vio - eta_conc_a                      # ??? anode activation overpotential [V]
    Ia = I0a*(exp(2*aa*F*eta_act_a/R/T)
            -exp(-2*(1-aa)*F*eta_act_a/R/T))               # ??? anode current density [A/m^3]

    # Shearing et al. 2010 model for anode current density
    # Voe = 1.23 - 2.304e-4*(T-298.15)        # standard potential [V]
    # Veq = Voe - R*T/2/F*log(nH2O/nH2)      # equilibrium potential [V]
    # eta = Veq - Vio + Vel                   # overpotential [V]
    # Ia = I0a*(exp(2*aa*F*eta/R/T)
    #          -exp(-2*(1-aa)*F*eta/R/T))               # ??? anode current density [A/m^3]

    # initialize the source function list
    source_func = [0]*3         # initialization
    source_func[0] = simplify(-Ia/2/F*MH2)   # mass [kg/m3/s]
    source_func[1] = simplify(-Ia)            # electron [A/m3]
    source_func[2] = simplify(Ia)           # ion [A/m3]
    expected_sign = [-1,-1,1]             # expected sign of the source terms
    # source_func[0] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[1] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[2] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case

    # source term treatment
    f = [0]*3
    f[0] = lambdify([cH2, Vel, Vio], source_func[0])    #[kg/m3/s]
    f[1] = lambdify([cH2, Vel, Vio], source_func[1])    #[A/m3]
    f[2] = lambdify([cH2, Vel, Vio], source_func[2])    #[A/m3]

    fp = [0]*3
    fp[0] = lambdify([cH2, Vel, Vio], diff(source_func[0], cH2))    #[1/s]
    fp[1] = lambdify([cH2, Vel, Vio], diff(source_func[1], Vel))    #[A/V/m3]
    fp[2] = lambdify([cH2, Vel, Vio], diff(source_func[2], Vio))    #[A/V/m3]

    # Threshold values
    cH2_min = 1e-7             # minimum concentration [kg/m^3]
    cH2_max = cH2_b         # maximum concentration [kg/m^3]
    Vel_min = 0             # minimum electronic potential [V]
    Vel_max = inputs['Vel_in']             # maximum electronic potential [V]
    Vio_min = inputs['Vio_in']             # minimum ionic potential [V]
    Vio_max = 3             # maximum ionic potential [V]
    thd = [[cH2_min,cH2_max], [Vel_min,Vel_max], [Vio_min,Vio_max]]


    field_functions = {
        'f': f,
        'fp': fp,
        'signs': expected_sign,
        'eta_act': lambdify([cH2, Vel, Vio], eta_act_a),
        'eta_con': lambdify([cH2], eta_conc_a),
        'Ia': lambdify([cH2, Vel, Vio], Ia)}
    
    bc = [{
    # boundary conditions:
    # when two perpendicular pair of boundaries are set to periodic (e.g., south/north and east/west)
    # the edges of these boundaries will not be solved (aP will be zero at those nodes). 
    # this problem can be resolved with some extra work, but for now, try not to use two perpendicular 
    # periodic boundaries. in the case of SOFC microstructure, two perpendicular periodic boundaries are
    # not realistic in the first place.
    # units for Neumann BCs are [W/m^2], for heat equation:
    # for charge conservation model replace [W] with [A],
    # for mass conservation model replace [W] with [kg/s]
    
    # the first phase [pores] ??
    # anode side
    'West':   ['Dirichlet', cH2_b],       
    # electrolyte side
    'East':   ['Neumann', 0],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{

    # the second phase [Nickel - electron] ??
    # anode side
    'West':   ['Dirichlet', inputs['Vel_in']],         
    # electrolyte side
    'East':   ['Neumann', 0],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{

    # the third phase [YSZ - ion] ??
    # anode side
    'West':   ['Neumann', 0],           
    # electrolyte side
    'East':   ['Dirichlet', inputs['Vio_in']],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]}]

    bc_dict = bc

    print("Done!")
    return field_functions, thd, bc_dict

def masks_and_vectors(inputs, domain, TPB_dict):
        
    import numpy as np
    domain -= 1
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    L = N[0]*N[1]*N[2]         # total number of nodes
    # ds & ds_lin are boolean masks
    ds = [0]*3
    ds[0] = np.zeros(domain.shape, dtype=bool)
    ds[1] = np.zeros(domain.shape, dtype=bool)
    ds[2] = np.zeros(domain.shape, dtype=bool)
    ds[0][domain==0] = True
    ds[1][domain==1] = True
    ds[2][domain==2] = True
    ds_lin = [0]*3
    ds_lin[0] = ds[0].reshape(L)
    ds_lin[1] = ds[1].reshape(L)
    ds_lin[2] = ds[2].reshape(L)
    # Creating vectors (ip, jp, kp) to store the indices of the nodes of each phase
    ip, jp, kp = [0]*3, [0]*3, [0]*3
    ip[0], jp[0], kp[0] = np.where(ds[0])    # indices of points in the solver domain
    ip[1], jp[1], kp[1] = np.where(ds[1])
    ip[2], jp[2], kp[2] = np.where(ds[2])
    # indices of points not in the solver domain (not percolating)
    ipNP, jpNP, kpNP = np.where(~ds[0] & ~ds[1] & ~ds[2])  
    # Finding the indices of boundary nodes (b_ind)
    b_ind = [0,0,0]
    for p in [0,1,2]:
        b_ind[p] = np.array(np.where(np.logical_or.reduce((
            ip[p]==0, 
            ip[p]==N[0]-1, 
            jp[p]==0, 
            jp[p]==N[1]-1, 
            kp[p]==0, 
            kp[p]==N[2]-1
            ))))  
    # Finding the indices of interior points (ip_int, jp_int, kp_int)
    ip_int, jp_int, kp_int = [0,0,0], [0,0,0], [0,0,0]    # initialization of index of interior points
    for p in [0,1,2]:
        ip_int[p] = np.delete(ip[p], b_ind[p])
        jp_int[p] = np.delete(jp[p], b_ind[p])
        kp_int[p] = np.delete(kp[p], b_ind[p])
    # create index arrays where source_mask==True and source_mask==False
    L_int_s, L_int_ns = [0,0,0], [0,0,0]
    ip_int_s, jp_int_s, kp_int_s = [0,0,0], [0,0,0], [0,0,0]
    ip_int_ns, jp_int_ns, kp_int_ns = [0,0,0], [0,0,0], [0,0,0]
    ds_lin_s = np.zeros_like(ds_lin[0], dtype=bool)

    TPB_mask = TPB_dict['TPB_mask']
    for p in [0,1,2]:
        L_int_s[p] = np.sum(np.logical_and(TPB_mask[1:-1,1:-1,1:-1], ds[p][1:-1,1:-1,1:-1]))
        L_int_ns[p] = np.sum(np.logical_and(np.logical_not(TPB_mask[1:-1,1:-1,1:-1]), ds[p][1:-1,1:-1,1:-1]))
        
        ip_int_s[p] = np.zeros(shape = L_int_s[p], dtype = int)      # interior points (source==True)
        jp_int_s[p] = np.zeros(shape = L_int_s[p], dtype = int)
        kp_int_s[p] = np.zeros(shape = L_int_s[p], dtype = int)

        ip_int_ns[p] = np.zeros(shape = L_int_ns[p], dtype = int)    # interior points (source==False)
        jp_int_ns[p] = np.zeros(shape = L_int_ns[p], dtype = int)
        kp_int_ns[p] = np.zeros(shape = L_int_ns[p], dtype = int)

        cntr_s = 0      # counter for source==True
        cntr_ns = 0     # counter for source==False
        for n in range(len(ip_int[p])):
            i,j,k = ip_int[p][n], jp_int[p][n], kp_int[p][n]
            if TPB_mask[i,j,k]:
                index = i*N[1]*N[2] + j*N[2] + k
                ip_int_s[p][cntr_s], jp_int_s[p][cntr_s], kp_int_s[p][cntr_s] = i,j,k
                ds_lin_s[index] = True
                cntr_s += 1
            else:
                ip_int_ns[p][cntr_ns], jp_int_ns[p][cntr_ns], kp_int_ns[p][cntr_ns] = i,j,k
                cntr_ns += 1

    indices_dict = {
        'ip': ip, 'jp': jp, 'kp': kp,
        'ipNP': ipNP, 'jpNP': jpNP, 'kpNP': kpNP,
        'b_ind': b_ind,
        'ip_int': ip_int, 'jp_int': jp_int, 'kp_int': kp_int,
        'ip_int_s': ip_int_s, 'jp_int_s': jp_int_s, 'kp_int_s': kp_int_s,
        'ip_int_ns': ip_int_ns, 'jp_int_ns': jp_int_ns, 'kp_int_ns': kp_int_ns,
        'L_int_s': L_int_s, 'L_int_ns': L_int_ns}
    
    masks_dict = {
        'ds': ds, 
        'ds_lin': ds_lin,     
        'ds_lin_s': ds_lin_s}

    return indices_dict, masks_dict

def treat_edges(inputs, J, rhs, indices_dict):
    """
    Setting up the Jacobian matrix for the edges (interior nodes).
    Arbitrarily set the edges and corners to zero. 
    This choice does not affect the final results, since the values 
    at the edges (and corners) do not diffuse in the domain.
    """
    import numpy as np
    # global J, rhs
    ip, jp, kp = indices_dict['ip'], indices_dict['jp'], indices_dict['kp']
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    K = [inputs['K_pores'], inputs['K_Ni'], inputs['K_YSZ']]
    dx = inputs['dx']

    for p in [0,1,2]:
        i = 0
        for n in np.array(np.where(ip[p]==i)).T:
            j,k = jp[p][n], kp[p][n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1:
                index = i*N[1]*N[2] + j*N[2] + k
                J[index, :] = 0
                J[index, index] = -1*K[p]*dx
                rhs[index] = 0
        i = N[0]-1
        for n in np.array(np.where(ip[p]==i)).T:
            j,k = jp[p][n], kp[p][n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1:
                index = i*N[1]*N[2] + j*N[2] + k
                J[index, :] = 0
                J[index, index] = -1*K[p]*dx
                rhs[index] = 0
        j = 0
        for n in np.array(np.where(jp[p]==j)).T:
            i,k = ip[p][n], kp[p][n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1:
                index = i*N[1]*N[2] + j*N[2] + k
                J[index, :] = 0
                J[index, index] = -1*K[p]*dx
                rhs[index] = 0
        j = N[1]-1
        for n in np.array(np.where(jp[p]==j)).T:
            i,k = ip[p][n], kp[p][n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1:
                index = i*N[1]*N[2] + j*N[2] + k
                J[index, :] = 0
                J[index, index] = -1*K[p]*dx
                rhs[index] = 0
        k = 0
        for n in np.array(np.where(kp[p]==k)).T:
            i,j = ip[p][n], jp[p][n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1:
                index = i*N[1]*N[2] + j*N[2] + k
                J[index, :] = 0
                J[index, index] = -1*K[p]*dx
                rhs[index] = 0
        k = N[2]-1
        for n in np.array(np.where(kp[p]==k)).T:
            i,j = ip[p][n], jp[p][n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1:
                index = i*N[1]*N[2] + j*N[2] + k
                J[index, :] = 0
                J[index, index] = -1*K[p]*dx
                rhs[index] = 0
    return J, rhs

def create_SOLE(inputs, indices_dict, masks_dict, bc_dict):

    from scipy.sparse import lil_matrix
    import numpy as np

    bc = bc_dict
    ip = indices_dict['ip']
    jp = indices_dict['jp']
    kp = indices_dict['kp']
    ipNP = indices_dict['ipNP']
    jpNP = indices_dict['jpNP']
    kpNP = indices_dict['kpNP']
    ip_int = indices_dict['ip_int']
    jp_int = indices_dict['jp_int']
    kp_int = indices_dict['kp_int']
    
    ds = masks_dict['ds']

    K = [inputs['K_pores'], inputs['K_Ni'], inputs['K_YSZ']]
    dx = inputs['dx']

    # initializing the Jacobian matrix and the right hand side of the system of equaitons
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    L = N[0] * N[1] * N[2]
    J = lil_matrix((L,L), dtype = float) # Jacobian sparse matrix
    rhs = np.zeros(shape = L, dtype = float) # right hand side vector
    # vector to collect the sum of the coefficients of the neighbors
    sum_nb = np.zeros(shape = L, dtype = float)


    # arbitrarily set the points that are not in the solver domain (ipNP, jpNP, kpNP) to zero
    # this is done to avoid any diagonal elements=zero in the Jacobian matrix
    for n in range(len(ipNP)):
        i,j,k = ipNP[n], jpNP[n], kpNP[n]
        index = i*N[1]*N[2] + j*N[2] + k
        # if len(J[index,:].data[0])!=0: # if the row is not empty
        #     raise Exception('The row is not empty')
        J[index, index] = -1*K[0]*dx        # why K[0]? why not K[1] or K[2]? does it make any differences on convergence rate?
        rhs[index] = 0

    # Setting up the Jacobian matrix and the right hand side vector for the boundary nodes
    print(f'Writing Jacobian matrix (boundary nodes)', end = '...')
    for p in [0,1,2]:    
        J, rhs = boundaries_all(J, rhs, bc[p], dx, [ip[p], jp[p], kp[p]], N, ds[p], K[p])
    print('Done!')
    
    print(f'Writing Jacobian matrix (edge and corner nodes)', end = '...')
    J, rhs = treat_edges(inputs, J, rhs, indices_dict)
    print('Done!')

    # Setting up the off diagonal elements of Jacobian matrix for the interior nodes
    print(f'Writing Jacobian matrix (interior nodes)', end = '...')
    for p in [0,1,2]:
        for n in range(len(ip_int[p])):#tqdm(range(len(ip_int[p]))):
            i,j,k = ip_int[p][n], jp_int[p][n], kp_int[p][n]
            index = i*(N[1]*N[2]) + j*N[2] + k
            # if len(J[index,:].data[0])!=0: raise Exception('The row is not empty')
            aW = J[index, index - N[1]*N[2]] = -K[p]*ds[p][i-1,j,k]*dx   # aW   [W/K]
            aE = J[index, index + N[1]*N[2]] = -K[p]*ds[p][i+1,j,k]*dx   # aE   [W/K]
            aS = J[index, index - N[2]]      = -K[p]*ds[p][i,j-1,k]*dx   # aS   [W/K]
            aN = J[index, index + N[2]]      = -K[p]*ds[p][i,j+1,k]*dx   # aN   [W/K]
            aB = J[index, index - 1]         = -K[p]*ds[p][i,j,k-1]*dx   # aB   [W/K]
            aT = J[index, index + 1]         = -K[p]*ds[p][i,j,k+1]*dx   # aT   [W/K]
            sum_nb[index] = aW + aE + aS + aN + aB + aT
            J[index, index] = -sum_nb[index]                          # aP   [W/K]
    print('Done!')

    # Setting up the Jacobian matrix and the right hand side vector for the interior nodes (source==False)
    # print(f'Writing Jacobian matrix (interior nodes [∀ x ∉ Source], main diagonal elements)', end = '...')
    # for p in [0,1,2]:
    #     for n in tqdm(range(len(ip_int_ns[p]))):
    #         i,j,k = ip_int_ns[p][n], jp_int_ns[p][n], kp_int_ns[p][n]
    #         index = i*(N[1]*N[2]) + j*N[2] + k
    #         aW = J[index, index - N[1]*N[2]] 
    #         aE = J[index, index + N[1]*N[2]] 
    #         aS = J[index, index - N[2]]      
    #         aN = J[index, index + N[2]]      
    #         aB = J[index, index - 1]         
    #         aT = J[index, index + 1]         
    #         J[index, index] = -(aW+aE+aS+aN+aB+aT) #-np.sum(J[index,:])   
    #         # rhs[index] = 0    # no need to change rhs here since it is initialized to zero
    # print('Done!')

    J_csr = J.tocsr()
    J_csr.eliminate_zeros()
    J = J_csr.tolil()

    return J, rhs, sum_nb

def threshold(phi_new, masks_dict, thd):
    """
    this function thresholds the phase field variable between feasible minimum and maimum values.
    """
    ds_lin = masks_dict['ds_lin']
    infeas_nodes = [0]*3
    for p in [0,1,2]: 
        temp = phi_new[ds_lin[p]]
        infeas_nodes[p] = (len(temp[temp<thd[p][0]])+len(temp[temp>thd[p][1]])) / len(temp)*100
        temp[temp<thd[p][0]] = thd[p][0]
        temp[temp>thd[p][1]] = thd[p][1]
        phi_new[ds_lin[p]] = temp

    return phi_new, infeas_nodes

def initilize_field_variables(inputs, masks_dict):
    import numpy as np
    # Initializing the field variables
    # It is important to set the initial condition to some value that is close to the expected solution.
    ds_lin = masks_dict['ds_lin']
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    L = N[0]*N[1]*N[2]         # total number of nodes
    # residuals = np.ones([3,inputs['max_iter_non']])
    residuals = [[],[],[]]

    rhoH2 = 0.08988e3       # Density of hydrogen [kg/m^3]
    cH2_b = inputs['pH2_in'] / inputs['P'] * rhoH2

    init_cond = [cH2_b - inputs['cH2_offset'], 
                inputs['Vel_in'] - inputs['Vel_offset'],
                inputs['Vio_in'] + inputs['Vio_offset']]

    phi = np.ones(shape = L, dtype = float)
    phi[ds_lin[0]] = 0##init_cond[0]
    phi[ds_lin[1]] = 0#init_cond[1]
    phi[ds_lin[2]] = 0#init_cond[2]

    return phi, residuals

def Newton_loop(inputs, J, rhs, phi, indices_dict, masks_dict, bc_dict, field_functions, sum_nb, residuals, thd):
    import time
    from scipy.sparse import linalg
    import numpy as np
    
    # M_x = lambda x: spla.spsolve(identity(L), x)
    # M = spla.LinearOperator((L, L), M_x)
    uf = np.array(inputs['uf'])
    
    t = time.time()
    prev_iters = len(residuals[0])
    iter = prev_iters
    max_res = 1
    while iter < prev_iters+inputs['max_iter_non']:
        J, rhs = periodic_bc_update(inputs, J, rhs, phi, bc_dict, indices_dict)
        J, rhs, non_feas_source, cont_s = update_SOLE(inputs, J, rhs, phi, indices_dict, masks_dict, field_functions, sum_nb)
        J_scl, rhs_scl = matrix_scaling(inputs, J, rhs, masks_dict, iter)
        
        # Solving the scaled SOLE using GMRES algorithm [linear loop]
        if iter < inputs['few_iters']:
            phi_new, info = linalg.gmres(J_scl, rhs_scl, x0=phi, maxiter=inputs['max_iter_lin'], atol=inputs['tol'])
        else:
            phi_new, info = linalg.gmres(J_scl, rhs_scl, x0=phi, atol=inputs['tol'])

        max_res, residuals = error_monitoring(inputs, masks_dict, phi, phi_new, J_scl, rhs_scl, residuals, iter, non_feas_source, info, cont_s)
        # phi_new, _ = threshold(phi_new, masks_dict, thd)
        phi = update_phi(inputs, masks_dict, phi, phi_new, uf, iter)

        # check the convergence
        if max_res < inputs['tol'] and iter>5:
            break

        # check_periodicity(phi)
        iter += 1
    print(f'elapsed time: {time.time()-t}')

    return phi, residuals

def periodic_bc_update(inputs, J, rhs, phi, bc_dict, indices_dict):

    import numpy as np
    ip = indices_dict['ip']
    jp = indices_dict['jp']
    kp = indices_dict['kp']
    dx = inputs['dx']
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    K = [inputs['K_pores'], inputs['K_Ni'], inputs['K_YSZ']]
    bc = bc_dict

    #periodic boundary conditions
    for p in [0,1,2]:
        if bc[p]['East'][0] == 'Periodic':
            i_E, i_W = N[0]-1, 0
            for n in np.array(np.where(ip[p]==i_E)).T:
                j,k = jp[p][n], kp[p][n]
                if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
                index_E = i_E*(N[1]*N[2]) + j*N[2] + k
                index_W = i_W*(N[1]*N[2]) + j*N[2] + k
                J[index_E, index_E] = -1*K[p]*dx
                # aW[index_E], aE[index_E], aS[index_E], aN[index_E], aB[index_E], aT[index_E] = 0,0,0,0,0,0
                rhs[index_E] = -phi[index_W]*K[p]*dx
        if bc[p]['North'][0] == 'Periodic':
            j_N, j_S = N[1]-1, 0
            for n in np.array(np.where(jp[p]==j_N)).T:
                i,k = ip[p][n], kp[p][n]
                index_N = i*(N[1]*N[2]) + j_N*N[2] + k
                index_S = i*(N[1]*N[2]) + j_S*N[2] + k
                J[index_N, index_N] = -1*K[p]*dx
                # aW[index_N], aE[index_N], aS[index_N], aN[index_N], aB[index_N], aT[index_N] = 0,0,0,0,0,0
                rhs[index_N] = -phi[index_S]*K[p]*dx
        if bc[p]['Top'][0] == 'Periodic':
            k_T, k_B = N[2]-1, 0
            for n in np.array(np.where(kp[p]==k_T)).T:
                i,j = ip[p][n], jp[p][n]
                index_T = i*(N[1]*N[2]) + j*N[2] + k_T
                index_B = i*(N[1]*N[2]) + j*N[2] + k_B
                J[index_T, index_T] = -1*K[p]*dx   # [W/K]
                # aW[index_T], aE[index_T], aS[index_T], aN[index_T], aB[index_T], aT[index_T] = 0,0,0,0,0,0
                rhs[index_T] = -phi[index_B]*K[p]*dx   # [W]

    return J, rhs

def update_SOLE(inputs, J, rhs, phi, indices_dict, masks_dict, field_functions, sum_nb):
    import numpy as np
    # global J, rhs, J_scl, rhs_scl, non_feas_source
    # interior points (source==True)
    ip_int_s = indices_dict['ip_int_s']
    jp_int_s = indices_dict['jp_int_s']
    kp_int_s = indices_dict['kp_int_s']
    ds = masks_dict['ds']

    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    f = field_functions['f']
    fp = field_functions['fp']
    signs = field_functions['signs']
    dx = inputs['dx']
    contribution_source = np.zeros_like(sum_nb)

    phi_dense = phi.reshape(N)
    non_feas_source = [0]*3
    for p in [0,1,2]:
        for n in range(len(ip_int_s[p])):
            i,j,k = ip_int_s[p][n], jp_int_s[p][n], kp_int_s[p][n]
            index = i*(N[1]*N[2]) + j*N[2] + k

            cH2_i = phi[index] if p==0 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[0][i-1:i+2,j-1:j+2,k-1:k+2]])
            Vel_i = phi[index] if p==1 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[1][i-1:i+2,j-1:j+2,k-1:k+2]])
            Vio_i = phi[index] if p==2 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[2][i-1:i+2,j-1:j+2,k-1:k+2]])
            
            f_i = f[p](cH2_i, Vel_i, Vio_i)
            fp_i = fp[p](cH2_i, Vel_i, Vio_i)

            # if sign of f is not as expected, replace it with zero and count it as non-feasible source term
            # Unexpected signs can happen in the first few iterations
            # if f_i*signs[p] < 0:
            #     f_i = 0
            #     fp_i = 0
            #     non_feas_source[p] += 1

            contribution_source[index] = fp_i*dx**3 / sum_nb[index]
            J[index, index] = -sum_nb[index] - fp_i*dx**3       # [W/K] [S or A/V] [mass???]
            rhs[index] = (f_i - fp_i*phi[index]) * dx**3     # [W] 
    
    return J, rhs, non_feas_source, contribution_source

def matrix_scaling(inputs, J, rhs, masks_dict, iter):
    from scipy.sparse import diags
    # Scaling the Jacobian matrix and RHS vector 
    # Basically same as using Jacobi preconditioner (other preconditioners should also be considered)
    ds_lin = masks_dict['ds_lin']
    ds_lin_s = masks_dict['ds_lin_s']
    scl_vec = 1/J.diagonal(0).ravel()
    # if iter % (4*inputs['iter_swap']) < 1*(inputs['iter_swap']):
    #     scl_vec[ds_lin[0]] *= inputs['scl_fac']
    # elif (iter % (4*inputs['iter_swap']) >= 1*(inputs['iter_swap'])) and (iter % (4*inputs['iter_swap']) < (2*inputs['iter_swap'])):
    #     scl_vec[ds_lin[1]] *= inputs['scl_fac']
    # elif (iter % (4*inputs['iter_swap']) >= 2*(inputs['iter_swap'])) and (iter % (4*inputs['iter_swap']) < (3*inputs['iter_swap'])):
    #     scl_vec[ds_lin[2]] *= inputs['scl_fac']
    # else:
    #     scl_vec[ds_lin_s] *= inputs['scl_fac']

    scl_mat = diags(scl_vec)

    J_scl = scl_mat @ J
    rhs_scl = scl_mat @ rhs

    return J_scl, rhs_scl

def error_monitoring(inputs, masks_dict, phi, phi_new, J_scl, rhs_scl, residuals, iter, non_feas_source, info, cont_s):
    import numpy as np
    import microstructure as ms
    
    ds_lin = masks_dict['ds_lin']
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    L = N[0] * N[1] * N[2]
    # Computation of residuals 
    # res = rhs_scl - J_scl.tocsr()@phi_new       # first method
    res = np.abs(phi_new - phi)                 # second method

    # remove edges from residuals
    res = ms.remove_edges(res.reshape(N)).reshape(L)

    # Contribution of source terms to the rhs vector
    cont = np.zeros(3)
    cont[0] = np.average(np.abs(cont_s[np.logical_and(ds_lin[0],cont_s!=0)]))
    cont[1] = np.average(np.abs(cont_s[np.logical_and(ds_lin[1],cont_s!=0)]))
    cont[2] = np.average(np.abs(cont_s[np.logical_and(ds_lin[2],cont_s!=0)]))

    ds_lin = masks_dict['ds_lin']
    mask0 = np.logical_and(ds_lin[0], ~np.isnan(res))#ds_lin[0]#
    mask1 = np.logical_and(ds_lin[1], ~np.isnan(res))#ds_lin[1]#
    mask2 = np.logical_and(ds_lin[2], ~np.isnan(res))#ds_lin[2]#
    
    residuals[0].append(np.linalg.norm(res[mask0])/
                        np.linalg.norm(phi_new[mask0]))
    residuals[1].append(np.linalg.norm(res[mask1])/
                        np.linalg.norm(phi_new[mask1]))
    residuals[2].append(np.linalg.norm(res[mask2])/
                        np.linalg.norm(phi_new[mask2]))

    max_res = max(residuals[0][iter], residuals[1][iter], residuals[2][iter])
    if iter % 10 == 0:
        print('\nIter:  res cH2:       res Vel:       res Vio:       info:    S_contribution:                  infeasible_S:')
        print('-----------------------------------------------------------------------------------------------------------')
    print(f'{iter:<7}{residuals[0][iter]:<15.2e}{residuals[1][iter]:<15.2e}{residuals[2][iter]:<15.2e}{info:<9}{cont[0]:<9.1e},{cont[1]:<9.1e},{cont[2]:<15.1e}{non_feas_source}')
    
    return max_res, residuals

def update_phi(inputs, masks_dict, phi, phi_new, uf, iter):
    # global phi, uf
    ds_lin = masks_dict['ds_lin']
    if iter > inputs['few_iters']:
        uf = 1 - (1-uf)/10
        # uf = np.ones_like(uf)*1.5

    phi[ds_lin[0]] = uf[0]*phi_new[ds_lin[0]] + (1-uf[0])*phi[ds_lin[0]]
    phi[ds_lin[1]] = uf[1]*phi_new[ds_lin[1]] + (1-uf[1])*phi[ds_lin[1]]
    phi[ds_lin[2]] = uf[2]*phi_new[ds_lin[2]] + (1-uf[2])*phi[ds_lin[2]]
    
    return phi

def check_periodicity(inputs, masks_dict, phi):
    import numpy as np
    # Checking periodicity
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    ds = masks_dict['ds']
    phi_dense = phi.reshape(N)
    if np.min(phi_dense[-1,:,:][ds[-1,:,:]] == phi_dense[0 ,:,:][ds[0 ,:,:]])==False:
        print('Periodic BCs not satisfied east-west')
    if np.min(phi_dense[:,-1,:][ds[:,-1,:]] == phi_dense[:,0 ,:][ds[:,0 ,:]])==False:
        print('Periodic BCs not satisfied north-south')
    if np.min(phi_dense[:,:,-1][ds[:,:,-1]] == phi_dense[:,:,0 ][ds[:,:,0 ]])==False:
        raise Exception('Periodic BCs not satisfied top-bottom')

def visualize_residuals(residuals):
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # visualize the error
    x = np.arange(len(residuals[0]))

    r0 = np.stack((x, residuals[0]), axis=-1)
    df0 = pd.DataFrame(r0, columns=['iteration', 'residual'])
    df0.insert(2, 'Variable', 'Hydrogen concentration')

    r1 = np.stack((x, residuals[1]), axis=-1)
    df1 = pd.DataFrame(r1, columns=['iteration','residual'])
    df1.insert(2, 'Variable', 'Electron potential')

    r2 = np.stack((x, residuals[2]), axis=-1)
    df2 = pd.DataFrame(r2, columns=['iteration','residual'])
    df2.insert(2, 'Variable', 'Ion potential')

    df = pd.concat([df0, df1, df2])
    fig = px.line(df, x='iteration', y='residual', color='Variable', log_y=True)
    fig.show()

def create_TPB_field_variable(inputs, phi, indices_dict, masks_dict, func):
    # visualize a function on the TPB
    import numpy as np
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    ds = masks_dict['ds']
    ip_int_s = indices_dict['ip_int_s']
    jp_int_s = indices_dict['jp_int_s']
    kp_int_s = indices_dict['kp_int_s']

    mat = np.zeros(shape = N)
    phi_dense = phi.reshape(N)
    for p in [0,1,2]:
        for n in range(len(ip_int_s[p])):
            i,j,k = ip_int_s[p][n], jp_int_s[p][n], kp_int_s[p][n]
            if close_to_edge(inputs, i,j,k): continue
            index = i*(N[1]*N[2]) + j*N[2] + k

            cH2_i = phi[index] if p==0 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[0][i-1:i+2,j-1:j+2,k-1:k+2]])
            Vel_i = phi[index] if p==1 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[1][i-1:i+2,j-1:j+2,k-1:k+2]])
            Vio_i = phi[index] if p==2 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[2][i-1:i+2,j-1:j+2,k-1:k+2]])

            mat[i,j,k] = func(cH2_i, Vel_i, Vio_i)
    mat[mat==0] = np.nan
    return mat

def close_to_edge(inputs, i,j,k):
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    if \
    (i==1 and j==1) or\
    (i==1 and k==1) or\
    (j==1 and k==1) or\
    (i==N[0]-2 and j==1) or\
    (i==N[0]-2 and k==1) or\
    (j==N[1]-2 and k==1) or\
    (i==1 and j==N[1]-2) or\
    (i==1 and k==N[2]-2) or\
    (j==1 and k==N[2]-2) or\
    (i==N[0]-2 and j==N[1]-2) or\
    (i==N[0]-2 and k==N[2]-2) or\
    (j==N[1]-2 and k==N[2]-2):
        return True

def create_field_variable(inputs, phi, indices_dict, func):

    import numpy as np
    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    ip = indices_dict['ip']
    jp = indices_dict['jp']
    kp = indices_dict['kp']

    field_mat = np.zeros(shape = N)
    for n in range(len(ip[0])):
        i,j,k = ip[0][n], jp[0][n], kp[0][n]
        index = i*N[1]*N[2] + j*N[2] + k
        field_mat[i,j,k] = func(phi[index])
    field_mat[field_mat==0] = np.nan
    return field_mat

def visualize_3D_matrix(inputs, phi, masks_dict, TPB_dict, titles, cH2=None, Vel=None, Vio=None, field_mat=None, TPB_mat=None):
    # visualize the solution
    import numpy as np
    import pyvista as pv
    import pandas as pd
    import plotly.express as px
    import microstructure as ms
    pv.set_plot_theme("document")

    N = [inputs['Nx'], inputs['Ny'], inputs['Nz']]
    ds = masks_dict['ds']
    vertices = TPB_dict['vertices']
    lines = TPB_dict['lines']
    phi_dense = phi.reshape(N)
    mats = []
    thds = []
    log_scale = []

    # removing the phi points outside the domain for visualization purposes
    if cH2 is not None:
        sol_cH2 = np.copy(phi_dense)
        sol_cH2[ds[0] == False] = np.nan
        # sol_cH2 = ms.remove_edges(sol_cH2)
        mats.append(sol_cH2)
        thds.append(())
        log_scale.append(False)

    if Vel is not None:
        sol_Vel = np.copy(phi_dense)
        sol_Vel[ds[1] == False] = np.nan
        # sol_Vel = ms.remove_edges(sol_Vel)
        mats.append(sol_Vel)
        thds.append(())
        log_scale.append(False)
    
    if Vio is not None:
        sol_Vio = np.copy(phi_dense)
        sol_Vio[ds[2] == False] = np.nan
        # sol_Vio = ms.remove_edges(sol_Vio)
        mats.append(sol_Vio)
        thds.append(())
        log_scale.append(False)

    if field_mat is not None:
        field_mat = ms.remove_edges(field_mat)    
        mats.append(field_mat)
        thds.append(())
        log_scale.append(False)

    if TPB_mat is not None:
        mats.append(TPB_mat)
        thds.append(())
        log_scale.append(True)
    
    TPB_mesh = pv.PolyData(vertices, lines=lines)

    if inputs['show_1D_plots']:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        x = np.arange(inputs['Nx'])*inputs['dx']*1e6
        cH2_lin = np.zeros(inputs['Nx'])
        cH2_min = np.zeros(inputs['Nx'])
        cH2_max = np.zeros(inputs['Nx'])
        cH2_c_up = np.zeros(inputs['Nx'])
        cH2_c_down = np.zeros(inputs['Nx'])

        Vel_lin = np.zeros(inputs['Nx'])
        Vel_min = np.zeros(inputs['Nx'])
        Vel_max = np.zeros(inputs['Nx'])
        Vel_c_up = np.zeros(inputs['Nx'])
        Vel_c_down = np.zeros(inputs['Nx'])

        Vio_lin = np.zeros(inputs['Nx'])
        Vio_min = np.zeros(inputs['Nx'])
        Vio_max = np.zeros(inputs['Nx'])
        Vio_c_up = np.zeros(inputs['Nx'])
        Vio_c_down = np.zeros(inputs['Nx'])

        Ia_lin = np.zeros(inputs['Nx'])
        Ia_min = np.zeros(inputs['Nx'])
        Ia_max = np.zeros(inputs['Nx'])
        Ia_c_up = np.zeros(inputs['Nx'])
        Ia_c_down = np.zeros(inputs['Nx'])

        for i in range(inputs['Nx']):
            a = sol_cH2[i, :, :][~np.isnan(sol_cH2[i, :, :])]
            cH2_lin[i] = np.average(a)
            cH2_max[i] = np.max(a)
            cH2_min[i] = np.min(a)
            cH2_c_down[i], cH2_c_up[i] = mean_confidence_interval(a)

            a = sol_Vel[i, :, :][~np.isnan(sol_Vel[i, :, :])]
            Vel_lin[i] = np.average(a)
            Vel_max[i] = np.max(a)
            Vel_min[i] = np.min(a)
            Vel_c_down[i], Vel_c_up[i] = mean_confidence_interval(a)

            a = sol_Vio[i, :, :][~np.isnan(sol_Vio[i, :, :])]
            Vio_lin[i] = np.average(a)
            Vio_max[i] = np.max(a)
            Vio_min[i] = np.min(a)
            Vio_c_down[i], Vio_c_up[i] = mean_confidence_interval(a)

            if i == 0 or i == inputs['Nx']-1:
                Ia_lin[i] = np.nan
                Ia_max[i] = np.nan
                Ia_min[i] = np.nan
                Ia_c_down[i], Ia_c_up[i] = np.nan, np.nan
            else:
                a = TPB_mat[i, :, :][~np.isnan(TPB_mat[i, :, :])]
                Ia_lin[i]  = np.average(a)
                Ia_max[i]  = np.max(a)
                Ia_min[i]  = np.min(a)
                Ia_c_down[i], Ia_c_up[i] = mean_confidence_interval(a)


        plot_with_continuous_error(x, cH2_lin, cH2_max, cH2_min, cH2_c_down, cH2_c_up, x_title='Distance from anode (µm)', y_title='Hydrogen concentration (kg/m3)', title='Hydrogen concentration (kgm-3)')
        plot_with_continuous_error(x, Ia_lin, Ia_max, Ia_min, Ia_c_down, Ia_c_up, x_title='Distance from anode (µm)', y_title='Exchange current density (A/m3)', title='Exchange current density (Am-3)')
        # plot_with_continuous_error(x, cH2_lin, cH2_max, cH2_min, x_title='Distance from anode (µm)', y_title='Hydrogen concentration (kg/m3)', title='Hydrogen concentration (kg/m3)')
        # plot_with_continuous_error(x, Vel_lin, Vel_max, Vel_min, x_title='Distance from anode (µm)', y_title='Electron potential (V)', title='Electron potential (V)')
        # plot_with_continuous_error([x,x], [Vio_lin, Vel_lin], [Vio_max, Vel_max], [Vio_min, Vel_min], [Vio_c_down, Vel_c_down], [Vio_c_up, Vel_c_up], x_title='Distance from anode (µm)', y_title='Ion and electron potential (V)', title='Ion and electron potential (V)')
        plot_with_continuous_error(x, Vio_lin, Vio_max, Vio_min, Vio_c_down, Vio_c_up, x_title='Distance from anode (µm)', y_title='Ion and electron potential (V)', title='Ion and electron potential (V)')


    if inputs['show_3D_plots']:
        ms.visualize_mesh(
            mat = mats,
            thd = thds,
            titles = titles,
            clip_widget = False, 
            TPB_mesh = TPB_mesh,
            log_scale = log_scale)

def save_case(case_name, 
        inputs, TPB_dict, 
        field_functions, thd,
        indices_dict, masks_dict, 
        J, rhs, sum_nb, bc_dict):
    # save the case
    import os
    from dill import dump, load
    if not os.path.exists('cases'):
        os.makedirs('cases')

    save_obj = [
        inputs, TPB_dict, 
        field_functions, thd,
        indices_dict, masks_dict, 
        J, rhs, sum_nb, bc_dict]
    
    with open('cases/' + case_name + '.pkl', 'wb') as case_file:
        dump(save_obj, case_file)

def save_data(data_name, phi, residuals):
    # save the data
    import os
    from dill import dump

    print('Saving data...', end='')
    if not os.path.exists('data'):
        os.makedirs('data')

    save_obj = [phi, residuals]
    with open('data/' + data_name + '.pkl', 'wb') as file:
        dump(save_obj, file)
    
    print('Done!')

def load_case(case_name):
    # load the case
    from dill import load
    with open('cases/' + case_name + '.pkl', 'rb') as file:
        inputs, TPB_dict, field_functions, thd, indices_dict, masks_dict, J, rhs, sum_nb, bc_dict = load(file)
    return inputs, TPB_dict, field_functions, thd, indices_dict, masks_dict, J, rhs, sum_nb, bc_dict

def load_case_data(case_name, data_name):
    # load the case
    from dill import load
    inputs, TPB_dict, field_functions, thd, indices_dict, masks_dict, J, rhs, sum_nb, bc_dict = load_case(case_name)
    # load the data
    with open('data/' + data_name + '.pkl', 'rb') as data_file:
        phi, residuals = load(data_file)
    return inputs, TPB_dict, field_functions, thd, indices_dict, masks_dict, J, rhs, sum_nb, bc_dict, phi, residuals

def boundaries_all(J, rhs, bc, dx, indices, N, ds, K):

    """
    units:
    phi:
        charge transport: [V]
        mass transport: [kg/m3]
        heat transport: [K]
    
    J, matrix of coefficients (Jacobian matrix): 
        charge transport: [S or A/V]
        mass transport: [m3/s]
        heat transport: [W/K]
    
    rhs, right hand side vector:
        charge transport: [A]
        mass transport: [kg/s]
        heat transport: [W]
    
    conductivity, K:
        charge transport: [S/m or A/V/m]
        mass transport: [m2/s]
        heat transport: [W/mK]

    flux, (Neumann boundary condition):
        charge transport: [A/m2]
        mass transport: [kg/m2/s]
        heat transport: [W/m2]
    """
    import numpy as np
    ip, jp, kp = indices

    # west side (i=0)
    i = 0
    if bc['West'][0] == 'Dirichlet':
        for n in np.array(np.where(ip==i)).T:
            j,k = jp[n], kp[n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            J[index, index] = -1*K*dx       # [W/K] [S or A/V]
            # aW[index], aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0,0
            rhs[index] = -bc['West'][1]*K*dx    # [W] [A] [mol/s]
    elif bc['West'][0] == 'Neumann':
        for n in np.array(np.where(ip==i)).T:
            j,k = jp[n], kp[n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            J[index, index] = -1*K*dx       # [W/K]
            J[index, index+N[1]*N[2]] = 1*K*dx   # aE[index] [W/K]
            # aW[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
            rhs[index] = -bc['West'][1]*dx**2    # [W]
    elif bc['West'][0] == 'Periodic':
        for n in np.array(np.where(ip==i)).T:
            j,k = jp[n], kp[n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            # anb[index] = [W/K]
            aW, aE, aS, aN, aB, aT = \
                K*ds[-2,j,k]*dx, K*ds[i+1,j,k]*dx, K*ds[i,j-1,k]*dx, \
                K*ds[i,j+1,k]*dx, K*ds[i,j,k-1]*dx, K*ds[i,j,k+1]*dx
            
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, (N[0]-2)*(N[1]*N[2]) + j*N[2] + k] = aW
            J[index, index+N[2]*N[1]] = aE
            J[index, index-N[2]] = aS
            J[index, index+N[2]] = aN
            J[index, index-1] = aB
            J[index, index+1] = aT
            J[index, index] = -(aW + aE + aS + aN + aB + aT)
            # rhs[index] = 0  # no need to set the rhs to zero, it is already zero

    # east side (i=N[0]-1)
    i = N[0]-1
    if bc['East'][0] == 'Dirichlet':
        for n in np.array(np.where(ip==i)).T:
            j,k = jp[n], kp[n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            J[index, index] = -1*K*dx       # [W/K]
            # aW[index], aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0,0
            rhs[index] = -bc['East'][1]*K*dx
    elif bc['East'][0] == 'Neumann':
        for n in np.array(np.where(ip==i)).T:
            j,k = jp[n], kp[n]
            if j==0 or j==N[1]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            J[index, index] = -1*K*dx    # [W/K]
            # aW[index] = 1
            J[index, index-N[1]*N[2]] = 1*K*dx   # aW[index]
            # aE[index], aN[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
            rhs[index] = -bc['East'][1]*dx**2   # [W]


    # south side (j=0)
    j = 0
    if bc['South'][0] == 'Dirichlet':
        for n in np.array(np.where(jp==j)).T:
            i,k = ip[n], kp[n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            J[index, index] = -1*K*dx
            # aW[index], aE[index], aN[index], aT[index], aB[index] = 0,0,0,0,0
            rhs[index] = -bc['South'][1]*K*dx
    elif bc['South'][0] == 'Neumann':
        for n in np.array(np.where(jp==j)).T:
            i,k = ip[n], kp[n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            J[index, index] = -1*K*dx
            # aN[index] = 1
            J[index, index+N[2]] = 1*K*dx   # aN[index]
            # aW[index], aE[index], aT[index], aB[index] = 0,0,0,0
            rhs[index] = -bc['South'][1]*dx**2  # [W]
    elif bc['South'][0] == 'Periodic':
        for n in np.array(np.where(jp==j)).T:
            i,k = ip[n], kp[n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            aW, aE, aS, aN, aB, aT = \
                K*ds[i-1,j,k]*dx, K*ds[i+1,j,k]*dx, K*ds[i,-2,k]*dx,\
                K*ds[i,j+1,k]*dx, K*ds[i,j,k-1]*dx, K*ds[i,j,k+1]*dx
            
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, index-N[2]*N[1]] = aW
            J[index, index+N[2]*N[1]] = aE
            J[index, i*(N[1]*N[2]) + (N[1]-2)*N[2] + k] = aS
            J[index, index+N[2]] = aN
            J[index, index-1] = aB
            J[index, index+1] = aT
            J[index, index] = -(aW + aE + aS + aN + aB + aT)
            # rhs[index] = 0  # no need to set the rhs to zero, it is already zero


    # north side (j=N[1]-1)
    j = N[1]-1
    if bc['North'][0] == 'Dirichlet':
        for n in np.array(np.where(jp==j)).T:
            i,k = ip[n], kp[n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            # aN[index] = 0
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, index] = -1*K*dx
            # aW[index], aE[index], aS[index], aT[index], aB[index] = 0,0,0,0,0
            rhs[index] = -bc['North'][1]*K*dx
    elif bc['North'][0] == 'Neumann':   
        for n in np.array(np.where(jp==j)).T:
            i,k = ip[n], kp[n]
            if i==0 or i==N[0]-1 or k==0 or k==N[2]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, index] = -1*K*dx
            # aS[index] = 1
            J[index, index-N[2]] = 1*K*dx   # aS[index]
            # aW[index], aE[index], aT[index], aB[index] = 0,0,0,0
            rhs[index] = -bc['North'][1]*dx**2  # [W]

    # bottom side (k=0)
    k = 0
    if bc['Bottom'][0] == 'Dirichlet':
        for n in np.array(np.where(kp==k)).T:
            i,j = ip[n], jp[n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            # aB[index] = 0
            J[index, index] = -1*K*dx
            # aW[index], aE[index], aS[index], aN[index], aT[index] = 0,0,0,0,0
            rhs[index] = -bc['Bottom'][1]*K*dx
    elif bc['Bottom'][0] == 'Neumann':
        for n in np.array(np.where(kp==k)).T:
            i,j = ip[n], jp[n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, index] = -1*K*dx
            # aT[index] = 1
            J[index, index+1] = 1*K*dx   # aT[index]
            # aW[index], aE[index], aS[index], aN[index] = 0,0,0,0
            rhs[index] = -bc['Bottom'][1]*dx**2  # [W]
    elif bc['Bottom'][0] == 'Periodic':
        for n in np.array(np.where(kp==k)).T:
            i,j = ip[n], jp[n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            aW, aE, aS, aN, aB, aT = \
                K*ds[i-1,j,k]*dx, K*ds[i+1,j,k]*dx, K*ds[i,j-1,k]*dx, \
                K*ds[i,j+1,k]*dx, K*ds[i,j,-2]*dx, K*ds[i,j,k+1]*dx

            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, index-N[2]*N[1]] = aW
            J[index, index+N[2]*N[1]] = aE
            J[index, index-N[2]] = aS
            J[index, index+N[2]] = aN
            J[index, i*(N[1]*N[2]) + j*N[2] + (N[2]-2)] = aB
            J[index, index+1] = aT
            J[index, index] = -(aW + aE + aS + aN + aB + aT)
            # rhs[index] = 0  # no need to set the rhs to zero, it is already zero

    # top side (k=N[2]-1)
    k = N[2]-1
    if bc['Top'][0] == 'Dirichlet':
        for n in np.array(np.where(kp==k)).T:
            i,j = ip[n], jp[n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            # aT[index] = 0
            J[index, index] = -1*K*dx
            # aW[index], aE[index], aS[index], aN[index], aB[index] = 0,0,0,0,0
            rhs[index] = -bc['Top'][1]*K*dx
    elif bc['Top'][0] == 'Neumann':
        for n in np.array(np.where(kp==k)).T:
            i,j = ip[n], jp[n]
            if i==0 or i==N[0]-1 or j==0 or j==N[1]-1: continue
            index = i*(N[1]*N[2]) + j*N[2] + k
            if len(J[index,:].data[0])!=0: # if the row is not empty
                raise Exception('The row is not empty')
            J[index, index] = -1*K*dx
            # aB[index] = 1
            J[index, index-1] = 1*K*dx   # aB[index]
            # aW[index], aE[index], aS[index], aN[index] = 0,0,0,0
            rhs[index] = -bc['Top'][1]*dx**2  # [W]

    return J, rhs

def plot_with_continuous_error(x, y, y_min, y_max, y_c_down=None, y_c_up=None, x_title='x', y_title='y', title=None):
    import plotly.graph_objects as go
    x = [x] if type(x) is not list else x 
    y = [y] if type(y) is not list else y
    y_min = [y_min] if type(y_min) is not list else y_min
    y_max = [y_max] if type(y_max) is not list else y_max
    y_c_down = [y_c_down] if type(y_c_down) is not list else y_c_down
    y_c_up = [y_c_up] if type(y_c_up) is not list else y_c_up

    # fig = go.Figure([
    #     go.Scatter(
    #         name='Upper Bound',
    #         x=x,
    #         y=y_max,
    #         mode='lines',
    #         marker=dict(color="#444"),
    #         line=dict(width=0),
    #         showlegend=False,
    #     ),
    #     go.Scatter(
    #         name='Lower Bound',
    #         x=x,
    #         y=y_min,
    #         marker=dict(color="#444"),
    #         line=dict(width=0),
    #         mode='lines',
    #         fillcolor='rgba(68, 68, 68, 0.3)',
    #         fill='tonexty',
    #         showlegend=False,
    #     )
    # ])
    fig = go.Figure()
    for i in range(len(y)):
        fig.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=x[i],
                y=y_max[i],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ))
        fig.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=x[i],
                y=y_min[i],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False,
            ))

        # add y curve
        if y is not None:
            fig.add_trace(go.Scatter(
                name='y',
                x=x[i],
                y=y[i],
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                showlegend=False,
            ))

        # add confidence level
        if y_c_down is not None:
            fig.add_trace(go.Scatter(
                name='Continuous Lower Bound',
                x=x[i],
                y=y_c_down[i],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                # fillcolor='rgba(68, 68, 68, 0.7)',
                # fill='tonexty',
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                name='Continuous Upper Bound',
                x=x[i],
                y=y_c_up[i],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.7)',
                fill='tonexty',
                showlegend=False,
            ))


    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title=x_title,
        title=title,
        hovermode=None,
    )
    
    fig.show()
    str = f'svg\\{title if title is not None else "fig"}.svg'
    fig.write_image(str)

def mean_confidence_interval(data, confidence=0.95):
    import numpy as np
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

