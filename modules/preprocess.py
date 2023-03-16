def sourcefunc_calc(inputs, TPB_dict):
    print("Source function calculations...", end='')
    from sympy import diff, exp, lambdify, log, simplify, symbols
    import numpy as np
    # Constants
    dx = inputs['microstructure']['dx']
    MH2 = 2.01588e-3        # Molar weight of hydrogen [kg/mol]
    MH2O = 18.01528e-3      # Molar weight of water [kg/mol]
    rhoH2 = 0.0251          # Density of hydrogen [kg/m^3]
    rhoH2O = 997            # Density of water [kg/m^3]
    R = 8.31446261815324    # Universal gas constant [J/mol/K]
    F = 96485.33289         # Faraday's constant [C/mol]
    aa = 0.5                # anode transfer coefficient []

    T = inputs['operating_conditions']['T']
    # I0a_l = inputs['I0a_l']

    # boundary conditions
    pH2_b = inputs['boundary_conditions']['pH2_b']
    Vel_b = inputs['boundary_conditions']['Vel_b']
    Vio_b = inputs['boundary_conditions']['Vio_b']
    pH2_inlet = inputs['boundary_conditions']['pH2_inlet']


    # other variables
    pH2O_b = inputs['operating_conditions']['P'] - pH2_b          # partial pressure of H2O [atm]
    pH2O_inlet = inputs['operating_conditions']['P'] - pH2_inlet  # partial pressure of H2O @ fuel cell inlet [atm]
    ptot = inputs['operating_conditions']['P']       # total pressure [atm]
    
    xH2_b = pH2_b/ptot          # mole fraction of hydrogen @ anode boundary, bulk []
    xH2_inlet = pH2_inlet/ptot  # mole fraction of hydrogen @ fuel cell inlet []
    xH2O_b = pH2O_b/ptot        # mole fraction of water @ anode boundary, bulk []
    xH2O_inlet = pH2O_inlet/ptot    # mole fraction of water @ fuel cell inlet []
    
    cH2_b = rhoH2*xH2_b         # Hydrogen concentration @ anode boundary, bulk [kg/m^3] - boundary condition

    # symbolic expressions
    cH2 = symbols('cH2')    # hydrogen mass fraction [pores]
    Vel = symbols('Vel')    # electron potential [Nickel]
    Vio = symbols('Vio')    # ion potential [YSZ]
    vars = [cH2, Vel, Vio]

    xH2 = cH2/rhoH2             # mole fraction of hydrogen, symbolic []
    pH2 = ptot*xH2         # partial pressure of hydrogen, symbolic [atm]
    pH2O = ptot - pH2           # partial pressure of water, symbolic [atm]
    xH2O = pH2O/ptot       # mole fraction of water, symbolic []
    
    # ??? Exchange current per TPB length, anode [A/m], Prokop et al. 2018 
    # This equation is not correct right now. I'm not sure what variables should be used instead
    # of xH2 and xH2O. This equation is apprarently extrapolated from Boer 1998. 
    # However, in Boer's thesis, this equation is not mentioned explicitly. 
    # There are experimental results that are used to fit the equation, but it's not 
    # clear what variables are used in the equation. 
    # I am confused if these two variables should be:
    # 1- mole fractions, or
    # 2- normalized mole fractions, or
    # 3- partial pressures.
    # If normalized mole fractions should be used, actual mole fractions (xH2, and xH2O) should 
    # be devided by the standard mole fraction. 
    # The unit for this equation is not clear. It is not clear if it is [A/m] or [A/cm] or anything similar.
    # by comparing this equation with other values reported elsewhere, I think that the unit is [A/m].
    I0a_l = 31.4 * (pH2*101325)**(-0.03) * (pH2O*101325)**(0.4) * np.exp(-152155/R/T)  
    # I0a_l = 100 * 2.14e-10 * 1e6      # from Shearing et al. 2010 for T=900 C [A/m]  (for test purposes)

    # The way that lineal exchange current density is transformed to volumetric exchange
    # current density is not clear. Two possible conversion factors can be used. 
    # Is the conversion factor correct?
    # conversion factor 1 is larger than conversion factor 2.
    conversion_fac_1 = dx / dx**3       # [m/m3]
    conversion_fac_2 = TPB_dict['TPB_density']      # [m/m3]
    I0a = I0a_l*conversion_fac_1      # Exchange current density, anode [A/m^3]

    # Tseronis et al. 2012 model for anode current density
    if inputs['solver_options']['ion_only']:
        eta_a_con = R*T/2/F*log(xH2_inlet/xH2_b*xH2O_b/xH2O_inlet)         # anode concentration overpotential [V]
        if eta_a_con > (Vel_b - Vio_b):
            raise ValueError('Concentration overpotential is greater than the sum of the electrode overpotentials.')
    else:
        eta_a_con = R*T/2/F*log(xH2_inlet/xH2*xH2O/xH2O_inlet)         # anode concentration overpotential [V]
        

    eta_a_act = Vel - Vio - eta_a_con                      # anode activation overpotential [V]
    Ia = I0a*(exp(2*aa*F*eta_a_act/R/T)
             -exp(-2*(1-aa)*F*eta_a_act/R/T))               # anode current density [A/m^3]

    # Shearing et al. 2010 model for anode current density
    # Voe = 1.23 - 2.304e-4*(T-298.15)        # standard potential [V]
    # Veq = Voe - R*T/2/F*log(nH2O/nH2)      # equilibrium potential [V]
    # eta = Veq - Vio + Vel                   # overpotential [V]
    # Ia = I0a*(exp(2*aa*F*eta/R/T)
    #          -exp(-2*(1-aa)*F*eta/R/T))               # ??? anode current density [A/m^3]

    # initialize the source function list
    source_func = [None]*3         # initialization
    source_func[0] = simplify(-Ia/2/F*MH2) if inputs['solver_options']['ion_only'] is False else None # mass [kg/m3/s]
    source_func[1] = simplify(-Ia) if inputs['solver_options']['ion_only'] is False else None            # electron [A/m3]
    source_func[2] = simplify(Ia)           # ion [A/m3]
    expected_sign = [-1,-1,1]             # expected sign of the source terms
    # source_func[0] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[1] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[2] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case

    # source term treatment
    f = [None]*3
    fp = [None]*3
    for p in [0,1,2]:
        if inputs['solver_options']['ion_only'] and p!=2:
            continue
        f[p] = lambdify(vars, source_func[p]) #[kg/m3/s] or [A/m3]
        fp[p] = lambdify(vars, diff(source_func[p], vars[p]))

    # Threshold values
    cH2_min = 1e-7             # minimum concentration [kg/m^3]
    cH2_max = cH2_b         # maximum concentration [kg/m^3]
    Vel_min = -2             # minimum electronic potential [V]
    Vel_max = Vel_b             # maximum electronic potential [V]
    Vio_min = Vio_b             # minimum ionic potential [V]
    Vio_max = 2             # maximum ionic potential [V]
    thd = [[cH2_min,cH2_max], [Vel_min,Vel_max], [Vio_min,Vio_max]]


    field_functions = {
        'f': f,
        'fp': fp,
        'signs': expected_sign,
        'eta_act': lambdify([cH2, Vel, Vio], eta_a_act),
        'eta_con': lambdify([cH2], eta_a_con),
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
    'West':   ['Dirichlet', Vel_b],         
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
    'East':   ['Dirichlet', Vio_b],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]}]

    bc_dict = bc

    print("Done!")
    return field_functions, thd, bc_dict

def get_indices_all(inputs, domain, TPB_dict):
    print('Identifying neighboring cells and obtaining indices...', end=' ')
    import numpy as np
    TPB_mask = TPB_dict['TPB_mask']
    ds = [None]*3
    indices = [None]*3
    domain -= 1
    ds = [
        np.zeros(domain.shape, dtype=bool), 
        np.zeros(domain.shape, dtype=bool), 
        np.zeros(domain.shape, dtype=bool)]
    
    for p in [0,1,2]: 
        ds[p][domain==p] = True
        # if inputs['solver_options']['ion_only'] and p!=2:
        #     continue
        indices[p] = get_indices(inputs, domain, TPB_mask, ds[p], p)

    masks_dict = {
        'ds': ds, 
        'ds_lin': [],     
        'ds_lin_s': []}
    print('Done!')
    return masks_dict, indices

def get_indices(inputs, domain, TPB_mask_old, ds, phase):
    import numpy as np
    import concurrent.futures 
    # import multiprocessing as mp
    from time import time

    N = domain.shape
    
    TPB_mask = np.copy(TPB_mask_old)
    TPB_mask[np.logical_and(ds==False, TPB_mask_old)] = False
    N_sources = np.sum(TPB_mask[1:-1,1:-1,1:-1])

    # allocating the neighboring arrays (aP, aW, aE, aS, aN, aB, aT)
    ip1, jp1, kp1 = np.where(ds)    # locations of points in the solver domain
    ind_stack = np.stack((ip1, jp1, kp1), axis=1) # stack the indicess
    L = len(ip1)  # total number of points in diagonal of A (matrix of coefficients)
    
    # identifying elements who has a neighbor in the solver domain
    ip1_E, jp1_E, kp1_E = np.where(np.logical_and(ds[:-1,:,:],ds[1:,:,:]))  # has east neighbor
    ip1_W, jp1_W, kp1_W = ip1_E+1, jp1_E, kp1_E
    ip1_N, jp1_N, kp1_N = np.where(np.logical_and(ds[:,:-1,:],ds[:,1:,:]))  # has north neighbor
    ip1_S, jp1_S, kp1_S = ip1_N, jp1_N+1, kp1_N
    ip1_T, jp1_T, kp1_T = np.where(np.logical_and(ds[:,:,:-1],ds[:,:,1:]))  # has top neighbor
    ip1_B, jp1_B, kp1_B = ip1_T, jp1_T, kp1_T+1

    ind_east_stack = np.stack((ip1_E, jp1_E, kp1_E), axis=1)    # has east neighbor
    ind_west_stack = np.stack((ip1_W, jp1_W, kp1_W), axis=1)    # has west neighbor
    ind_north_stack = np.stack((ip1_N, jp1_N, kp1_N), axis=1)   # has north neighbor
    ind_south_stack = np.stack((ip1_S, jp1_S, kp1_S), axis=1)   # has south neighbor
    ind_top_stack = np.stack((ip1_T, jp1_T, kp1_T), axis=1)     # has top neighbor
    ind_bottom_stack = np.stack((ip1_B, jp1_B, kp1_B), axis=1)  # has bottom neighbor
    
    flag_west, flag_east, flag_south, flag_north, flag_bottom, flag_top = np.zeros(shape=(6, L), dtype=int)-1
    
    if not(inputs['solver_options']['ion_only'] and phase!=2): 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            WE_res = executor.submit(get_flags, ind_west_stack, ind_east_stack, ind_stack, L)
            SN_res = executor.submit(get_flags, ind_south_stack, ind_north_stack, ind_stack, L)
            BT_res = executor.submit(get_flags, ind_bottom_stack, ind_top_stack, ind_stack, L)

            flag_west, flag_east = WE_res.result()
            flag_south, flag_north = SN_res.result()
            flag_bottom, flag_top = BT_res.result()
    
    # # identifying the west-east pairs of elements 
    # a = 0
    # b = []
    # for n in range(len(ind_west_stack)):
    #     if inputs['solver_options']['ion_only'] and phase!=2: break
    #     while np.any(ind_east_stack[n,:] != ind_stack[a,:]):
    #         a += 1
    #     b = b+1 if b!=[] else a+1
    #     while np.any(ind_west_stack[n,:] != ind_stack[b,:]):
    #         b += 1
    #     flag_west[b] = a
    #     flag_east[a] = b

    # # identifying the south-north pairs of elements
    # a = 0
    # b = []
    # for n in range(len(ind_south_stack)):
    #     if inputs['solver_options']['ion_only'] and phase!=2: break
    #     while np.any(ind_north_stack[n,:] != ind_stack[a,:]):
    #         a += 1
    #     b = b+1 if b!=[] else a+1
    #     while np.any(ind_south_stack[n,:] != ind_stack[b,:]):
    #         b += 1
    #     flag_south[b] = a
    #     flag_north[a] = b

    # # identifying the bottom-top pairs of elements
    # a = 0
    # b = []
    # for n in range(len(ind_bottom_stack)):
    #     if inputs['solver_options']['ion_only'] and phase!=2: break
    #     while np.any(ind_top_stack[n,:] != ind_stack[a,:]):
    #         a += 1
    #     b = b+1 if b!=[] else a+1
    #     while np.any(ind_bottom_stack[n,:] != ind_stack[b,:]):
    #         b += 1
    #     flag_bottom[b] = a
    #     flag_top[a] = b
    
    # if phase==2: print('Time for get_flags (serial): ', time()-t)
    # print('dummy line')
    # another method to find the indices of the neighbors [yields the same result but enifficient when N is large]
    # flag_west2, flag_east2, flag_south2, flag_north2, flag_bottom2, flag_top2 = np.zeros(shape=(6, L), dtype=int)-1
    # t = time.time()
    # for n in range(L):
    #     # os.system('cls')
    #     print('processing point {}/{}'.format(n+1, L))
    #     i,j,k = ind_stack[n,:]
        
    #     a = np.where(np.all(np.equal([i-1,j,k], ind_stack[:n]), axis=1))[0]
    #     flag_west2[n] = a if len(a) else -1
    #     flag_east2[a] = n
        
    #     a = np.where(np.all(np.equal([i,j-1,k], ind_stack[:n]), axis=1))[0]
    #     flag_south2[n] = a if len(a) else -1
    #     flag_north2[a] = n

    #     a = np.all([i,j,k-1] == ind_stack[n-1])
    #     flag_bottom2[n] = n-1 if a else -1
    #     flag_top2[n-1] = n if a else -1
    

    # Indices of west, east, south, north, bottom, top boundaries
    ind_west_bound = np.where(ip1 == 0)[0]
    ind_east_bound = np.where(ip1 == N[0]-1)[0]
    ind_south_bound = np.where(jp1 == 0)[0]
    ind_north_bound = np.where(jp1 == N[1]-1)[0]
    ind_bottom_bound = np.where(kp1 == 0)[0]
    ind_top_bound = np.where(kp1 == N[2]-1)[0]

    # Indices of all boundaries
    ind_allbounds = np.where(np.logical_or.reduce((
        ip1==0, 
        ip1==N[0]-1, 
        jp1==0, 
        jp1==N[1]-1, 
        kp1==0, 
        kp1==N[2]-1
        )))[0]

    # Indices of all interior points
    ind_interior = np.delete(np.arange(L),ind_allbounds)

    # Indices of all interior points inside and outside the TPB
    ind_source = np.zeros(shape = N_sources, dtype=int)
    ind_notsource = np.zeros(shape = L-N_sources-len(ind_allbounds), dtype=int)
    i_source = 0
    i_notsource = 0

    for n in ind_interior:
        if TPB_mask[ip1[n],jp1[n],kp1[n]]:
            # consider replacing with append (idk if append is faster or not)
            ind_source[i_source] = n
            i_source += 1
        else:
            ind_notsource[i_notsource] = n
            i_notsource += 1

    indices_dict = {
        'all_points': ind_stack,
        
        'west_bound': ind_west_bound,
        'east_bound': ind_east_bound,
        'south_bound': ind_south_bound,
        'north_bound': ind_north_bound,
        'bottom_bound': ind_bottom_bound,
        'top_bound': ind_top_bound,
        
        'all_bounds': ind_allbounds,
        'interior': ind_interior,
        
        'source': ind_source,
        'notsource': ind_notsource,

        'west_nb': flag_west,
        'east_nb': flag_east,
        'south_nb': flag_south,
        'north_nb': flag_north,
        'bottom_nb': flag_bottom,
        'top_nb': flag_top,
        }

    return indices_dict

def create_SOLE_individual(inputs, bc_dict, indices, masks_dict):
    print('Writing Jacobian and rhs matrix...', end=' ')
    import numpy as np
    cond_H2 = 2.17e6        # [m^2/s]
    cond_el = 3.27e6 - 1065.3 * inputs['operating_conditions']['T']       # [S/m]
    cond_ion = 3.34e4 * np.exp(-10350/inputs['operating_conditions']['T'])    # [S/m]
    K = [cond_H2, cond_el, cond_ion] 

    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]

    dx = inputs['microstructure']['dx']

    J = [None]*3
    rhs = [None]*3
    sum_nb = [None]*3
    bc = bc_dict
    ds = masks_dict['ds']
    isMi = inputs['is_multiple_instances']
    M_ins = inputs['M_instances']
    scaling_factor = inputs['scaling_factor']

    for p in [0,1,2]: # only solve the ion phase, otherwise: for p in [0,1,2]:
        if inputs['solver_options']['ion_only'] and p!=2:
            continue
        J[p], rhs[p] = boundaries_individual(K[p], dx, bc[p], indices[p], N, isMi, M_ins, scaling_factor)
        J[p], sum_nb[p] = interior_individual(J[p], indices[p], K[p], ds[p], dx, bc[p], isMi, M_ins, scaling_factor)

    print('Done!')
    return J, rhs, sum_nb

def boundaries_individual(K, dx, bc, indices, N, isMi, M_ins=None, scaling_factor=None):
    import numpy as np
    from scipy.sparse import lil_matrix

    # initializing left hand side and right hand side of the system of equaitons
    L = len(indices['all_points'])
    rhs = np.zeros(shape = L, dtype = float) # right hand side vector

    # rhs = lil_matrix((1,L), dtype = float)
    J = lil_matrix((L,L), dtype = float) # sparse matrix
    if isMi:
        j_seg = [N[1]//M_ins*i for i in range(M_ins)]
        j_seg.append(N[1])
        j_seg = np.array(j_seg)
        # j_seg[1:] -= 1

    # west side (i=0)
    if bc['West'][0] == 'Dirichlet':
        for n in indices['west_bound']:
            _,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K*dx     # aP
            rhs[n] = bc['West'][1]*sf*K*dx


    # east side (i=N[0]-1)
    if bc['East'][0] == 'Dirichlet':
        for n in indices['east_bound']:
            _,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K*dx     # aP
            rhs[n] = bc['East'][1]*sf*K*dx


    # south side (j=0)
    if bc['South'][0] == 'Dirichlet':
        for n in indices['south_bound']:
            J[n,n] = 1*K*dx    # aP
            rhs[n] = bc['South'][1]*K*dx


    # north side (j=N[1]-1)
    if bc['North'][0] == 'Dirichlet':
        for n in indices['north_bound']:
            J[n,n] = 1*K*dx     # aP
            rhs[n] = bc['North'][1]*K*dx


    # bottom side (k=0)
    if bc['Bottom'][0] == 'Dirichlet':
        for n in indices['bottom_bound']:
            _,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K*dx   # aP
            rhs[n] = bc['Bottom'][1]*sf*K*dx


    # top side (k=N[2]-1)
    if bc['Top'][0] == 'Dirichlet':
        for n in indices['top_bound']:
            _,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K*dx  # aP
            rhs[n] = bc['Top'][1]*sf*K*dx


    return J, rhs

def interior_individual_obsolete(J, indices, K, ds, dx, M_instances=None, scaling_factor=None):
    import numpy as np
    L = len(indices['all_points'])
    sum_nb = np.zeros(shape = L, dtype = float) # sigma a_nb vector

    for n in indices['interior']:
        i,j,k = indices['all_points'][n]

        # assign a_nb values for all interior points 
        aW = -K*ds[i-1,j,k]*dx 
        aE = -K*ds[i+1,j,k]*dx
        aS = -K*ds[i,j-1,k]*dx
        aN = -K*ds[i,j+1,k]*dx
        aB = -K*ds[i,j,k-1]*dx
        aT = -K*ds[i,j,k+1]*dx

        if ds[i-1,j,k]: J[n,indices['west_nb'][n]]   = aW
        if ds[i+1,j,k]: J[n,indices['east_nb'][n]]   = aE
        if ds[i,j-1,k]: J[n,indices['south_nb'][n]]  = aS
        if ds[i,j+1,k]: J[n,indices['north_nb'][n]]  = aN
        if ds[i,j,k-1]: J[n,indices['bottom_nb'][n]] = aB
        if ds[i,j,k+1]: J[n,indices['top_nb'][n]]    = aT
        
        sum_nb[n] = aW + aE + aS + aN + aB + aT
        J[n,n] = -sum_nb[n]
        
    if M_instances is not None:
        j_max = np.max(indices['all_points'][:,1])
        j_seg = j_max//M_instances+1
        for n in indices['interior']:
            i,j,k = indices['all_points'][n]
            if np.mod(j+1, j_seg)==0 and j!=0 and j!=j_max:
                aS_old = J[n,indices['south_nb'][n]]
                aS_new = aS_old * scaling_factor
                sum_nb[n] = sum_nb[n] - aS_old + aS_new
                J[n,indices['south_nb'][n]] = aS_new
                J[n,n] = -sum_nb[n]
            if np.mod(j+1, j_seg)==j_seg-1 and j!=0 and j!=j_max:
                aN_old = J[n,indices['north_nb'][n]]
                aN_new = aN_old * 1/scaling_factor
                sum_nb[n] = sum_nb[n] - aN_old + aN_new
                J[n,indices['north_nb'][n]] = aN_new
                J[n,n] = -sum_nb[n]
    
    return J, sum_nb

def interior_individual(J, indices, K, ds, dx, bc, isMi, M_ins=None, scaling_factor=None):
    import numpy as np
    L = len(indices['all_points'])
    sum_nb = np.zeros(shape = L, dtype = float) # sigma a_nb vector
    i_max = np.max(indices['all_points'][:,0])
    j_max = np.max(indices['all_points'][:,1])
    k_max = np.max(indices['all_points'][:,2])

    for n in range(len(indices['all_points'])):
        i,j,k = indices['all_points'][n]

        # skip Dirichlet boundary points
        if bc['West'][0]   == 'Dirichlet' and i==0:        continue
        if bc['East'][0]   == 'Dirichlet' and i==i_max:    continue
        if bc['South'][0]  == 'Dirichlet' and j==0:        continue
        if bc['North'][0]  == 'Dirichlet' and j==j_max:    continue
        if bc['Bottom'][0] == 'Dirichlet' and k==0:        continue
        if bc['Top'][0]    == 'Dirichlet' and k==k_max:    continue

        # assign a_nb values for all points (including Neumann boundary points)
        aW = -K*ds[i-1,j,k]*dx if i!=0     else 0  
        aE = -K*ds[i+1,j,k]*dx if i!=i_max else 0    
        aS = -K*ds[i,j-1,k]*dx if j!=0     else 0 
        aN = -K*ds[i,j+1,k]*dx if j!=j_max else 0    
        aB = -K*ds[i,j,k-1]*dx if k!=0     else 0 
        aT = -K*ds[i,j,k+1]*dx if k!=k_max else 0    

        if i!=0     and ds[i-1,j,k]: J[n,indices['west_nb'][n]]   = aW
        if i!=i_max and ds[i+1,j,k]: J[n,indices['east_nb'][n]]   = aE
        if j!=0     and ds[i,j-1,k]: J[n,indices['south_nb'][n]]  = aS
        if j!=j_max and ds[i,j+1,k]: J[n,indices['north_nb'][n]]  = aN
        if k!=0     and ds[i,j,k-1]: J[n,indices['bottom_nb'][n]] = aB
        if k!=k_max and ds[i,j,k+1]: J[n,indices['top_nb'][n]]    = aT
        
        sum_nb[n] = aW + aE + aS + aN + aB + aT
        J[n,n] = -sum_nb[n]
        
    if isMi:
        j_max = np.max(indices['all_points'][:,1])
        j_seg = j_max//M_ins+1
        for n in indices['interior']:
            i,j,k = indices['all_points'][n]
            if np.mod(j, j_seg)==j_seg-1 and j!=0 and j!=j_max:     #lower interface
                aN_old = J[n,indices['north_nb'][n]]
                aN_new = aN_old * scaling_factor
                sum_nb[n] = sum_nb[n] - aN_old + aN_new
                J[n,indices['north_nb'][n]] = aN_new
                J[n,n] = -sum_nb[n]
            if np.mod(j, j_seg)==0 and j!=0 and j!=j_max:           #upper interface
                aS_old = J[n,indices['south_nb'][n]]
                aS_new = aS_old * 1/scaling_factor
                sum_nb[n] = sum_nb[n] - aS_old + aS_new
                J[n,indices['south_nb'][n]] = aS_new
                J[n,n] = -sum_nb[n]
    
    return J, sum_nb

def initilize_field_variables_individual(inputs, masks_dict, indices, isMi, M_instances = None, scaling_factor = None):
    # initial guess
    import numpy as np
    print('Initializing field variables...', end = ' ')
    residuals = [[]]*3
    phi = [[]]*3
    N_y = inputs['microstructure']['Ny']
    ds = masks_dict['ds']

    rhoH2 = 0.0251       # Density of hydrogen [kg/m^3]
    cH2_b = inputs['boundary_conditions']['pH2_b'] / inputs['operating_conditions']['P'] * rhoH2

    init_cond = [
        cH2_b, 
        inputs['boundary_conditions']['Vel_b'], 
        inputs['boundary_conditions']['Vio_b']]

    # if M_instances is None:
    for p in [0,1,2]:
        # L = len(np.where(ds[p]))
        L = np.sum(ds[p])
        phi[p] = np.ones(shape = L, dtype = float) * init_cond[p]
        
    if isMi:
        j_seg = [N_y//M_instances*i for i in range(M_instances)]
        j_seg.append(N_y)
        j_seg = np.array(j_seg)
        
        for p in [0,1,2]:
            for n in range(len(indices[p]['all_points'])):
                _,j,_ = indices[p]['all_points'][n]
                sf = scaling_factor**(np.sum(j>=j_seg)-1)
                phi[p][n] = init_cond[p]*sf

    print('Done!')
    return phi, residuals

# specific functions for the entire cell
def sourcefunc_calc_entire_cell(inputs, TPB_dict):
    """
    Calculates the source function for the entire cell.
    phases:
    0: anode - pores
    1: anode - Ni
    2: anode, and electrolyte: YSZ
    3: cathode - pores
    4
    """
    print("Source function calculations...", end='')
    from sympy import diff, exp, lambdify, log, simplify, symbols
    import numpy as np
    # Constants
    MH2 = 2.01588e-3        # Molar weight of hydrogen [kg/mol]
    MH2O = 18.01528e-3      # Molar weight of water [kg/mol]
    MO2 = 31.9988e-3        # Molar weight of oxygen [kg/mol]
    MN2 = 28.0134e-3        # Molar weight of nitrogen [kg/mol]
    rhoH2 = 0.0251          # Density of hydrogen [kg/m^3]
    rhoH2O = 997            # Density of water [kg/m^3]
    rhoO2 = 1.429           # Density of oxygen [kg/m^3]
    rhoN2 = 1.250           # Density of nitrogen [kg/m^3]
    R = 8.31446261815324    # Universal gas constant [J/mol/K]
    F = 96485.33289         # Faraday's constant [C/mol]
    aa = 0.5                # anode transfer coefficient []
    ac = 0.5                # cathode transfer coefficient []

    T = inputs['T']
    # I0a_l = inputs['I0a_l']

    # boundary conditions
    pH2_b = inputs['pH2_in']
    pO2_b = inputs['pO2_in']

    # other variables
    pH2O_b = inputs['P'] - pH2_b          # partial pressure of H2O [atm]
    pN2_b = inputs['P'] - pO2_b           # partial pressure of N2 [atm]
    ptot = inputs['P']       # total pressure [atm]
    nH2_b = pH2_b/ptot          # mole fraction of hydrogen @ anode boundary, bulk []
    nH2O_b = pH2O_b/ptot        # mole fraction of water @ anode boundary, bulk []
    nO2_b = pO2_b/ptot          # mole fraction of oxygen @ cathode boundary, bulk []
    nN2_b = pN2_b/ptot          # mole fraction of nitrogen @ cathode boundary, bulk []
    cH2_b = rhoH2*nH2_b         # Hydrogen concentration @ anode boundary, bulk [kg/m^3] - boundary condition
    cH2O_b = rhoH2O*nH2O_b      # Water concentration @ anode boundary, bulk [kg/m^3]
    cO2_b = rhoO2*nO2_b         # Oxygen concentration @ cathode boundary, bulk [kg/m^3] - boundary condition
    cN2_b = rhoN2*nN2_b         # Nitrogen concentration @ cathode boundary, bulk [kg/m^3]

    # symbolic expressions
    cH2 = symbols('cH2')    # hydrogen mass fraction [pores - anode]
    Vel_a = symbols('Vel_a')    # electron potential [Nickel - anode]
    Vio = symbols('Vio')    # ion potential [YSZ - anode, electrolyte, cathode]
    cO2 = symbols('cO2')    # oxygen mass fraction [pores - cathode]
    Vel_c = symbols('Vel_c')    # electron potential [LSM - cathode]

    nH2 = cH2/rhoH2             # mole fraction of hydrogen, symbolic []
    nO2 = cO2/rhoO2             # mole fraction of oxygen, symbolic []
    pH2 = ptot*nH2         # partial pressure of hydrogen, symbolic [atm]
    pO2 = ptot*nO2         # partial pressure of oxygen, symbolic [atm]
    pH2O = ptot - pH2           # partial pressure of water, symbolic [atm]
    pN2 = ptot - pO2            # partial pressure of nitrogen, symbolic [atm]
    
    I0a_l = 31.4 * (pH2*1e5)**(-0.03) * (pH2O*1e5)**(0.4) * np.exp(-152155/R/T)     # Exchange current per TPB length, anode [A/m], Prokop et al. 2018
    I0a = I0a_l*TPB_dict['anode']['TPB_density']      # Exchange current density, anode [A/m^3]
    
    I0c_l = np.exp(-152155/R/T)  #??? [A/m]
    I0c_TPB = I0c_l*TPB_dict['cathode']['TPB_density']    # Exchange current density, cathode [A/m^3]
    
    I0c_area = np.exp(-152155/R/T) #??? [A/m^2]
    I0c_ISA = I0c_area*TPB_dict['cathode']['ISA_density']  # Exchange current density, cathode [A/m^3]

    nH2O = pH2O/ptot       # mole fraction of water, symbolic []
    nN2 = pN2/ptot         # mole fraction of nitrogen, symbolic []

    # Tseronis et al. 2012 model for anode current density
    eta_conc_a = R*T/2/F*log(nH2_b/nH2*nH2O/nH2O_b)         # ??? (make sure, reverse?) anode concentration overpotential [V]
    eta_conc_c = R*T/4/F*log(nO2_b/nO2)          # ??? (make sure [Nitrogen]??, reverse?) cathode concentration overpotential [V]
    eta_act_a = Vel_a - Vio - eta_conc_a                      # ??? anode activation overpotential [V]
    eta_act_c = Vio - Vel_c - eta_conc_c                      # ??? cathode activation overpotential [V]
    Ia = I0a*(exp(2*aa*F*eta_act_a/R/T)
            -exp(-2*(1-aa)*F*eta_act_a/R/T))               # ??? anode current density [A/m^3]
    Ic_TPB = I0c_TPB*(exp(2*ac*F*eta_act_c/R/T)
            -exp(-2*(1-ac)*F*eta_act_c/R/T))               # ??? cathode current density [A/m^3]
    Ic_ISA = I0c_ISA*(exp(2*ac*F*eta_act_c/R/T)
            -exp(-2*(1-ac)*F*eta_act_c/R/T))               # ??? cathode current density [A/m^3]
    
    # Shearing et al. 2010 model for anode current density
    # Voe = 1.23 - 2.304e-4*(T-298.15)        # standard potential [V]
    # Veq = Voe - R*T/2/F*log(nH2O/nH2)      # equilibrium potential [V]
    # eta = Veq - Vio + Vel                   # overpotential [V]
    # Ia = I0a*(exp(2*aa*F*eta/R/T)
    #          -exp(-2*(1-aa)*F*eta/R/T))               # ??? anode current density [A/m^3]

    # initialize the source function list
    source_func = [None]*6         # initialization
    source_func[0] = simplify(-Ia/2/F*MH2)   # mass [kg/m3/s] - anode
    source_func[1] = simplify(-Ia)            # electron [A/m3] - anode
    source_func[2] = simplify(Ia)           # ion [A/m3] - anode
    source_func[3] = simplify(-Ic_TPB/4/F*MO2)   # mass [kg/m3/s] - cathode
    source_func[4] = simplify(Ic_TPB)            # electron [A/m3] - [plus or minus sign??] cathode
    source_func[5] = simplify(-Ic_TPB)           # ion [A/m3] - cathode

    expected_sign = [-1,-1,1, -1,-1,1]             # ???expected sign of the source terms
    # source_func[0] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[1] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[2] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case

    # source term treatment
    f = [None]*6
    f[0] = lambdify([cH2, Vel_a, Vio], source_func[0])    #[kg/m3/s]
    f[1] = lambdify([cH2, Vel_a, Vio], source_func[1])    #[A/m3]
    f[2] = lambdify([cH2, Vel_a, Vio], source_func[2])    #[A/m3]
    f[3] = lambdify([cO2, Vel_c, Vio], source_func[3])    #[kg/m3/s]
    f[4] = lambdify([cO2, Vel_c, Vio], source_func[4])    #[A/m3]
    f[5] = lambdify([cO2, Vel_c, Vio], source_func[5])    #[A/m3]

    fp = [None]*6
    fp[0] = lambdify([cH2, Vel_a, Vio], diff(source_func[0], cH2))    #[1/s]
    fp[1] = lambdify([cH2, Vel_a, Vio], diff(source_func[1], Vel_a))    #[A/V/m3]
    fp[2] = lambdify([cH2, Vel_a, Vio], diff(source_func[2], Vio))    #[A/V/m3]
    fp[3] = lambdify([cO2, Vel_c, Vio], diff(source_func[3], cO2))    #[1/s]
    fp[4] = lambdify([cO2, Vel_c, Vio], diff(source_func[4], Vel_c))    #[A/V/m3]
    fp[5] = lambdify([cO2, Vel_c, Vio], diff(source_func[5], Vio))    #[A/V/m3]

    # Threshold values
    # cH2_min = 1e-7             # minimum concentration [kg/m^3]
    # cH2_max = cH2_b         # maximum concentration [kg/m^3]
    # Vel_min = 0             # minimum electronic potential [V]
    # Vel_max = inputs['Vel_in']             # maximum electronic potential [V]
    # Vio_min = inputs['Vio_in']             # minimum ionic potential [V]
    # Vio_max = 3             # maximum ionic potential [V]
    # thd = [[cH2_min,cH2_max], [Vel_min,Vel_max], [Vio_min,Vio_max]]


    field_functions = {
        'f': f,
        'fp': fp,
        'signs': expected_sign,
        'eta_act_a': lambdify([cH2, Vel_a, Vio], eta_act_a),
        'eta_conc_a': lambdify([cH2], eta_conc_a),
        'eta_act_c': lambdify([cO2, Vel_c, Vio], eta_act_c),
        'eta_conc_c': lambdify([cO2], eta_conc_c),
        'Ia': lambdify([cH2, Vel_a, Vio], Ia),
        'Ic': lambdify([cO2, Vel_c, Vio], Ic_TPB)}

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
    
    # the first phase [pores anode] ??
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
    'West':   ['Dirichlet', inputs['Vel_anode']],         
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
    # cathode side
    'East':   ['Neumann', 0],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{
    # the fourth phase [pores - cathode] ??
    # electrolyte side
    'West':   ['Neumann', 0],           
    # cathode side
    'East':   ['Dirichlet', cO2_b],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{
    # the fifth phase [LSM - ion] ??
    # electrolyte side
    'West':   ['Neumann', 0],           
    # cathode side
    'East':   ['Dirichlet', inputs['Vel_cathode']],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    }]

    bc_dict = bc

    print("Done!")
    return field_functions, bc_dict

def get_indices_entire_cell(inputs, domain_entire, TPB_dict):
    import numpy as np

    print('Identifying neighbors and getting indices...', end=' ')
    domain_a = domain_entire[:inputs['Nx_a'],:,:]
    domain_c = domain_entire[-inputs['Nx_c']:,:,:]

    TPB_mask_a = TPB_dict['anode']['TPB_mask']
    TPB_mask_c = TPB_dict['cathode']['TPB_mask']
    TPB_mask_YSZ = np.concatenate((TPB_mask_a, np.zeros((inputs['Nx_e'],inputs['Ny'], inputs['Nz']), dtype=bool), TPB_mask_c), axis=0)
    ds = [None]*5
    indices = [None]*5
    domain_entire -= 1
    for p in [0,1,2,3,4]: # 1,4 are not needed - electron transport assumed to be spontaneous
        if p == 0 or p == 1:
            domain = domain_a
            TPB_mask = TPB_mask_a
        elif p == 2:
            domain = domain_entire
            TPB_mask = TPB_mask_YSZ
        elif p == 3 or p == 4:
            domain = domain_c
            TPB_mask = TPB_mask_c

        ds[p], indices[p] = get_indices(domain, p, TPB_mask)

    masks_dict = {
        'ds': ds, 
        'ds_lin': [],     
        'ds_lin_s': []}
    print('Done!')
    return masks_dict, indices

def create_SOLE_individual_entire_cell(inputs, bc_dict, indices, masks_dict):
    print('Writing Jacobian and rhs matrix...', end=' ')
    K = [inputs['K_pores_a'], inputs['K_Ni'], inputs['K_YSZ'], inputs['K_pores_c'], inputs['K_LSM']]
    N_a = [inputs['Nx_a'], inputs['Ny'], inputs['Nz']]
    N_c = [inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    N_YSZ = [inputs['Nx_a']+inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    dx = inputs['dx']
    ds = masks_dict['ds']
    J = [None]*5
    rhs = [None]*5
    sum_nb = [None]*5
    bc = bc_dict

    isMi = inputs['is_multiple_instances']
    M_ins = inputs['M_instances']
    scaling_factor = inputs['scaling_factor']

    for p in [0,2,3]: # 1,4 are not needed - electron transport assumed to be spontaneous
        if p == 0 or p == 1:
            N = N_a
        elif p == 2:
            N = N_YSZ
        elif p == 3 or p == 4:
            N = N_c
        J[p], rhs[p] = boundaries_individual(K[p], dx, bc[p], indices[p], N, isMi, M_ins, scaling_factor)
        J[p], sum_nb[p] = interior_individual(J[p], indices[p], K[p], ds[p], dx, bc[p], isMi, M_ins, scaling_factor)

    print('Done!')
    return J, rhs, sum_nb

def initilize_field_variables_individual_entire_cell(inputs, indices, isMi, M_instances = None, scaling_factor = None):
    # initial guess
    import numpy as np
    residuals = [[]]*5
    phi = [None]*5
    N_y = inputs['Ny']

    rhoH2 = 0.0251       # Density of hydrogen [kg/m^3]
    cH2_b = inputs['pH2_in'] / inputs['P'] * rhoH2
    rhoO2 = 0.001429     # Density of oxygen [kg/m^3]
    cO2_b = inputs['pO2_in'] / inputs['P'] * rhoO2

    # phi[0] = cH2
    # phi[1] = Vel_anode
    # phi[2] = Vio
    # phi[3] = cO2
    # phi[4] = Vel_cathode
    init_cond = [cH2_b, inputs['Vel_anode'], inputs['Vio_init'], cO2_b, inputs['Vel_cathode']]

    # if M_instances is None:
    for p in range(5):
        L = len(indices[p]['all_points'])
        phi[p] = np.ones(shape = L, dtype = float) * init_cond[p]
        
    if isMi:
        j_seg = [N_y//M_instances*i for i in range(M_instances)]
        j_seg.append(N_y)
        j_seg = np.array(j_seg)
        
        for p in [0,1,2]:
            for n in range(len(indices[p]['all_points'])):
                _,j,_ = indices[p]['all_points'][n]
                sf = scaling_factor**(np.sum(j>=j_seg)-1)
                phi[p][n] = init_cond[p]*sf

    return phi, residuals

def get_flags(ind_1_stack, ind_2_stack, ind_stack, L):
    import numpy as np
    flag_1, flag_2 = np.zeros(shape=(2, L), dtype=int)-1
    a = 0
    b = []
    for n in range(len(ind_1_stack)):
        while np.any(ind_2_stack[n,:] != ind_stack[a,:]):
            a += 1
        b = b+1 if b!=[] else a+1
        while np.any(ind_1_stack[n,:] != ind_stack[b,:]):
            b += 1
        flag_1[b] = a
        flag_2[a] = b

    return flag_1, flag_2