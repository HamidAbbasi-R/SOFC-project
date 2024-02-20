# if __name__ == '__main__':
import numpy as np
import json

try:
    from config import ID
    from config import simulation_type 
    input_str = f'input files/inputs_{simulation_type}_' + str(ID).zfill(3) + '.json'
    inputs = json.load(open(input_str))
except: 
    pass

def sourcefunc_calc(
        TPB_dict,
        plot_source_term=False,
        ):
    
    print("Source function calculations...", end='')
    from sympy import diff, exp, lambdify, log, simplify, symbols
    # Constants
    dx = inputs['microstructure']['dx']
    MH2 = 2.01588e-3        # Molar weight of hydrogen [kg/mol]
    Ru = 8.31446261815324   # Universal gas constant [J/mol/K]
    F = 96485.33289         # Faraday's constant [C/mol]
    aa = 0.5                # anode transfer coefficient []

    # conductivities
    cond_H2 = 7.474e-4        # [m^2/s]  https://doi.org/10.1016/j.ces.2008.07.037  should Knudsen diffusivity be considered?
    cond_el = 3.27e6 - 1065.3 * inputs['operating_conditions']['T']       # [S/m]
    cond_ion = 3.34e4 * np.exp(-10350/inputs['operating_conditions']['T'])    # [S/m]
    K = [cond_H2, cond_el, cond_ion] 

    T = inputs['operating_conditions']['T']

    # boundary conditions
    pH2_b = np.array(inputs['boundary_conditions']['pH2_b']) * 101325      # partial pressure of hydrogen @ anode boundary, bulk [Pa]
    pH2_inlet = inputs['boundary_conditions']['pH2_inlet'] * 101325      # partial pressure of hydrogen @ fuel cell inlet [Pa]
    ptot = inputs['operating_conditions']['P'] * 101325       # total pressure [Pa]
    Vel_b = np.array(inputs['boundary_conditions']['Vel_b'])      # electron potential @ anode boundary, bulk [V]
    Vio_b = np.array(inputs['boundary_conditions']['Vio_b'])      # ion potential @ electrolyte boundary, bulk [V]

    # other variables
    pH2O_b = ptot - pH2_b[0]          # partial pressure of H2O @ anode boundary, bulk [Pa]
    pH2O_inlet = ptot - pH2_inlet  # partial pressure of H2O @ fuel cell inlet [Pa]

    # concentrations (or mass densities)
    cH2_b = pH2_b*MH2/Ru/T               # Hydrogen concentration @ anode boundary, bulk [kg/m^3] - boundary condition

    # symbolic expressions
    cH2 = symbols('cH2')    # hydrogen mass fraction [pores]
    Vel = symbols('Vel')    # electron potential [Nickel]
    Vio = symbols('Vio')    # ion potential [YSZ]
    vars = [cH2, Vel, Vio]

    pH2 = cH2*Ru*T/MH2         # partial pressure of hydrogen, symbolic [Pa]
    pH2O = ptot - pH2          # partial pressure of water, symbolic [Pa]
    
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
    # Boer model:
    I0a_l = 31.4 * (pH2)**(-0.03) * (pH2O)**(0.4) * np.exp(-152155/Ru/T)  
    # Bieberle model: (It's not very different from Boer model, refer to )
    # I0a_l = 0.0013 * (pH2)**(0.11) * (pH2O)**(0.67) * np.exp(-0.849e5/R/T)
    # I0a_l = 1 * 2.14e-10 * 1e6      # from Shearing et al. 2010 for T=900 C [A/m]  (for test purposes)

    # The way that lineal exchange current density is transformed to volumetric exchange
    # current density is not clear. Two possible conversion factors can be used. 
    # Is the conversion factor correct?
    # conversion factor 1 > 2 > 3
    conversion_fac_1 = dx / (4*dx**3)        # [m/m3]
    conversion_fac_2 = TPB_dict['TPB_density']      # [m/m3]
    conversion_fac_3 = 5e12      # [m/m3] typical TPB density reported in Prokop PhD thesis
    I0a = I0a_l*conversion_fac_1      # volumetric Exchange current density, anode [A/m^3]

    # Tseronis et al. 2012 model for anode current density
    n = 2       # number of electrons transferred per reaction
    if inputs['solver_options']['transport_eqs'] == ['ion']:
        eta_a_con = Ru*T/n/F*np.log(pH2_inlet/pH2_b[0]*pH2O_b/pH2O_inlet)         # anode concentration overpotential [V]
        if eta_a_con > (Vel_b[0] - Vio_b[0]):
            raise ValueError('Concentration overpotential is greater than the difference between ion and electron potentials.')
    else:
        eta_a_con = Ru*T/n/F*log(pH2_inlet/pH2*pH2O/pH2O_inlet)         # anode concentration overpotential [V]
        

    eta_a_act = Vel - Vio - eta_a_con                      # anode activation overpotential [V]
    Ia = I0a*(
        exp( n*   aa * F * eta_a_act /Ru/T)-
        exp(-n*(1-aa)* F * eta_a_act /Ru/T))               # anode current density [A/m^3]

    # Shearing et al. 2010 model for anode current density
    # Voe = 1.23 - 2.304e-4*(T-298.15)        # standard potential [V]
    # Veq = Voe - R*T/2/F*log(nH2O/nH2)      # equilibrium potential [V]
    # eta = Veq - Vio + Vel                   # overpotential [V]
    # Ia = I0a*(exp(2*aa*F*eta/R/T)
    #          -exp(-2*(1-aa)*F*eta/R/T))               # ??? anode current density [A/m^3]

    # initialize the source function list
    source_func = [None]*3         # initialization
    source_func[0] = simplify(-Ia/n/F*MH2) if inputs['solver_options']['transport_eqs'] != ['ion'] else None # mass [kg/m3/s]
    source_func[1] = simplify(-Ia) if inputs['solver_options']['transport_eqs'] != ['ion'] else None            # electron [A/m3]
    source_func[2] = simplify(Ia)           # ion [A/m3]
    expected_sign = [-1,-1,1]             # expected sign of the source terms
    # source_func[0] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[1] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case
    # source_func[2] = exp(-cH2**2 - Vel**2 - Vio**2)       # test case

    # source term treatment
    f = [None]*3
    fp = [None]*3
    phase_names = ['gas', 'elec', 'ion']
    for p in [0,1,2]:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        f[p] = lambdify(vars, source_func[p]) #[kg/m3/s] or [A/m3]
        fp[p] = lambdify(vars, diff(source_func[p], vars[p]))

    # Threshold values
    cH2_min = 1e-7             # minimum concentration [kg/m^3]
    cH2_max = cH2_b[0]         # maximum concentration [kg/m^3]
    Vel_min = -2             # minimum electronic potential [V]
    Vel_max = Vel_b[0]             # maximum electronic potential [V]
    Vio_min = Vio_b[0]             # minimum ionic potential [V]
    Vio_max = 2             # maximum ionic potential [V]
    thd = [[cH2_min,cH2_max], [Vel_min,Vel_max], [Vio_min,Vio_max]]

    # Flux values at top and bottom boundaries
    # This is just an approximation of the gradient in X direction
    # gradient_ion_X = Vel_b/inputs['microstructure']['length']['X']     # [V/m] 
    gradient_ion_X = Vel_b[0]/10e-6     # [V/m] assuming the thickness of the active layer is 10 um 
    # Ratio of ion potential gradient in Z direction to ion potential gradient in X direction
    C = inputs['boundary_conditions']['flux_Z_constant']    # [-]
    gradient_ion_Z = C * gradient_ion_X     # [V/m]
    flux_ion_Z = gradient_ion_Z * K[2]      # [A/m^2]

    field_functions = {
        'f': f,
        'fp': fp,
        'signs': expected_sign,
        'eta_a_act': lambdify([cH2, Vel, Vio], eta_a_act),
        'eta_a_con': lambdify([cH2], eta_a_con),
        'Ia': lambdify([cH2, Vel, Vio], Ia)}
    
    # plot variation of ion source term with ion potential
    if plot_source_term:
        import plotly.express as px
        import plotly.graph_objects as go

        Vio_range = np.linspace(0, Vel_b[0], 100)
        source_term = f[2](cH2_b[0], Vel_b[0], Vio_range)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Vel_b[0]-Vio_range, y=source_term))
        fig.update_layout(
            title='Ion source term variation with ion potential',
            xaxis_title='(Electron potential - Ion potential) [V]',
            yaxis_title='Ion source term [A/m3]',
            )
        # fig.update_yaxes(type="log", exponentformat='power', showexponent='all')
        fig.show()
    
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
    'West':   ['Dirichlet', cH2_b[0], cH2_b[1]],       
    # electrolyte side
    'East':   ['Neumann', 0],
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{

    # the second phase [Nickel - electron] ??
    # anode side
    'West':   ['Dirichlet', Vel_b[0], Vel_b[1]],         
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
    'East':   ['Dirichlet', Vio_b[0], Vio_b[1]],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', -flux_ion_Z],
    'Top':    ['Neumann', flux_ion_Z]}]

    bc_dict = bc

    print("Done!")
    return field_functions, thd, bc_dict, K

def get_indices_all(
        domain, 
        TPB_dict,
        ):
    
    print('Identifying neighboring cells and obtaining indices...', end=' ')
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
        indices[p] = get_indices(domain, TPB_mask, ds[p], p)

    masks_dict = {
        'ds': ds, 
        'ds_lin': [],     
        'ds_lin_s': []}
    print('Done!')
    return masks_dict, indices

def get_indices(
        domain, 
        TPB_mask_old, 
        ds, 
        phase,
        ):
    
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
    
    phase_names = ['gas', 'elec', 'ion']
    if phase_names[phase] in inputs['solver_options']['transport_eqs']: 
        with concurrent.futures.ProcessPoolExecutor() as executor:
            WE_res = executor.submit(get_flags, ind_west_stack, ind_east_stack, ind_stack, L)
            SN_res = executor.submit(get_flags, ind_south_stack, ind_north_stack, ind_stack, L)
            BT_res = executor.submit(get_flags, ind_bottom_stack, ind_top_stack, ind_stack, L)

            flag_west, flag_east = WE_res.result()
            flag_south, flag_north = SN_res.result()
            flag_bottom, flag_top = BT_res.result()
    

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

def create_SOLE_individual(
        bc_dict, 
        indices, 
        masks_dict,
        K,
        ):
    
    print('Writing Jacobian and rhs matrix...', end=' ')

    dx = inputs['microstructure']['dx']

    N = [
        inputs['microstructure']['length']['X']//dx, 
        inputs['microstructure']['length']['Y']//dx, 
        inputs['microstructure']['length']['Z']//dx]

    J = [None]*3
    rhs = [None]*3
    sum_nb = [None]*3
    bc = bc_dict
    ds = masks_dict['ds']
    isMi = inputs['is_multiple_instances']
    M_ins = inputs['M_instances']
    scaling_factor = inputs['scaling_factor']

    phase_names = ['gas', 'elec', 'ion']
    for p in [0,1,2]: # only solve the ion phase, otherwise: for p in [0,1,2]:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        J[p], rhs[p], sum_nb[p] = boundaries_individual(K[p], dx, bc[p], indices[p], N, isMi, M_ins, scaling_factor)
        J[p], sum_nb[p] = interior_individual(J[p], indices[p], K[p], ds[p], dx, bc[p], sum_nb[p], isMi, M_ins, scaling_factor)

    print('Done!')
    return J, rhs, sum_nb

def boundaries_individual(
        K, 
        dx, 
        bc, 
        indices, 
        N, 
        isMi, 
        M_ins=None, 
        scaling_factor=None,
        ):
    
    from scipy.sparse import lil_matrix

    if not isinstance(K,np.ndarray):
        K = np.ones(shape = N, dtype = float)*K

    # initializing left hand side and right hand side of the system of equaitons
    L = len(indices['all_points'])
    rhs = np.zeros(shape = L, dtype = float) # right hand side vector
    sum_nb = np.zeros(shape = L, dtype = float) # sigma a_nb vector

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
            _,j,k = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K[0,j,k]*dx     # aP
            cte = bc['West'][1] + (k/N[2])*(bc['West'][2] - bc['West'][1])
            rhs[n] = cte*sf*K[0,j,k]*dx

    if bc['West'][0] == 'Neumann' and bc['West'][1] != 0:
        for n in indices['west_bound']:
            _,j,k = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            aE = -K[0,j,k]*dx
            J[n,indices['east_nb'][n]] = aE
            sum_nb[n] = aE
            J[n,n] = -aE
            rhs[n] = bc['West'][1]*sf*dx**2


    # east side (i=N[0]-1)
    if bc['East'][0] == 'Dirichlet':
        for n in indices['east_bound']:
            _,j,k = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K[-1,j,k]*dx     # aP
            cte = bc['East'][1] + (k/N[2])*(bc['East'][2] - bc['East'][1])
            rhs[n] = cte*sf*K[-1,j,k]*dx

    if bc['East'][0] == 'Neumann' and bc['East'][1] != 0:
        for n in indices['east_bound']:
            _,j,k = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            aW = -K[-1,j,k]*dx
            J[n,indices['west_nb'][n]] = aW
            sum_nb[n] = aW
            J[n,n] = -aW
            rhs[n] = bc['East'][1]*sf*dx**2


    # south side (j=0)
    if bc['South'][0] == 'Dirichlet':
        for n in indices['south_bound']:
            i,_,k = indices['all_points'][n]
            J[n,n] = 1*K[i,0,k]*dx    # aP
            rhs[n] = bc['South'][1]*K[i,0,k]*dx

    if bc['South'][0] == 'Neumann' and bc['South'][1] != 0:
        for n in indices['south_bound']:
            i,_,k = indices['all_points'][n]
            aN = -K[i,0,k]*dx
            J[n,indices['north_nb'][n]] = aN
            sum_nb[n] = aN
            J[n,n] = -aN
            rhs[n] = bc['South'][1]*dx**2


    # north side (j=N[1]-1)
    if bc['North'][0] == 'Dirichlet':
        for n in indices['north_bound']:
            i,_,k = indices['all_points'][n]
            J[n,n] = 1*K[i,-1,k]*dx     # aP
            rhs[n] = bc['North'][1]*K[i,-1,k]*dx

    if bc['North'][0] == 'Neumann' and bc['North'][1] != 0:
        for n in indices['north_bound']:
            i,_,k = indices['all_points'][n]
            aS = -K[i,-1,k]*dx
            J[n,indices['south_nb'][n]] = aS
            sum_nb[n] = aS
            J[n,n] = -aS
            rhs[n] = bc['North'][1]*dx**2


    # bottom side (k=0)
    if bc['Bottom'][0] == 'Dirichlet':
        for n in indices['bottom_bound']:
            i,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K[i,j,0]*dx   # aP
            rhs[n] = bc['Bottom'][1]*sf*K[i,j,0]*dx

    if bc['Bottom'][0] == 'Neumann' and bc['Bottom'][1] != 0:
        for n in indices['bottom_bound']:
            i,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            aT = -K[i,j,0]*dx
            J[n,indices['top_nb'][n]] = aT
            sum_nb[n] = aT
            J[n,n] = -aT
            rhs[n] = bc['Bottom'][1]*sf*dx**2

    # top side (k=N[2]-1)
    if bc['Top'][0] == 'Dirichlet':
        for n in indices['top_bound']:
            i,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            J[n,n] = 1*K[i,j,0]*dx  # aP
            rhs[n] = bc['Top'][1]*sf*K[i,j,0]*dx
    
    if bc['Top'][0] == 'Neumann' and bc['Top'][1] != 0:
        for n in indices['top_bound']:
            i,j,_ = indices['all_points'][n]
            sf = scaling_factor**(np.sum(j>=j_seg)-1) if isMi else 1
            aB = -K[i,j,0]*dx
            J[n,indices['bottom_nb'][n]] = aB
            sum_nb[n] = aB
            J[n,n] = -aB
            rhs[n] = bc['Top'][1]*sf*dx**2

    return J, rhs, sum_nb

def interior_individual_obsolete(
        J, 
        indices, 
        K, 
        ds, 
        dx, 
        M_instances=None, 
        scaling_factor=None,
        ):
    
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

def harmonic_mean(a, b):
    return 2*a*b/(a+b)

def interior_individual(
        J, 
        indices, 
        K, 
        ds, 
        dx, 
        bc, 
        sum_nb,
        isMi, 
        M_ins=None, 
        scaling_factor=None,
        ):
    
    N = ds.shape
    if not isinstance(K,np.ndarray):
        K = np.ones(shape = N, dtype = float)*K

    i_max = np.max(indices['all_points'][:,0])
    j_max = np.max(indices['all_points'][:,1])
    k_max = np.max(indices['all_points'][:,2])
    hm = harmonic_mean

    for n in range(len(indices['all_points'])):
        i,j,k = indices['all_points'][n]

        # skip Dirichlet boundary points and non-zero Neumann boundary points that lie on the boundaries
        if not(bc['West'][0]   == 'Neumann' and bc['West'][1] == 0)    and i==0:        continue
        if not(bc['East'][0]   == 'Neumann' and bc['East'][1] == 0)    and i==i_max:    continue
        if not(bc['South'][0]  == 'Neumann' and bc['South'][1] == 0)   and j==0:        continue
        if not(bc['North'][0]  == 'Neumann' and bc['North'][1] == 0)   and j==j_max:    continue
        if not(bc['Bottom'][0] == 'Neumann' and bc['Bottom'][1] == 0)  and k==0:        continue
        if not(bc['Top'][0]    == 'Neumann' and bc['Top'][1] == 0)     and k==k_max:    continue

        # assign a_nb values for all points (including Neumann boundary points) [S]
        Kp = K[i,j,k]       # Kp is the conductivity of the point itself
        aW = -hm(K[i-1,j,k],Kp) * ds[i-1,j,k] * dx if i!=0     else 0 
        aE = -hm(K[i+1,j,k],Kp) * ds[i+1,j,k] * dx if i!=i_max else 0 
        aS = -hm(K[i,j-1,k],Kp) * ds[i,j-1,k] * dx if j!=0     else 0 
        aN = -hm(K[i,j+1,k],Kp) * ds[i,j+1,k] * dx if j!=j_max else 0 
        aB = -hm(K[i,j,k-1],Kp) * ds[i,j,k-1] * dx if k!=0     else 0 
        aT = -hm(K[i,j,k+1],Kp) * ds[i,j,k+1] * dx if k!=k_max else 0 

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

def initilize_field_variables_individual(
        masks_dict, 
        indices, 
        bc_dict,
        ):
    
    # initial guess
    print('Initializing field variables...', end = ' ')

    isMi = inputs['is_multiple_instances']
    M_instances = inputs['M_instances']
    scaling_factor = inputs['scaling_factor']

    residuals = [[]]*3
    phi = [[]]*3
    ds = masks_dict['ds']

    # using the same initial condition for ion and electron
    # might speed up the convergence
    init_cond = [bc_dict[0]['West'][1], 
                 bc_dict[1]['West'][1], 
                #  bc_dict[2]['East'][1],
                bc_dict[1]['West'][1]
                ]

    # if M_instances is None:
    for p in [0,1,2]:
        # L = len(np.where(ds[p]))
        L = np.sum(ds[p])
        phi[p] = np.ones(shape = L, dtype = float) * init_cond[p]
        
    if isMi:
        N_y = inputs['microstructure']['length']['Y']//inputs['microstructure']['dx']
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
def sourcefunc_calc_entire_cell(
        TPB_dict,
        ):

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

    # Constants
    dx = inputs['microstructure']['dx']
    MH2 = 2.01588e-3        # Molar weight of hydrogen [kg/mol]
    MH2O = 18.01528e-3      # Molar weight of water [kg/mol]
    MO2 = 31.9988e-3        # Molar weight of oxygen [kg/mol]
    MN2 = 28.0134e-3        # Molar weight of nitrogen [kg/mol]
    rho_H2 = 0.0251          # Density of hydrogen [kg/m^3]
    rhoH2O = 997            # Density of water [kg/m^3]
    rho_O2 = 1.429           # Density of oxygen [kg/m^3]
    rhoN2 = 1.250           # Density of nitrogen [kg/m^3]
    Ru = 8.31446261815324    # Universal gas constant [J/mol/K]
    F = 96485.33289         # Faraday's constant [C/mol]
    aa = 0.5                # anode transfer coefficient []
    ac = 0.5                # cathode transfer coefficient []

    # boundary conditions
    XH2_b = np.array(inputs['boundary_conditions']['XH2_b'])    # hydrogen mole fraction @ anode boundary, bulk [pores - anode] [-]
    XO2_b = np.array(inputs['boundary_conditions']['XO2_b'])    # oxygen mole fraction @ cathode boundary, bulk [pores - cathode] [-]
    XH2O_b = 1 - XH2_b      # water vapor mole fraction @ anode boundary, bulk [pores - anode] [-]
    
    ptot = inputs['operating_conditions']['P'] * 101325       # total pressure [Pa]
    pH2_b = np.array(inputs['boundary_conditions']['XH2_b']) * ptot      # partial pressure of hydrogen @ anode boundary, bulk [Pa]
    pH2_inlet = inputs['boundary_conditions']['XH2_inlet'] * ptot      # partial pressure of hydrogen @ fuel cell inlet [Pa]
    pO2_b = np.array(inputs['boundary_conditions']['XO2_b']) * ptot       # partial pressure of oxygen @ cathode boundary, bulk [Pa]
    pO2_inlet = inputs['boundary_conditions']['XO2_inlet'] * ptot       # partial pressure of oxygen @ fuel cell inlet [Pa]
    Vel_a = np.array(inputs['boundary_conditions']['Vel_a'])      # electron potential @ anode boundary, bulk [V]
    Vel_c = np.array(inputs['boundary_conditions']['Vel_c'])      # electron potential @ cathode boundary, bulk [V]
    

    # other variables
    pH2O_b = ptot - pH2_b[0]          # partial pressure of H2O @ anode boundary, bulk [Pa]
    pH2O_inlet = ptot - pH2_inlet  # partial pressure of H2O @ fuel cell inlet [Pa]
    pN2_b = ptot - pO2_b           # partial pressure of nitrogen @ cathode boundary, bulk [Pa]
    pN2_inlet = ptot - pO2_inlet   # partial pressure of nitrogen @ fuel cell inlet [Pa]

    # concentrations (or mass densities)
    T = inputs['operating_conditions']['T']         # temperature [K]
    rho_H2_b = pH2_b*MH2/Ru/T             # hydrogen mass density @ anode boundary, bulk [pores - anode] [-]
    rho_O2_b = pO2_b*MO2/Ru/T             # oxygen mass density @ cathode boundary, bulk [pores - cathode] [-]
    
    # symbolic expressions
    rho_H2 = symbols('rho_H2')    # hydrogen mass density [pores - anode] [-]
    Vel = symbols('Vel')    # electron potential [both anode and cathode]
    Vio = symbols('Vio')    # ion potential [YSZ - anode, electrolyte, cathode]
    rho_O2 = symbols('rho_O2')    # oxygen mass density [pores - cathode] [-]
    vars_a = [rho_H2, Vel, Vio]
    vars_c = [rho_O2, Vel, Vio]

    pH2 = rho_H2*Ru*T/MH2         # partial pressure of hydrogen, symbolic [Pa]
    pH2O = ptot - pH2          # partial pressure of water, symbolic [Pa]
    pO2 = rho_O2*Ru*T/MO2         # partial pressure of oxygen, symbolic [Pa]
    pN2 = ptot - pO2           # partial pressure of nitrogen, symbolic [Pa]
    
    # Exchange current per TPB length, anode [A/m], Prokop et al. 2018
    # Apparently the units of pH2 and pH2O should be [Pa] in this equations; refer to Boer's thesis for more info
    I0a_l = 31.4 * (pH2)**(-0.03) * (pH2O)**(0.4) * np.exp(-152155/Ru/T)     
    I0c_l = 1e-5 # ref: 10.1149/07801.2711ecst [A/m]
    
    # test values (for cases where the source function is not used; debugging)
    # I0a_l = 0
    # I0c_l = 0
    
    # conversion factors
    vf_TPB_a = TPB_dict['anode']['vf_TPB']      # volume fraction of TPB, anode [-]
    vf_TPB_c = TPB_dict['cathode']['vf_TPB']      # volume fraction of TPB, cathode [-]
    
    # ratio TPB
    cf_TPB_len_vol_a = TPB_dict['anode']['ratio_TPB']      # ratio of TPB length to volume [-]
    cf_TPB_len_vol_c = TPB_dict['cathode']['ratio_TPB']      # ratio of TPB length to volume [-]

    conversion_H2       = 0.25 / vf_TPB_a[0]
    conversion_Ni       = 0.25 / vf_TPB_a[1]
    conversion_YSZ_a    = 0.25 / vf_TPB_a[2]
    conversion_YSZ_c    = 0.25 / vf_TPB_c[0]
    conversion_O2       = 0.25 / vf_TPB_c[1]
    conversion_LSM      = 0.25 / vf_TPB_c[2]

    cf_len_to_vol = dx / (dx**3)        # [m/m3]
    
    I0a     = I0a_l * cf_len_to_vol * cf_TPB_len_vol_a      # volumetric Exchange current density, anode [A/m^3]
    I0c_TPB = I0c_l * cf_len_to_vol * cf_TPB_len_vol_c      # volumetric Exchange current density, cathode [A/m^3]
    
    # I0c_area = np.exp(-152155/Ru/T) #??? [A/m^2]
    # I0c_ISA = I0c_area*TPB_dict['cathode']['ISA_density']  # Exchange current density, cathode [A/m^3]

    n_a = 2       # number of electrons transferred per reaction
    n_c = 4       # number of electrons transferred per reaction
    if inputs['solver_options']['transport_eqs'] == ['ion']:
        pH2_var = pH2_b[0]
        pH2O_var = pH2O_b
        pO2_var = pO2_b[0]
    else:
        pH2_var = pH2
        pH2O_var = pH2O
        pO2_var = pO2

    eta_a_con = Ru*T/n_a/F*log(pH2_inlet/pH2_var * pH2O_var/pH2O_inlet)         # anode concentration overpotential [V]
    eta_c_con = Ru*T/n_c/F*log(pO2_inlet/pO2_var)         # cathode concentration overpotential [V]
    
    E0 = 1.253 - 2.4516e-4*(T-298.15)        # standard potential [V]
    # Nernst potential [V]; mole fractions [-] should be used instead of partial pressure [Pa]. 
    # Yes! It makes a difference! because we are dealing with square roots!
    E_N = E0 + Ru*T/2/F*np.log(XH2_b[0]*XO2_b[0]**0.5/XH2O_b[0])
    eta_a_act = (Vel - Vio) - eta_a_con                 # refer to figure [voltage distribution]
    eta_c_act = E_N - (Vel - Vio) - eta_c_con           # refer to figure [voltage distribution]

    Ia = I0a*(
        exp( n_a*   aa * F * eta_a_act /Ru/T)-
        exp(-n_a*(1-aa)* F * eta_a_act /Ru/T))               # anode current density [A/m^3]

    Ic_TPB = I0c_TPB*(
        exp( n_c*   ac * F * eta_c_act /Ru/T)-
        exp(-n_c*(1-ac)* F * eta_c_act /Ru/T))               # cathode current density [A/m^3]

    source_func = [None]*6         # initialization
    # gas anode
    source_func[0] = simplify(conversion_H2*Ia/n_a/F*MH2) if inputs['solver_options']['transport_eqs'] != ['ion'] else None # mass [kg/m3/s]
    
    # electron anode
    source_func[1] = simplify(-conversion_Ni*Ia) if inputs['solver_options']['transport_eqs'] != ['ion'] else None            # electron [A/m3]
    
    # ion anode
    source_func[2] = simplify(conversion_YSZ_a*Ia)           # ion [A/m3]
    
    # gas cahtode
    source_func[3] = simplify(-conversion_O2*Ic_TPB/n_c/F*MO2) if inputs['solver_options']['transport_eqs'] != ['ion'] else None # mass [kg/m3/s]
    
    # electron cathode
    source_func[4] = simplify(conversion_LSM*Ic_TPB) if inputs['solver_options']['transport_eqs'] != ['ion'] else None            # electron [A/m3]
    
    # ion cathode
    source_func[5] = simplify(-conversion_YSZ_c*Ic_TPB)           # ion [A/m3]

    expected_sign = [           # expected sign of the source terms
        -1,     # gas anode         : sink
        -1,     # electron anode    : sink
        +1,     # ion anode         : source
        -1,     # gas cathode       : sink
        +1,     # electron cathode  : source
        -1,     # ion cathode       : sink
        ]

    # source term treatment
    f = [None]*6
    fp = [None]*6
    
    phase_names = ['gas', 'elec', 'ion', 'gas', 'elec', 'ion']
    for p in range(6):
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        if p<3: # anode
            f[p] = lambdify(vars_a, source_func[p])    #[kg/m3/s] or [A/m3]
            fp[p] = lambdify(vars_a, diff(source_func[p], vars_a[p]))    #[1/s]
        else: # cathode
            f[p] = lambdify(vars_c, source_func[p])     #[kg/m3/s] or [A/m3]
            fp[p] = lambdify(vars_c, diff(source_func[p], vars_c[p-3]))    #[1/s]

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
        'eta_a_act': lambdify(vars_a, eta_a_act),
        'eta_a_con': lambdify([rho_H2], eta_a_con),
        'eta_c_act': lambdify(vars_c, eta_c_act),
        'eta_c_con': lambdify([rho_O2], eta_c_con),
        'Ia': lambdify(vars_a, Ia),
        'Ic': lambdify(vars_c, Ic_TPB)}

    # boundary conditions:
    # when two perpendicular pair of boundaries are set to periodic (e.g., south/north and east/west)
    # the edges of these boundaries will not be solved (aP will be zero at those nodes). 
    # this problem can be resolved with some extra work, but for now, try not to use two perpendicular 
    # periodic boundaries. in the case of SOFC microstructure, two perpendicular periodic boundaries are
    # not realistic in the first place.
    # units for Neumann BCs are [W/m^2], for heat equation:
    # for charge conservation model replace [W] with [A],
    # for mass conservation model replace [W] with [kg/s]
    
    bc = [{
    # the first phase [pores anode] ??
    # anode side
    'West':   ['Dirichlet', rho_H2_b[0], rho_H2_b[1]],       
    # electrolyte side
    'East':   ['Neumann', 0],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{
    # the second phase [Nickel - electron] ??
    # anode side
    'West':   ['Dirichlet', Vel_a[0], Vel_a[1]],         
    # cathode side
    'East':   ['Dirichlet', Vel_c[0], Vel_c[1]],           
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    },{
    # the third phase [YSZ - ion] ??
    # anode side
    'West':   ['Neumann', 0],           
    # 'West':   ['Dirichlet', -0.1,-0.1],       # only for test    
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
    'East':   ['Dirichlet', rho_O2_b[0], rho_O2_b[1]],          
    'South':  ['Neumann', 0],
    'North':  ['Neumann', 0],
    'Bottom': ['Neumann', 0],
    'Top':    ['Neumann', 0]
    }]

    bc_dict = bc

    print("Done!")
    return field_functions, bc_dict

def get_indices_entire_cell(
        separated_domains, 
        TPB_dict,
        ):

    from time import time
    import concurrent.futures
    t = time()
    print('Identifying neighbors and getting indices...', end=' ')
    domain_a = separated_domains[0]
    domain_e = separated_domains[1]
    domain_c = separated_domains[2]
    domain_entire = np.concatenate((domain_a, domain_e, domain_c), axis=0)
    domain = [domain_a, domain_entire, domain_entire, domain_c]

    TPB_mask_a = TPB_dict['anode']['TPB_mask']
    TPB_mask_c = TPB_dict['cathode']['TPB_mask']
    TPB_mask_entire = np.concatenate((TPB_mask_a, np.zeros_like(domain_e, dtype=bool), TPB_mask_c), axis=0)
    TPB_mask = [TPB_mask_a, TPB_mask_entire, TPB_mask_entire, TPB_mask_c]

    ds = [None]*4
    indices = [None]*4

    ds[0] = np.zeros(domain_a.shape, dtype=bool)
    ds[1] = np.zeros(domain_entire.shape, dtype=bool)
    ds[2] = np.zeros(domain_entire.shape, dtype=bool)
    ds[3] = np.zeros(domain_c.shape, dtype=bool)

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    for p in [0,1,2,3]:
        if      p==0:   mask = domain_a==1
        elif    p==1:   mask = np.logical_or(domain_entire==2, domain_entire==5)
        elif    p==2:   mask = domain_entire==3
        elif    p==3:   mask = domain_c==4
            
        ds[p][mask] = True
        # results_multithrd = executor.submit(
        #     get_indices, 
        #     domain[p], 
        #     TPB_mask[p], 
        #     ds[p], 
        #     p if p<3 else 0,
        #     )
        # indices[p] = results_multithrd.result()
        indices[p] = get_indices(
            domain[p], 
            TPB_mask[p], 
            ds[p], 
            p if p<3 else 0,
            )

    masks_dict = {
        'ds': ds, 
        'ds_lin': [],     
        'ds_lin_s': []}
    
    print('Done!')
    print('Time elapsed: %.2f s'%(time()-t))

    return masks_dict, indices

def create_SOLE_individual_entire_cell(
        bc_dict, 
        indices, 
        masks_dict,
        voxels,
        ):
    
    print('Writing Jacobian and rhs matrix...', end=' ')
    
    # conductivities#
    T = inputs['operating_conditions']['T']
    cond_H2 = 7.474e-4        # [m^2/s]  https://doi.org/10.1016/j.ces.2008.07.037  should Knudsen diffusivity be considered?  
    cond_el_a = 3.27e6 - 1065.3 * T       # [S/m]
    cond_ion_a = 3.34e4 * np.exp(-10350/T)    # [S/m]

    cond_O2 = 0.64e-4       # [m^2/s] https://doi.org/10.1016/j.mex.2020.100822 should Knudsen diffusivity be considered?
    cond_el_c = 3.27e6 - 1065.3 * T       # ??? needs reference [S/m]
    cond_ion_YSZ_c = 4e-1 # [S/m] @ T=1173.15 https://doi.org/10.1016/j.ssi.2004.11.019
    # cond_ion_LSM_c = 4e-6 # [S/m] @ T=1173.15 https://doi.org/10.1016/j.ssi.2004.11.019
    
    cond_ion_electrolyte = 4e-1  # ??? [S/m] @ T=1173.15 https://doi.org/10.1016/j.ssi.2004.11.019
    
    K = [
        cond_H2, 
        cond_el_a, 
        cond_ion_a, 
        cond_O2, 
        cond_el_c, 
        cond_ion_YSZ_c, 
        cond_ion_electrolyte,
        # cond_ion_LSM_c,
        ] 
    
    # voxels
    N_a = voxels[0]
    N_c = voxels[1]
    N_entire = [voxels[0][0] + voxels[1][0] + voxels[2][0], voxels[0][1], voxels[0][2]]
    N = [N_a, N_entire, N_entire, N_c]
    Lx_a = N[0][0]
    Lx_e = N[1][0] - N[0][0] - N[3][0]
    Lx_c = N[3][0]

    dx = inputs['microstructure']['dx']
    ds = masks_dict['ds']
    J = [None]*4
    rhs = [None]*4
    sum_nb = [None]*4
    bc = bc_dict

    isMi = inputs['is_multiple_instances']
    M_ins = inputs['M_instances']
    scaling_factor = inputs['scaling_factor']

    phase_names = ['gas', 'elec', 'ion', 'gas']
    for p in [0,1,2,3]:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        
        if p == 1:
            K_array = np.zeros(shape = N_entire, dtype = float)

            # Ni phase - anode
            K_array[:Lx_a,...][ds[1][:Lx_a,...]] = K[1] 
            
            # LSM phase - cathode
            K_array[Lx_a+Lx_e:,...][ds[1][Lx_a+Lx_e:,...]] = K[4]
        
        elif p == 2:
            K_array = np.zeros(shape = N_entire, dtype = float) 

            # YSZ phase - anode
            K_array[:Lx_a,...] = K[2] 
            
            # YSZ phase - electrolyte 
            K_array[Lx_a:Lx_a+Lx_e,...] = K[6] 

            # YSZ phase - cathode
            K_array[Lx_a+Lx_e:,...] = K[5]

            K_array_ion = K_array.copy()
        
        elif p==0 or p==3:
            K_array = K[p]

        J[p], rhs[p], sum_nb[p] = boundaries_individual(
            K_array,
            dx, 
            bc[p], 
            indices[p], 
            N[p], 
            isMi, M_ins, scaling_factor,
            )
        
        J[p], sum_nb[p] = interior_individual(
            J[p], 
            indices[p], 
            K_array, 
            ds[p], 
            dx, 
            bc[p], 
            sum_nb[p],
            isMi, M_ins, scaling_factor,
            )
            
    print('Done!')
    return J, rhs, sum_nb, K, K_array_ion

def initilize_field_variables_individual_entire_cell(
        masks_dict, 
        indices, 
        bc_dict,
        voxels,
        ):
    
    # initial guess
    print('Initializing field variables...', end = ' ')

    residuals = [[]]*4
    phi = [[]]*4
    ds = masks_dict['ds']

    # using the same initial condition for ion and electron
    # might speed up the convergence
    init_cond = [
        bc_dict[0]['West'][1], 
        [bc_dict[1]['West'][1], bc_dict[1]['East'][1]],
        -0.4,
        bc_dict[3]['East'][1],
        ]

    # if M_instances is None:
    for p in [0,1,2,3]:
        if p==1:
            L_a = np.sum(ds[p][:voxels[0][0],...])
            L_c = np.sum(ds[p][voxels[0][0] + voxels[2][0]:,...])
            phi[p] = np.concatenate((
                np.ones(shape = L_a, dtype = float) * init_cond[p][0],
                np.ones(shape = L_c, dtype = float) * init_cond[p][1],
            ))
        else:
            L = np.sum(ds[p])
            phi[p] = np.ones(shape = L, dtype = float) * init_cond[p]

    print('Done!')
    return phi, residuals

def get_flags(
        ind_1_stack, 
        ind_2_stack, 
        ind_stack, 
        L,
        ):
    
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