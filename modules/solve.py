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

def threshold(
        phi_new, 
        masks_dict, 
        thd,
        ):
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

def Newton_loop_individual(
        J, 
        rhs, 
        phi, 
        indices, 
        field_functions, 
        masks_dict, 
        sum_nb, 
        residuals,
        ):
    
    from scipy.sparse import linalg
    from time import time

    prev_iters = len(residuals[0])
    iter = prev_iters
    ds = masks_dict['ds']

    while iter < prev_iters+inputs['solver_options']['max_iter_non']:
    
        # update J[n,n] and rhs[b] for interior points (source==True)
        t = time()
        J, rhs = update_interior(
            J, 
            rhs, 
            indices, 
            field_functions, 
            phi, 
            ds,
            sum_nb,
            )
        time_update_J = time() - t

        # scaling here
        J_scl, rhs_scl = matrix_scaling_individual(J, rhs)

        # solve the linear loop with the exact Jacobian matrix for a few iterations
        phi_new = [None]*3
        phase_names = ['gas', 'elec', 'ion']
        t = time()
        for p in [0,1,2]:
            if phase_names[p] not in inputs['solver_options']['transport_eqs']:
                phi_new[p] = phi[p]
                continue
            phi_new[p], _ = linalg.gmres(
                J_scl[p], 
                rhs_scl[p], 
                x0=phi[p], 
                maxiter=inputs['solver_options']['max_iter_lin'], 
                tol=1e-20,
                )
        time_gmres = time() - t
        
        # error monitoring here
        max_res, residuals = error_monitoring_individual(
            phi, 
            phi_new, 
            J, 
            rhs, 
            residuals, 
            iter, 
            time_update_J, 
            time_gmres,
            )
        
        # updating the solution here
        phi = update_phi_individual(
            phi, 
            phi_new,
            )

        # check the convergence
        if max_res < inputs['solver_options']['tol'] and iter>5:
            tol = inputs['solver_options']['tol']
            print(f'Newton loop is converged to a tolerance of {tol} in {iter} iterations.')
            break

        iter += 1
    
    return phi, residuals

def update_interior(
        J, 
        rhs, 
        indices, 
        field_functions, 
        phi, 
        ds, 
        sum_nb,
        ):

    N = ds[0].shape
    dx = inputs['microstructure']['dx']

    f = field_functions['f']
    fp = field_functions['fp']

    ratio = [None]*3
    ratio[0] = np.zeros_like(rhs[0])
    ratio[1] = np.zeros_like(rhs[1])
    ratio[2] = np.zeros_like(rhs[2])

    phi_dense = [np.zeros(N)]*3
    phi_dense[0][ds[0]] = phi[0]
    phi_dense[1][ds[1]] = phi[1]
    phi_dense[2][ds[2]] = phi[2]

    isMi = inputs['is_multiple_instances']
    M_ins = inputs['M_instances']
    if isMi:
        j_max = N[1]-1
        j_seg = j_max//M_ins+1

    phase_names = ['gas', 'elec', 'ion']
    for phase in [0,1,2]:
        if phase_names[phase] not in inputs['solver_options']['transport_eqs']:
            continue
        p = [[]]*3
        phi_block = [[]]*2
        mask = [[]]*2
        out = [0,1,2]
        out.remove(phase)

        for n in indices[phase]['source']:

            i,j,k = indices[phase]['all_points'][n]
            
            p[phase] = phi[phase][n]

            if isMi:
                if np.mod(j, j_seg)==0 and j!=0 and j!=j_max:
                    phi_block[0] = phi_dense[out[0]][i-1:i+2,j:j+2,k-1:k+2]
                    phi_block[1] = phi_dense[out[1]][i-1:i+2,j:j+2,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j:j+2,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j:j+2,k-1:k+2]
                elif np.mod(j, j_seg)==j_seg-1 and j!=0 and j!=j_max:
                    phi_block[0] = phi_dense[out[0]][i-1:i+2,j-1:j+1,k-1:k+2]
                    phi_block[1] = phi_dense[out[1]][i-1:i+2,j-1:j+1,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j-1:j+1,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j-1:j+1,k-1:k+2]
                else:   
                    phi_block[0] = phi_dense[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                    phi_block[1] = phi_dense[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
            else:
                phi_block[0] = phi_dense[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                phi_block[1] = phi_dense[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
                mask[0] = ds[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                mask[1] = ds[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
                
            p[out[0]] = np.average(phi_block[0][mask[0]])
            p[out[1]] = np.average(phi_block[1][mask[1]])

            if np.sum(np.isnan(p))>0:
                continue
            ratio[phase][n] = fp[phase](p[0],p[1],p[2])*dx**3 / sum_nb[phase][n]
            J[phase][n,n] = -sum_nb[phase][n] - fp[phase](p[0],p[1],p[2])*dx**3    # aP
            rhs[phase][n] = (f[phase](p[0],p[1],p[2]) - fp[phase](p[0],p[1],p[2])*phi[phase][n]) * dx**3 # RHS

    return J, rhs

def matrix_scaling_individual(
        J, 
        rhs,
        ):
    
    from scipy.sparse import diags
    # Scaling the Jacobian matrix and RHS vector 
    # Basically same as using Jacobi preconditioner (other preconditioners should also be considered)
    scl_vec = [None]*3
    scl_mat = [None]*3
    J_scl = [None]*3
    rhs_scl = [None]*3

    phase_names = ['gas', 'elec', 'ion']
    for p in [0,1,2]:       # Only solve for ion transport, otherwise: for p in [0,1,2]:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        scl_vec[p] = 1/J[p].diagonal(0).ravel()
        scl_mat[p] = diags(scl_vec[p])
        J_scl[p] = scl_mat[p] @ J[p]
        rhs_scl[p] = scl_mat[p] @ rhs[p]

    return J_scl, rhs_scl

def error_monitoring_individual(
        phi, 
        phi_new, 
        J_scl, 
        rhs_scl, 
        residuals, 
        iter,
        time_update_J,
        time_gmres,
        ):

    res = [None]*len(phi)
    phase_names = ['gas', 'elec', 'ion', 'gas']
    for p in range(len(phi)):
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            residuals[p] = residuals[p] + [10]       # arbitrary large number
            continue
        # res[p] = rhs_scl[p] - J_scl[p].tocsr()@phi_new[p]    # first method
        res[p] = np.abs(phi_new[p] - phi[p])                 # second method
        residuals[p] = residuals[p] + [np.linalg.norm(res[p])/np.linalg.norm(phi_new[p])]

    phases = [0,1,2,3]
    phases_to_remove = []
    for p in phases:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
                phases_to_remove.append(p)
    for p in phases_to_remove:
        phases.remove(p)

    max_res = 0

    for i in range(len(phases)):
        max_res = max(max_res, residuals[phases[i]][iter])
    
    if len(phi) == 3:
        if iter % 10 == 0:
            print('\nIter:  res rhoH2:       res Vel:       res Vio:       time GMRES:    time update J:', flush=True)
            print('----------------------------------------------------------------------------------', flush=True)
        if inputs['solver_options']['transport_eqs'] == ['ion']:
            print(f'{iter:<7}Not solved     Not solved     {residuals[2][iter]:<15.2e}{time_gmres:<15.1e}{time_update_J:<.1e}', flush=True)
        elif inputs['solver_options']['transport_eqs'] == ['ion', 'gas']:
            print(f'{iter:<7}{residuals[0][iter]:<15.2e}Not solved     {residuals[2][iter]:<15.2e}{time_gmres:<15.1e}{time_update_J:<.1e}', flush=True)
        else:
            print(f'{iter:<7}{residuals[0][iter]:<15.2e}{residuals[1][iter]:<15.2e}{residuals[2][iter]:<15.2e}{time_gmres:<15.1e}{time_update_J:<.1e}', flush=True)
    
    if len(phi) == 4:
        if iter % 10 == 0:
            print('\nIter:  rhoH2:     Vel:       Vio:       rhoO2:     ', flush=True)
            print('----------------------------------------------------', flush=True)
        print(f'{iter:<7}{residuals[0][iter]:<11.2e}{residuals[1][iter]:<11.2e}{residuals[2][iter]:<11.2e}{residuals[3][iter]:<11.2e}', flush=True)

    return max_res, residuals

def update_phi_individual(
        phi, 
        phi_new,
        ):
    
    uf = inputs['solver_options']['uf']

    if len(phi) == 3:
        und_fac = [uf['rhoH2'], uf['Vel'], uf['Vio']]
    elif len(phi) == 4:
        und_fac = [uf['rhoH2'], uf['Vel'], uf['Vio'], uf['rhoO2']]

    phase_names = ['gas', 'elec', 'ion', 'gas']
    phases = range(len(phi))
    for p in phases:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        phi[p] = und_fac[p]*phi_new[p] + (1-und_fac[p])*phi[p]
    
    return phi

# specific functions for the entire cell
def Newton_loop_individual_entire_cell(
        J, 
        rhs, 
        phi, 
        indices, 
        field_functions, 
        masks_dict, 
        sum_nb, 
        residuals,
        ):
    
    import warnings
    from scipy.sparse import linalg
    from time import time

    def matrix_scaling_individual_entire_cell():

        from scipy.sparse import diags
        # Scaling the Jacobian matrix and RHS vector 
        # Basically same as using Jacobi preconditioner 
        # (other preconditioners should also be considered)

        scl_vec = [None]*4
        scl_mat = [None]*4
        J_scl = [None]*4
        rhs_scl = [None]*4

        phase_names = ['gas', 'elec', 'ion', 'gas']
        for p in [0,1,2,3]:       # Only solve for ion transport, otherwise: for p in [0,1,2]:
            if phase_names[p] not in inputs['solver_options']['transport_eqs']:
                continue

            scl_vec[p] = 1/J[p].diagonal(0).ravel()
            scl_mat[p] = diags(scl_vec[p])
            J_scl[p] = scl_mat[p] @ J[p]
            rhs_scl[p] = scl_mat[p] @ rhs[p]

            J_scl[p] = J_scl[p].tocsr()
            J_scl[p].eliminate_zeros()

        return J_scl, rhs_scl
    
    def update_interior_entire_cell():
        
        dx = inputs['microstructure']['dx']
        ds = masks_dict['ds']
        N = [ds[i].shape for i in range(4)]
        Lx_a = N[0][0]
        Lx_e = N[1][0] - N[0][0] - N[3][0]
        Lx_c = N[3][0]

        f = field_functions['f']
        
        fp = field_functions['fp']

        signs = field_functions['signs']
        
        ratio = [None]*4
        ratio = [np.zeros_like(rhs[i]) for i in range(4)]

        ds_entire = [None]*4
        # gas phase (anode) - H2
        ds_entire[0] = np.concatenate((ds[0], np.zeros((Lx_e + Lx_c, N[0][1], N[0][2]),dtype=bool)), axis=0)
        # electron phase (anode & cathode)
        ds_entire[1] = ds[1]
        # ion phase (anode & cathode)
        ds_entire[2] = ds[2]
        # gas phase (cathode) - O2
        ds_entire[3] = np.concatenate((np.zeros((Lx_a+Lx_e, N[0][1], N[0][2]),dtype=bool),ds[3]),axis=0)
        
        phi_dense = np.zeros(N[1])
        for i in range(4):
            # 1: gas phase (anode) - H2
            # 2: electron phase (anode & cathode)
            # 3: ion phase (anode & cathode)
            # 4: gas phase (cathode) - O2
            phi_dense[ds_entire[i]] = phi[i]

        mask_entire = [None]*3
        # gas phase (anode & cathode)
        mask_entire[0] = np.concatenate((ds[0], np.zeros((Lx_e, N[0][1], N[0][2]),dtype=bool),ds[3]), axis=0)
        # electron phase (anode & cathode)
        mask_entire[1] = ds[1]
        # ion phase (anode & cathode)
        mask_entire[2] = ds[2]

        # microstructure of the anode side
        phase_names = ['gas', 'elec', 'ion', 'gas']
        for phase in [0,1,2,3]:
            if phase_names[phase] not in inputs['solver_options']['transport_eqs']:
                continue 

            p = [None]*3
            mask = [None]*2
            out = [0,1,2]           

            if phase==3:
                out.remove(0)
            else:
                out.remove(phase)

            for n in indices[phase]['source']:

                flag_a, flag_e, flag_c = False, False, False

                i,j,k = indices[phase]['all_points'][n]
                if phase==0: flag_a = True
                elif phase==3:
                    i += Lx_a + Lx_e
                    flag_c = True
                elif phase==1 or phase==2:
                    if i < Lx_a: flag_a = True
                    elif i >= Lx_a and i < Lx_a + Lx_e: flag_e = True       # it should never happen [no source points in the electrolyte]
                    elif i >= Lx_a + Lx_e: flag_c = True

                if phase==3:    p[0]        = phi[phase][n]
                else:           p[phase]    = phi[phase][n]

                phi_block = phi_dense[i-1:i+2,j-1:j+2,k-1:k+2]
                mask[0] = mask_entire[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                mask[1] = mask_entire[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]

                p[out[0]] = np.average(phi_block[mask[0]])
                p[out[1]] = np.average(phi_block[mask[1]])

                if np.sum(np.isnan(p))>0:
                    continue

                # f_idx: index of the function in the list of functions
                if phase==0:                f_idx = 0
                elif phase==3:              f_idx = 3
                elif phase==1 and flag_a:   f_idx = 1
                elif phase==1 and flag_c:   f_idx = 4
                elif phase==2 and flag_a:   f_idx = 2
                elif phase==2 and flag_c:   f_idx = 5

                # f and fp at the node
                f_node = f[f_idx](p[0],p[1],p[2])
                fp_node = fp[f_idx](p[0],p[1],p[2])

                if f_node * signs[f_idx] < 0:
                    # print warning message
                    warnings.warn(f'f{f_idx} is not consistent with the exptected sign!')

                ratio[phase][n] = fp_node*dx**3 / sum_nb[phase][n]
                J[phase][n,n] = -sum_nb[phase][n] - fp_node*dx**3    # aP
                rhs[phase][n] = (f_node - fp_node*phi[phase][n]) * dx**3 # RHS

        return J, rhs, ratio

    prev_iters = len(residuals[0])
    iter = prev_iters
    iter_converge = 0

    while iter < prev_iters+inputs['solver_options']['max_iter_non']:
    
        # update J[n,n] and rhs[b] for interior points where source==True
        t = time()
        J, rhs, ratio = update_interior_entire_cell()
        time_update_J = time() - t

        # scaling here
        J_scl, rhs_scl = matrix_scaling_individual_entire_cell()

        # solve the linear loop with the exact Jacobian matrix for a few (max_iter_lin) iterations
        phi_new = [None]*4
        phase_names = ['gas', 'elec', 'ion', 'gas']
        t = time()
        for p in [0,1,2,3]:
            if phase_names[p] not in inputs['solver_options']['transport_eqs']:
                phi_new[p] = phi[p]
                continue

            phi_new[p], _ = linalg.gmres(
                J_scl[p], 
                rhs_scl[p], 
                x0=phi[p], 
                maxiter=inputs['solver_options']['max_iter_lin'],
                # tol=1e-20,
                )
            
        time_gmres = time() - t

        # error monitoring here
        max_res, residuals = error_monitoring_individual(
            phi, 
            phi_new, 
            J_scl, 
            rhs_scl, 
            residuals, 
            iter,
            time_update_J,
            time_gmres, 
            )
        
        # updating the solution here
        phi = update_phi_individual(
            phi, 
            phi_new,
            )

        if max_res < inputs['solver_options']['tol']:
            iter_converge += 1
            if iter_converge == 5:      
                tol = inputs['solver_options']['tol']
                print(f'Newton loop is converged to a tolerance of {tol} in {iter} iterations.')
                break
                
        iter += 1
    
    return phi, residuals

