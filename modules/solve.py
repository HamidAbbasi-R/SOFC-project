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

def Newton_loop_individual(inputs, J, rhs, phi, indices, field_functions, masks_dict, sum_nb, residuals):
    import numpy as np
    from scipy.sparse import linalg

    dx = inputs['microstructure']['dx']
    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]

    prev_iters = len(residuals[0])
    iter = prev_iters
    ds = masks_dict['ds']

    while iter < prev_iters+inputs['solver_options']['max_iter_non']:
    
        # update J[n,n] and rhs[b] for interior points (source==True)
        J, rhs = update_interior(
            inputs, J, rhs, indices, 
            field_functions,
            phi,
            dx, N, ds,
            sum_nb, inputs['is_multiple_instances'], inputs['M_instances'])
        
        # scaling here
        J_scl, rhs_scl = matrix_scaling_individual(inputs, J, rhs)

        # solve the linear loop with the exact Jacobian matrix for a few iterations
        phi_new = [None]*3
        for p in [0,1,2]:
            if inputs['solver_options']['ion_only'] and p != 2:
                phi_new[p] = phi[p]
                continue
            phi_new[p], _ = linalg.gmres(J_scl[p], rhs_scl[p], x0=phi[p], maxiter=inputs['solver_options']['max_iter_lin'], tol=1e-20)

        
        # error monitoring here
        max_res, residuals = error_monitoring_individual(inputs, phi, phi_new, J, rhs, residuals, iter)
        
        # updating the solution here
        phi = update_phi_individual(phi, phi_new, inputs['solver_options']['uf'])

        # check the convergence
        if max_res < inputs['solver_options']['tol'] and iter>5:
            tol = inputs['solver_options']['tol']
            print(f'Newton loop is converged to a tolerance of {tol} in {iter} iterations.')
            break

        iter += 1
    
    return phi, residuals

def update_interior(inputs, J, rhs, indices, field_functions, phi, dx, N, ds, sum_nb, isMi, M_ins=None):
    import numpy as np

    f = [field_functions['f'][0], field_functions['f'][1], field_functions['f'][2]]
    fp = [field_functions['fp'][0], field_functions['fp'][1], field_functions['fp'][2]]

    ratio = [None]*3
    ratio[0] = np.zeros_like(rhs[0])
    ratio[1] = np.zeros_like(rhs[1])
    ratio[2] = np.zeros_like(rhs[2])

    phi_dense = [np.zeros(N)]*3
    phi_dense[0][ds[0]] = phi[0]
    phi_dense[1][ds[1]] = phi[1]
    phi_dense[2][ds[2]] = phi[2]

    if isMi:
        j_max = N[1]-1
        j_seg = j_max//M_ins+1

    for phase in [0,1,2]:
        if inputs['solver_options']['ion_only'] and phase != 2:
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

def matrix_scaling_individual(inputs, J, rhs):
    from scipy.sparse import diags
    # Scaling the Jacobian matrix and RHS vector 
    # Basically same as using Jacobi preconditioner (other preconditioners should also be considered)
    scl_vec = [None]*3
    scl_mat = [None]*3
    J_scl = [None]*3
    rhs_scl = [None]*3

    for p in [0,1,2]:       # Only solve for ion transport, otherwise: for p in [0,1,2]:
        if inputs['solver_options']['ion_only'] and p != 2:
            continue
        scl_vec[p] = 1/J[p].diagonal(0).ravel()
        scl_mat[p] = diags(scl_vec[p])
        J_scl[p] = scl_mat[p] @ J[p]
        rhs_scl[p] = scl_mat[p] @ rhs[p]

    return J_scl, rhs_scl

def error_monitoring_individual(inputs, phi, phi_new, J_scl, rhs_scl, residuals, iter):
    import numpy as np

    res = [None]*len(phi)
    for p in range(len(phi)):
        if inputs['solver_options']['ion_only'] and p != 2:
            residuals[p] = residuals[p] + [10]       # arbitrary large number
            continue
        # res[p] = rhs_scl[p] - J_scl[p].tocsr()@phi_new[p]    # first method
        res[p] = np.abs(phi_new[p] - phi[p])                 # second method
        residuals[p] = residuals[p] + [np.linalg.norm(res[p])/np.linalg.norm(phi_new[p])]

    if len(phi) == 3:
        if inputs['solver_options']['ion_only']:
            max_res = residuals[2][iter]
        else:
            max_res = max(residuals[0][iter], residuals[1][iter], residuals[2][iter])
    elif len(phi) == 5:
        max_res = max(residuals[0][iter], residuals[1][iter], residuals[2][iter], residuals[3][iter], residuals[4][iter])
    
    if len(phi) == 3:
        if iter % 10 == 0:
            print('\nIter:  res cH2:       res Vel:       res Vio:')
            print('-----------------------------------------------')
        if inputs['solver_options']['ion_only']:
            print(f'{iter:<7}Not solved     Not solved     {residuals[2][iter]:<.2e}')
        else:
            print(f'{iter:<7}{residuals[0][iter]:<15.2e}{residuals[1][iter]:<15.2e}{residuals[2][iter]:<.2e}')
    if len(phi) == 5:
        if iter % 10 == 0:
            print('\nIter:  res0:       res1:       res2:       res3:       res4:')
            print('-------------------------------------------------------------------')
        print(f'{iter:<7}{residuals[0][iter]:<12.2e}{residuals[1][iter]:<12.2e}{residuals[2][iter]:<12.2e}{residuals[3][iter]:<12.2e}{residuals[4][iter]:<12.2e}')
    return max_res, residuals

def update_phi_individual(phi, phi_new, uf):
    und_fac = [uf['cH2'], uf['Vel'], uf['Vio']]
    for p in [0,1,2]:
        phi[p] = und_fac[p]*phi_new[p] + (1-und_fac[p])*phi[p]
    
    return phi

# specific functions for the entire cell
def Newton_loop_individual_entire_cell(inputs, J, rhs, phi, indices, field_functions, masks_dict, sum_nb, residuals):
    import numpy as np
    from scipy.sparse import linalg

    dx = inputs['dx']
    N_a = [inputs['Nx_a'], inputs['Ny'], inputs['Nz']]
    N_c = [inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    N_YSZ = [inputs['Nx_a']+inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    N = [N_a, N_YSZ, N_c]
    prev_iters = len(residuals[0])
    iter = prev_iters
    ds = masks_dict['ds']

    while iter < prev_iters+inputs['max_iter_non']:
    
        # update J[n,n] and rhs[b] for interior points (source==True)
        J, rhs = update_interior_entire_cell(
                inputs, J, rhs, indices, 
                field_functions,
                phi, dx, ds,
                sum_nb, inputs['is_multiple_instances'], inputs['M_instances'])
        
        # scaling here
        J_scl, rhs_scl = matrix_scaling_individual_entire_cell(J,rhs)

        # solve the linear loop with the exact Jacobian matrix for a few iterations
        phi_new = [None]*5
        phi_new[1] = phi[1]
        phi_new[4] = phi[4]
        for p in [0,2,3]:  # 1,4 are not solved here - electron transport is assumed to be spontaneous
            phi_new[p], _ = linalg.gmres(J_scl[p], rhs_scl[p], x0=phi[p], maxiter=inputs['max_iter_lin'])
        
        # error monitoring here
        max_res, residuals = error_monitoring_individual(phi, phi_new, J, rhs, residuals, iter)
        
        # updating the solution here
        phi = update_phi_individual(phi, phi_new, inputs['uf'])

        # check the convergence
        # if max_res < inputs['tol'] and iter>5:
        #     break

        iter += 1
    
    return phi, residuals

def update_interior_entire_cell(inputs, J, rhs, indices, field_functions, phi, dx, ds, sum_nb, isMi, M_ins=None):
    import numpy as np

    N_a = [inputs['Nx_a'], inputs['Ny'], inputs['Nz']]
    N_c = [inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    N_YSZ = [inputs['Nx_a']+inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    N = [N_a, N_YSZ, N_c]
    f = [field_functions['f'][0], field_functions['f'][1], field_functions['f'][2], field_functions['f'][3], field_functions['f'][4], field_functions['f'][5]]
    fp = [field_functions['fp'][0], field_functions['fp'][1], field_functions['fp'][2], field_functions['fp'][3], field_functions['fp'][4], field_functions['fp'][5]]

    ratio = [None]*5
    ratio[0] = np.zeros_like(rhs[0])
    ratio[1] = np.zeros_like(rhs[1])
    ratio[2] = np.zeros_like(rhs[2])
    ratio[3] = np.zeros_like(rhs[3])
    ratio[4] = np.zeros_like(rhs[4])

    ds_entire_cell = [None]*5
    ds_entire_cell[0] = np.concatenate((ds[0], np.zeros((inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']),dtype=bool)), axis=0)
    ds_entire_cell[1] = np.concatenate((ds[1], np.zeros((inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']),dtype=bool)), axis=0)
    ds_entire_cell[2] = ds[2]
    ds_entire_cell[3] = np.concatenate((np.zeros((inputs['Nx_a']+inputs['Nx_e'], inputs['Ny'], inputs['Nz']),dtype=bool),ds[3]),axis=0)
    ds_entire_cell[4] = np.concatenate((np.zeros((inputs['Nx_a']+inputs['Nx_e'], inputs['Ny'], inputs['Nz']),dtype=bool),ds[4]),axis=0)
    
    phi_dense = np.zeros(N_YSZ)
    phi_dense[ds_entire_cell[0]] = phi[0]
    phi_dense[ds_entire_cell[1]] = phi[1]
    phi_dense[ds_entire_cell[2]] = phi[2]
    phi_dense[ds_entire_cell[3]] = phi[3]
    phi_dense[ds_entire_cell[4]] = phi[4]


    if isMi:
        j_max = N[1]-1
        j_seg = j_max//M_ins+1

    # microstructure of the anode side
    for phase in [0,2]: # 1 is not necessary - electron transport is assumed to be spontaneous
        p = np.zeros(5)
        mask = [None]*2
        out = [0,1,2]
        out.remove(phase)

        for n in indices[phase]['source']:

            i,j,k = indices[phase]['all_points'][n]
            if i > inputs['Nx_a']: continue
            
            p[phase] = phi[phase][n]

            if isMi:
                if np.mod(j, j_seg)==0 and j!=0 and j!=j_max:
                    phi_block = phi_dense[i-1:i+2,j:j+2,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j:j+2,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j:j+2,k-1:k+2]
                elif np.mod(j, j_seg)==j_seg-1 and j!=0 and j!=j_max:
                    phi_block = phi_dense[i-1:i+2,j-1:j+1,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j-1:j+1,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j-1:j+1,k-1:k+2]
                else:   
                    phi_block = phi_dense[i-1:i+2,j-1:j+2,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
            else:
                phi_block = phi_dense[i-1:i+2,j-1:j+2,k-1:k+2]
                mask[0] = ds_entire_cell[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                mask[1] = ds_entire_cell[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
                
            p[out[0]] = np.average(phi_block[mask[0]])
            p[out[1]] = np.average(phi_block[mask[1]])

            if np.sum(np.isnan(p))>0:
                continue
            ratio[phase][n] = fp[phase](p[0],p[1],p[2])*dx**3 / sum_nb[phase][n]
            J[phase][n,n] = -sum_nb[phase][n] - fp[phase](p[0],p[1],p[2])*dx**3    # aP
            rhs[phase][n] = (f[phase](p[0],p[1],p[2]) - fp[phase](p[0],p[1],p[2])*phi[phase][n]) * dx**3 # RHS

    # microstructure of the cathode side
    for phase in [2,3]: # 4 is not necessary - electron transport is assumed to be spontaneous
        p = np.zeros(5)
        mask = [None]*2
        out = [2,3,4]
        out.remove(phase)

        for n in indices[phase]['source']:

            i,j,k = indices[phase]['all_points'][n]
            if i < inputs['Nx_a']+inputs['Nx_e']: continue
            
            p[phase] = phi[phase][n]

            if isMi:
                if np.mod(j, j_seg)==0 and j!=0 and j!=j_max:
                    phi_block = phi_dense[i-1:i+2,j:j+2,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j:j+2,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j:j+2,k-1:k+2]
                elif np.mod(j, j_seg)==j_seg-1 and j!=0 and j!=j_max:
                    phi_block = phi_dense[i-1:i+2,j-1:j+1,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j-1:j+1,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j-1:j+1,k-1:k+2]
                else:   
                    phi_block = phi_dense[i-1:i+2,j-1:j+2,k-1:k+2]
                    mask[0] = ds[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                    mask[1] = ds[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
            else:
                phi_block = phi_dense[i-1:i+2,j-1:j+2,k-1:k+2]
                mask[0] = ds_entire_cell[out[0]][i-1:i+2,j-1:j+2,k-1:k+2]
                mask[1] = ds_entire_cell[out[1]][i-1:i+2,j-1:j+2,k-1:k+2]
                
            p[out[0]] = np.average(phi_block[mask[0]])
            p[out[1]] = np.average(phi_block[mask[1]])

            if np.sum(np.isnan(p))>0:
                continue

            phase_f = 5 if phase==2 else phase
            ratio[phase][n] = fp[phase_f](p[3],p[4],p[2])*dx**3 / sum_nb[phase][n]
            J[phase][n,n] = -sum_nb[phase][n] - fp[phase_f](p[3],p[4],p[2])*dx**3    # aP
            rhs[phase][n] = (f[phase_f](p[3],p[4],p[2]) - fp[phase_f](p[3],p[4],p[2])*phi[phase][n]) * dx**3 # RHS
    
    return J, rhs

def matrix_scaling_individual_entire_cell(J, rhs):
    from scipy.sparse import diags
    # Scaling the Jacobian matrix and RHS vector 
    # Basically same as using Jacobi preconditioner (other preconditioners should also be considered)
    scl_vec = [0]*5
    scl_mat = [0]*5
    J_scl = [0]*5
    rhs_scl = [0]*5

    for p in [0,2,3]:   # 1,4 are not necessary - electron transport is assumed to be spontaneous
        scl_vec[p] = 1/J[p].diagonal(0).ravel()
        scl_mat[p] = diags(scl_vec[p])
        J_scl[p] = scl_mat[p] @ J[p]
        rhs_scl[p] = scl_mat[p] @ rhs[p]

    return J_scl, rhs_scl