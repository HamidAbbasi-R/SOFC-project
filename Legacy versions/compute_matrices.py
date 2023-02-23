def boundaries_all(
    J,      # coefficient matrix
    rhs,    # right hand side vector
    bc,     # boundary condition dictionary
    dx,     # grid spacing
    indices,  # indice information (ip, jp, kp)
    N,      # number of grid points
    ds,     # boolean matrix for each phase's domain
    K,      # conductivity 
    phase_num,
    ):

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

def boundaries(inputs, bc, indices):
    import numpy as np
    from scipy.sparse import lil_matrix
    K = [inputs['K_pores'], inputs['K_Ni'], inputs['K_YSZ']]
    dx = inputs['dx']

    # initializing left hand side and right hand side of the system of equaitons
    L = len(indices['all_points'])
    rhs = np.zeros(shape = L, dtype = float) # right hand side vector
    sum_nb = np.zeros(shape = L, dtype = float) # sigma a_nb vector

    # rhs = lil_matrix((1,L), dtype = float)
    J = lil_matrix((L,L), dtype = float) # sparse matrix

    # west side (i=0)
    if bc['West'][0] == 'Dirichlet':
        for n in indices['west_bound']:
            J[n,n] = 1*K*dx     # aP
            rhs[n] = bc['West'][1]*K*dx
    elif bc['West'][0] == 'Neumann':
        for n in indices['west_bound']:
            J[n,n] = 1*K*dx     # aP
            J[n,indices['east_nb'][n]] = -1*K*dx   # aE
            rhs[n] = bc['West'][1]*dx**2


    # east side (i=N[0]-1)
    if bc['East'][0] == 'Dirichlet':
        for n in indices['east_bound']:
            J[n,n] = 1*K*dx     # aP
            rhs[n] = bc['East'][1]*K*dx
    elif bc['East'][0] == 'Neumann':
        for n in indices['east_bound']:
            J[n,n] = 1*K*dx       # aP
            J[n,indices['west_nb'][n]] = -1   # aW
            rhs[n] = bc['East'][1]*dx**2


    # south side (j=0)
    if bc['South'][0] == 'Dirichlet':
        for n in indices['south_bound']:
            J[n,n] = 1*K*dx    # aP
            rhs[n] = bc['South'][1]*K*dx
    elif bc['South'][0] == 'Neumann':
        for n in indices['south_bound']:
            J[n,n] = 1*K*dx    # aP
            J[n,indices['north_nb'][n]] = -1   # aN
            rhs[n] = bc['South'][1]*dx**2

    # north side (j=N[1]-1)
    if bc['North'][0] == 'Dirichlet':
        for n in indices['north_bound']:
            J[n,n] = 1*K*dx     # aP
            rhs[n] = bc['North'][1]*K*dx
    elif bc['North'][0] == 'Neumann':   
        for n in indices['north_bound']:
            J[n,n] = -1*K*dx     # aP
            J[n,indices['south_nb'][n]] = -1*K*dx   # aS
            rhs[n] = bc['North'][1]*dx**2

    # bottom side (k=0)
    if bc['Bottom'][0] == 'Dirichlet':
        for n in indices['bottom_bound']:
            J[n,n] = 1*K*dx   # aP
            rhs[n] = bc['Bottom'][1]*K*dx
    elif bc['Bottom'][0] == 'Neumann':
        for n in indices['bottom_bound']:
            J[n,n] = 1*K*dx  # aP
            J[n,indices['top_nb'][n]] = -1*K*dx   # aT
            rhs[n] = bc['Bottom'][1]*dx**2


    # top side (k=N[2]-1)
    if bc['Top'][0] == 'Dirichlet':
        for n in indices['top_bound']:
            J[n,n] = 1*K*dx  # aP
            rhs[n] = bc['Top'][1]*K*dx
    elif bc['Top'][0] == 'Neumann':
        for n in indices['top_bound']:
            J[n,n] = 1*K*dx
            J[n,indices['bottom_nb'][n]] = -1*K*dx   # aB
            rhs[n] = bc['Top'][1]*dx**2

    print('Done')

    return J, rhs, sum_nb

def interior(inputs, J, indices, K, ds, sum_nb):
    dx = inputs['dx']
    for n in indices['interior']:
        i,j,k = indices['all_points'][n]
        
        aW, aE, aS, aN, aB, aT = \
            -K*ds[i-1,j,k]*dx, -K*ds[i+1,j,k]*dx, -K*ds[i,j-1,k]*dx, -K*ds[i,j+1,k]*dx, -K*ds[i,j,k-1]*dx, -K*ds[i,j,k+1]*dx
        
        # assign a_nb values for all interior points 
        J[n,indices['west_nb'][n]] = aW
        J[n,indices['east_nb'][n]] = aE
        J[n,indices['south_nb'][n]] = aS
        J[n,indices['north_nb'][n]] = aN
        J[n,indices['bottom_nb'][n]] = aB
        J[n,indices['top_nb'][n]] = aT
        sum_nb[n] = aW + aE + aS + aN + aB + aT
        J[n,n] = -sum_nb[n]
    
    return J, sum_nb

def update_interior(J, rhs, indices, f, fp, phi, dx, N, ds, sum_nb):

    import numpy as np

    J0, J1, J2 = J
    rhs0, rhs1, rhs2 = rhs
    indices0, indices1, indices2 = indices
    f0, f1, f2 = f
    fp0, fp1, fp2 = fp
    phi0, phi1, phi2 = phi 
    ds0, ds1, ds2 = ds
    sum_nb0, sum_nb1, sum_nb2 = sum_nb

    phi_0_dense = np.zeros(N)
    phi_1_dense = np.zeros(N)
    phi_2_dense = np.zeros(N)
    phi_0_dense[ds0] = phi0
    phi_1_dense[ds1] = phi1
    phi_2_dense[ds2] = phi2

    for n in indices0['source']:

        i,j,k = indices0['all_points'][n]

        p0 = phi0[n]
        p1 = np.average(phi_1_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds1[i-1:i+2,j-1:j+2,k-1:k+2]])
        p2 = np.average(phi_2_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds2[i-1:i+2,j-1:j+2,k-1:k+2]])

        J0[n,n] = -(sum_nb0[n] - fp0(p0,p1,p2)*dx**3)    # aP
        rhs0[n] = -((f0(p0,p1,p2) - fp0(p0,p1,p2)*phi0[n]) * dx**3) # RHS

    for n in indices1['source']:

        i,j,k = indices1['all_points'][n]

        p0 = np.average(phi_0_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds0[i-1:i+2,j-1:j+2,k-1:k+2]])
        p1 = phi1[n]
        p2 = np.average(phi_2_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds2[i-1:i+2,j-1:j+2,k-1:k+2]])

        J1[n,n] = -(sum_nb1[n] - fp1(p0,p1,p2)*dx**3)    # aP
        rhs1[n] = -((f1(p0,p1,p2) - fp1(p0,p1,p2)*phi1[n]) * dx**3)

    for n in indices2['source']:

        i,j,k = indices2['all_points'][n]

        p0 = np.average(phi_0_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds0[i-1:i+2,j-1:j+2,k-1:k+2]])
        p1 = np.average(phi_1_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds1[i-1:i+2,j-1:j+2,k-1:k+2]])
        p2 = phi2[n]

        J2[n,n] = -(sum_nb2[n] - fp2(p0,p1,p2)*dx**3)    # aP
        rhs2[n] = -((f2(p0,p1,p2) - fp2(p0,p1,p2)*phi2[n]) * dx**3)
    
    J = [J0, J1, J2]
    rhs = [rhs0, rhs1, rhs2]
    sum_nb = [sum_nb0, sum_nb1, sum_nb2]
    return J, rhs, sum_nb