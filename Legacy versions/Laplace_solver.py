def Laplace_2D(domain, source_mask, source_func, bc_dict, 
ic, delx=1, iters=1000, orf=1.0):
    """
    This function solves the Laplace equation for a 2D domain with variable conductivity.
    The boundary conditions are specified by the user. The initial condition is also specified.
    
    inputs:
        domain: a numpy matrix of the domain to be solved
        source_mat: a numpy matrix of the source (same size as domain) [W/m^3]
        bc: a dictionary of the boundary conditions
        ic: a numpy array of the initial condition
        delx: the size of the phi spacing. it is assumed that
            delx is equal to dely and delz (default 1) [m]
        iters: an integer of the number of iterations to be performed (default 1000)
        display: a list of the display options (default []) 
            options: 'heatmap', 'contour', 'residual'
    
    outputs:
        phi: a numpy array of the solution to the Laplace equation
        err: a numpy array of the error at each iteration
    """
    import numpy as np
    from tqdm import tqdm

    # constants
    N = domain.shape         # phi size    

    # conductivity matrix
    cond_mat = np.copy(domain).astype(float)

    # create boundary and interior masks
    import numpy as np
    edge_mask = np.copy(domain).astype(bool)
    edge_mask[1:-1,1:-1] = True

    # conductivity matrix [W/mK]
    k = cond_mat

    # initial condition
    phi = np.ones_like(domain,dtype=float)*ic*domain

    # initialize error array
    err = []

    # initialize cnductivity at phi interfaces in x and y directions
    k_faceX = np.zeros([N[0]+1, N[1]  ])
    k_faceY = np.zeros([N[0]  , N[1]+1])

    # initialize neighbor coefficients
    aE = np.zeros_like(k_faceX)
    aW = np.zeros_like(k_faceX)
    aN = np.zeros_like(k_faceY)
    aS = np.zeros_like(k_faceY)

    # compute conductivity at phi interfaces in x and y directions [W/mK]
    k1 = k[0:-1 ,:]
    k2 = k[1:   ,:]
    k_faceX[1:-1,:] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)
        
    k1 = k[:, 0:-1]
    k2 = k[:, 1:  ]
    k_faceY[:,1:-1] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)

    # neighbor coefficients for interior points
    aE = delx*k_faceX[2:-1, 1:-1]
    aW = delx*k_faceX[1:-2, 1:-1]
    aN = delx*k_faceY[1:-1, 2:-1]
    aS = delx*k_faceY[1:-1, 1:-2]

    # source term treatment
    N_source = np.sum(source_mask[1:-1,1:-1])
    from sympy import lambdify
    from sympy.abc import x
    from sympy import diff
    f = lambdify(x, source_func)
    fp = lambdify(x, diff(source_func, x))

    phi_updated = np.copy(phi)
    import boundary_conditions as bc

    for n in tqdm(range(iters)):
        # source term linearlization
        sP = fp(phi[1:-1,1:-1])*source_mask[1:-1,1:-1]
        if np.sum(sP>0)/N_source > 0.2:
            print('Warning: SP in source term is negative')
            
        aP = aE+aW+aN+aS - sP*delx**2
        aP_inverse = np.divide(1, aP, out=np.zeros_like(aP), where=aP!=0)

        sC = (f(phi[1:-1,1:-1])*source_mask[1:-1,1:-1] -
            fp(phi[1:-1,1:-1])*phi[1:-1,1:-1]*source_mask[1:-1,1:-1])
        b = sC*delx**2

        # updating the solution matrix: interior points
        phi_updated[1:-1,1:-1] =aP_inverse*(
            aE*phi[2:,   1:-1]+
            aW*phi[0:-2, 1:-1]+
            aN*phi[1:-1, 2:  ]+
            aS*phi[1:-1, 0:-2]+
            b)
        
        # Setting boundary conditions
        for boundary in list(bc_dict):
            if bc_dict[boundary][0]=="Dirichlet":
                phi_updated = bc.Dirichlet(
                    phi_updated, boundary, bc_dict[boundary][1])
            elif bc_dict[boundary][0]=="Neumann":
                phi_updated = bc.Neumann(
                    phi_updated, boundary, bc_dict[boundary][1], delx)
            elif bc_dict[boundary][0]=="Periodic":
                phi_updated = bc.Periodic(
                    phi_updated, boundary, k_faceX, k_faceY)
        
        # updating the boundary values
        phi_updated = phi_updated*edge_mask

        # error calculation [to reduce runtime, once every ten iterations]
        if n%10==9:
            # err.append(np.mean((phi[domain==1]-phi_updated[domain==1])**2))

            # error calculation [another method based on continuity]
            dif = aP*phi_updated[1:-1, 1:-1] - (
                aE*phi_updated[2:   , 1:-1 ]+
                aW*phi_updated[0:-2 , 1:-1 ]+
                aN*phi_updated[1:-1 , 2:   ]+
                aS*phi_updated[1:-1 , 0:-2 ]+
                b)
            
            res = np.abs(dif) / np.max(np.abs(aP*phi_updated[1:-1, 1:-1]))
            max_res = np.max(res[domain[1:-1,1:-1]==1])
            err.append(max_res)

        # overrelaxation
        phi = orf*phi_updated + (1-orf)*phi

    # removing the phi points outside the domain for visualization purposes
    phi[domain==False] = np.nan
    
    return phi, err


def Laplace_3D(domain, conductivity, source_mask, source_func, bc_dict, 
ic, delx=1, iters=1000, orf=1):
    import numpy as np

    # number of computational nodes (voxels) in each direction
    N = domain.shape

    # Initial condition of solution matrix
    phi = domain.astype(float)*ic

    # detecting the edges of the microstructure [useful for assigning boundary conditions]
    edge_mask = np.copy(domain).astype(bool)
    edge_mask[1:-1,1:-1,1:-1] = True

    # conductivity matrix [w/mK]
    k = np.copy(domain).astype(float)*conductivity

    # defining neighboring matrices
    k_faceX = np.zeros([N[0]+1 ,N[1]   ,N[2]  ])
    k_faceY = np.zeros([N[0]   ,N[1]+1 ,N[2]  ])
    k_faceZ = np.zeros([N[0]   ,N[1]   ,N[2]+1])
    aE = np.zeros_like(k_faceX)
    aW = np.zeros_like(k_faceX)
    aN = np.zeros_like(k_faceY)
    aS = np.zeros_like(k_faceY)
    aT = np.zeros_like(k_faceZ)
    aB = np.zeros_like(k_faceZ)

    k1 = k[0:-1 , : , :]
    k2 = k[1:   , : , :]
    k_faceX[1:-1 , : , :] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)
        
    k1 = k[: , 0:-1 , :]
    k2 = k[: , 1:   , :]
    k_faceY[: , 1:-1 , :] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)
        
    k1 = k[: , : , 0:-1]
    k2 = k[: , : , 1:  ]
    k_faceZ[: , : , 1:-1] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)

    # neighbor coefficients for interior points
    aE = k_faceX[2:-1 , 1:-1 , 1:-1]
    aW = k_faceX[1:-2 , 1:-1 , 1:-1]
    aN = k_faceY[1:-1 , 2:-1 , 1:-1]
    aS = k_faceY[1:-1 , 1:-2 , 1:-1]
    aT = k_faceZ[1:-1 , 1:-1 , 2:-1]
    aB = k_faceZ[1:-1 , 1:-1 , 1:-2]

    # source term linearization
    N_source = np.sum(source_mask[1:-1,1:-1,1:-1])
    from sympy import lambdify
    from sympy.abc import x
    from sympy import diff
    f = lambdify(x, source_func)
    fp = lambdify(x, diff(source_func, x))

    err = []
    max_res = 1
    phi_updated = np.copy(phi)

    # iterative solution of Laplace equation
    import boundary_conditions as bc
    print("\nIterating the Jacobi solver...")
    for n in range(iters):
        # source term linearlization
        sP = fp(phi[1:-1,1:-1,1:-1])*source_mask[1:-1,1:-1,1:-1]
        aP = aE+aW+aN+aS+aT+aB - sP*delx**3
        aP_inverse = np.divide(1, aP, out=np.zeros_like(aP), where=aP!=0)

        sC = (f(phi[1:-1,1:-1,1:-1]) - fp(phi[1:-1,1:-1,1:-1])*phi[1:-1,1:-1,1:-1])*\
            source_mask[1:-1,1:-1,1:-1]
        b = sC*delx**3

        #updating the solution matrix
        phi_updated[1:-1 , 1:-1 , 1:-1] = aP_inverse*(
            aE*phi[2:   , 1:-1 , 1:-1]+
            aW*phi[0:-2 , 1:-1 , 1:-1]+
            aN*phi[1:-1 , 2:   , 1:-1]+
            aS*phi[1:-1 , 0:-2 , 1:-1]+
            aT*phi[1:-1 , 1:-1 , 2:  ]+
            aB*phi[1:-1 , 1:-1 , 0:-2]+
            b)
        
        # Setting boundary conditions
        for boundary in list(bc_dict):
            if bc_dict[boundary][0]=="Dirichlet":
                phi_updated = bc.Dirichlet(
                    phi_updated, boundary, bc_dict[boundary][1])
            elif bc_dict[boundary][0]=="Neumann":
                phi_updated = bc.Neumann(
                    phi_updated, boundary, bc_dict[boundary][1], delx)
            elif bc_dict[boundary][0]=="Periodic":
                phi_updated = bc.Periodic(
                    phi_updated, boundary, k_faceX, k_faceY, k_faceZ)
        
        # updating the boundary values
        phi_updated *= domain==True

        # error calculation [to reduce runtime, once every ten iterations]
        if (n+1)%10==0:
            # err.append(np.mean((phi[domain==1] - phi_updated[domain==1])**2))
            
            dif = aP*phi_updated[1:-1, 1:-1, 1:-1] - (
                aE*phi_updated[2:   , 1:-1 , 1:-1]+
                aW*phi_updated[0:-2 , 1:-1 , 1:-1]+
                aN*phi_updated[1:-1 , 2:   , 1:-1]+
                aS*phi_updated[1:-1 , 0:-2 , 1:-1]+
                aT*phi_updated[1:-1 , 1:-1 , 2:  ]+
                aB*phi_updated[1:-1 , 1:-1 , 0:-2]+
                b)
            
            res = np.abs(dif) / np.max(np.abs(aP*phi_updated[1:-1, 1:-1, 1:-1]))
            max_res = np.max(res[domain[1:-1, 1:-1, 1:-1]==True])
            err.append(max_res)

        # printing in terminal
        print('iter={}\t\tres={:.2e}\t\t(sP>0)=%{}'.format(
            n+1,max_res,100*(np.sum(sP>0)/N_source)))

        # explicit relaxation
        phi = orf*phi_updated + (1-orf)*phi
        
    # removing the phi points outside the domain for visualization purposes
    phi[domain==False] = np.nan
    
    return phi, err

def Laplace_3D_3phase(domain, conductivity, source_mask, source_func, bc_dict, 
ic, delx=1, iters=1000, orf=1):
    import numpy as np

    # number of computational nodes (voxels) in each direction
    N = domain.shape

    # elements in domain are (0,1,2)
    domain -= 1

    # Initial condition of solution matrix
    phi = np.ones(shape=(N[0],N[1],N[2],3),dtype=float)
    phi[:,:,:,0][domain==0] = ic[0]
    phi[:,:,:,1][domain==1] = ic[1]
    phi[:,:,:,2][domain==2] = ic[2]

    # conductivity matrix
    k = np.zeros(shape=(N[0],N[1],N[2],3), dtype=float)
    k[:,:,:,0][domain==0] = conductivity[0]
    k[:,:,:,1][domain==1] = conductivity[1]
    k[:,:,:,2][domain==2] = conductivity[2]

    # defining neighboring matrices
    k_faceX = np.zeros([N[0]+1 ,N[1]   ,N[2]  , 3])
    k_faceY = np.zeros([N[0]   ,N[1]+1 ,N[2]  , 3])
    k_faceZ = np.zeros([N[0]   ,N[1]   ,N[2]+1, 3])
    aE = np.zeros_like(k_faceX)
    aW = np.zeros_like(k_faceX)
    aN = np.zeros_like(k_faceY)
    aS = np.zeros_like(k_faceY)
    aT = np.zeros_like(k_faceZ)
    aB = np.zeros_like(k_faceZ)

    k1 = k[0:-1 , : , :, :]
    k2 = k[1:   , : , :, :]
    k_faceX[1:-1 , : , :, :] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)
        
    k1 = k[: , 0:-1 , :, :]
    k2 = k[: , 1:   , :, :]
    k_faceY[: , 1:-1 , :, :] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)
        
    k1 = k[: , : , 0:-1, :]
    k2 = k[: , : , 1:  , :]
    k_faceZ[: , : , 1:-1, :] = 2*np.divide(
        k1*k2, k1+k2, out = np.zeros_like(k1), where = k1+k2!=0)

    # neighbor coefficients for interior points
    aE = k_faceX[2:-1 , 1:-1 , 1:-1, :]
    aW = k_faceX[1:-2 , 1:-1 , 1:-1, :]
    aN = k_faceY[1:-1 , 2:-1 , 1:-1, :]
    aS = k_faceY[1:-1 , 1:-2 , 1:-1, :]
    aT = k_faceZ[1:-1 , 1:-1 , 2:-1, :]
    aB = k_faceZ[1:-1 , 1:-1 , 1:-2, :]

    # source term treatment
    N_source = np.sum(source_mask[1:-1,1:-1,1:-1])
    from sympy import lambdify
    from sympy import symbols
    from sympy import diff
    
    # defining field variables
    phi0 = symbols('phi0')
    phi1 = symbols('phi1')
    phi2 = symbols('phi2')

    f0 = lambdify([phi0, phi1, phi2], source_func[0])
    f1 = lambdify([phi0, phi1, phi2], source_func[1])
    f2 = lambdify([phi0, phi1, phi2], source_func[2])
    
    fp0 = lambdify([phi0, phi1, phi2], diff(source_func[0], phi0))
    fp1 = lambdify([phi0, phi1, phi2], diff(source_func[1], phi1))
    fp2 = lambdify([phi0, phi1, phi2], diff(source_func[2], phi2))

    # initializing the error array
    err0 = []
    err1 = []
    err2 = []
    max_res_0 = 1
    max_res_1 = 1
    max_res_2 = 1

    phi_updated = np.copy(phi)

    # iterative solution of Laplace equation
    import boundary_conditions as bc
    print("\nIterating the Jacobi solver...")

    for n in range(iters):
        # source term linearlization
        # sP terms are incorrect. in the current implementation, fp0(phi0,phi1,phi2) is
        # basically equal to fp0(phi0,0,0). because it is multiplied by domain[1:-1,1:-1,1:-1]==0.
        # this should be fixed.
        sP0 = fp0(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2])*\
            source_mask[1:-1,1:-1,1:-1] * (domain[1:-1,1:-1,1:-1]==0)
        sP1 = fp1(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2])*\
            source_mask[1:-1,1:-1,1:-1] * (domain[1:-1,1:-1,1:-1]==1)
        sP2 = fp2(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2])*\
            source_mask[1:-1,1:-1,1:-1] * (domain[1:-1,1:-1,1:-1]==2)
        
        aP = aE+aW+aN+aS+aT+aB - np.stack((sP0,sP1,sP2), axis=-1)*delx**3
        aP_inverse = np.divide(1, aP, out=np.zeros_like(aP), where=aP!=0)

        sC0 = (f0(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2]) - \
            fp0(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2])*phi[1:-1,1:-1,1:-1,0]) * \
            source_mask[1:-1,1:-1,1:-1] * (domain[1:-1,1:-1,1:-1]==0)
        sC1 = (f1(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2]) - \
            fp1(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2])*phi[1:-1,1:-1,1:-1,1]) * \
            source_mask[1:-1,1:-1,1:-1] * (domain[1:-1,1:-1,1:-1]==1)
        sC2 = (f2(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2]) - \
            fp2(phi[1:-1,1:-1,1:-1,0],phi[1:-1,1:-1,1:-1,1],phi[1:-1,1:-1,1:-1,2])*phi[1:-1,1:-1,1:-1,2]) * \
            source_mask[1:-1,1:-1,1:-1] * (domain[1:-1,1:-1,1:-1]==2)
        b = np.stack((sC0,sC1,sC2), axis=-1)*delx**3

        #updating the solution matrix
        phi_updated[1:-1 , 1:-1 , 1:-1,:] = aP_inverse*(
            aE*phi[2:   , 1:-1 , 1:-1, :]+
            aW*phi[0:-2 , 1:-1 , 1:-1, :]+
            aN*phi[1:-1 , 2:   , 1:-1, :]+
            aS*phi[1:-1 , 0:-2 , 1:-1, :]+
            aT*phi[1:-1 , 1:-1 , 2:  , :]+
            aB*phi[1:-1 , 1:-1 , 0:-2, :]+
            b)
        
        # Setting boundary conditions
        for i in range(3):
            for boundary in list(bc_dict[i]):
                if bc_dict[i][boundary][0]=="Dirichlet":
                    phi_updated[:,:,:,i] = bc.Dirichlet(
                        phi_updated[:,:,:,i], boundary, bc_dict[i][boundary][1])
                elif bc_dict[i][boundary][0]=="Neumann":
                    phi_updated[:,:,:,i] = bc.Neumann(
                        phi_updated[:,:,:,i], boundary, bc_dict[boundary][1], delx)
                elif bc_dict[i][boundary][0]=="Periodic":
                    phi_updated[:,:,:,i] = bc.Periodic(
                        phi_updated[:,:,:,i], boundary, k_faceX, k_faceY, k_faceZ)
        
        # updating the boundary values
        phi_updated[:,:,:,0] *= domain==0
        phi_updated[:,:,:,1] *= domain==1
        phi_updated[:,:,:,2] *= domain==2
        
        # error calculation [to reduce runtime, once every ten iterations]
        if (n+1)%10==0:
            # err0.append(np.mean((phi[:,:,:,0][domain==0] - phi_updated[:,:,:,0][domain==0])**2))
            # err1.append(np.mean((phi[:,:,:,1][domain==1] - phi_updated[:,:,:,1][domain==1])**2))
            # err2.append(np.mean((phi[:,:,:,2][domain==2] - phi_updated[:,:,:,2][domain==2])**2))
            
            dif = aP*phi_updated[1:-1, 1:-1, 1:-1, :] - (
                aE*phi_updated[2:   , 1:-1 , 1:-1, :]+
                aW*phi_updated[0:-2 , 1:-1 , 1:-1, :]+
                aN*phi_updated[1:-1 , 2:   , 1:-1, :]+
                aS*phi_updated[1:-1 , 0:-2 , 1:-1, :]+
                aT*phi_updated[1:-1 , 1:-1 , 2:  , :]+
                aB*phi_updated[1:-1 , 1:-1 , 0:-2, :]+
                b)
            
            res0 = np.abs(dif[:,:,:,0]) / np.max(np.abs(aP[:,:,:,0]*phi_updated[1:-1, 1:-1, 1:-1, 0]))
            res1 = np.abs(dif[:,:,:,1]) / np.max(np.abs(aP[:,:,:,1]*phi_updated[1:-1, 1:-1, 1:-1, 1]))
            res2 = np.abs(dif[:,:,:,2]) / np.max(np.abs(aP[:,:,:,2]*phi_updated[1:-1, 1:-1, 1:-1, 2]))

            max_res_0 = np.max(res0[domain[1:-1 , 1:-1 , 1:-1]==0])
            max_res_1 = np.max(res1[domain[1:-1 , 1:-1 , 1:-1]==1])
            max_res_2 = np.max(res2[domain[1:-1 , 1:-1 , 1:-1]==2])

            err0.append(max_res_0)
            err1.append(max_res_1)
            err2.append(max_res_2)

        # printing in terminal
        print('iter={}\t\tres0={:.2e}\t\tres1={:.2e}\t\tres2={:.2e}'.format(
            n+1,max_res_0,max_res_1,max_res_2))

        # explicit relaxation
        phi = orf*phi_updated + (1-orf)*phi
        
    # removing the phi points outside the domain for visualization purposes
    phi[:,:,:,0][domain!=0] = np.nan
    phi[:,:,:,1][domain!=1] = np.nan
    phi[:,:,:,2][domain!=2] = np.nan
    
    err = [err0, err1, err2]
    
    return phi, err