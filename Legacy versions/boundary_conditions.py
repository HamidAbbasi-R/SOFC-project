def Dirichlet(phi, boundary, value):
    """
    Dirichlet boundary condition
    phi(i,j,k)
    West: phi(0,j,k)
    East: phi(Nx-1,j,k)
    South: phi(i,0,k)
    North: phi(i,Ny-1,k)
    Bottom: phi(i,j,0)
    Top: phi(i,j,Nz-1)
    """
    if boundary=='North':
        if phi.ndim==2:
            phi[:,-1] = value
        elif phi.ndim==3:
            phi[:,-1,:] = value
    elif boundary=='South':
        if phi.ndim==2:
            phi[:,0] = value 
        elif phi.ndim==3:
            phi[:,0,:] = value
    elif boundary=='West':
        if phi.ndim==2:
            phi[0,:] = value
        elif phi.ndim==3:
            phi[0,:,:] = value
    elif boundary=='East':
        if phi.ndim==2:
            phi[-1,:] = value
        elif phi.ndim==3:
            phi[-1,:,:] = value
    elif boundary=='Bottom':
        phi[:,:,0] = value
    elif boundary=='Top':
        phi[:,:,-1] = value
    return phi
 
def Neumann(phi, boundary, flux, delx):
    if boundary=='North':
        if phi.ndim==2:
            phi[:,-1] = phi[:,-2] + flux*delx
        elif phi.ndim==3:
            phi[:,-1,:] = phi[:,-2,:] + flux*delx
    elif boundary=='South':
        if phi.ndim==2:
            phi[:,0] = phi[:,1] + flux*delx
        elif phi.ndim==3:
            phi[:,0,:] = phi[:,1,:] + flux*delx
    elif boundary=='West':
        if phi.ndim==2:
            phi[0,:] = phi[1,:] + flux*delx
        elif phi.ndim==3:
            phi[0,:,:] = phi[1,:,:] + flux*delx
    elif boundary=='East':
        if phi.ndim==2:
            phi[-1,:] = phi[-2,:] + flux*delx
        elif phi.ndim==3:
            phi[-1,:,:] = phi[-2,:,:] + flux*delx
    elif boundary=='Bottom':
        phi[:,:,0] = phi[:,:,1] + flux*delx
    elif boundary=='Top':
        phi[:,:,-1] = phi[:,:,-2] + flux*delx
    return phi

def Periodic(phi, boundary, k_faceX, k_faceY, k_faceZ=0, delx=1):
    import numpy as np
    if boundary=='North and South':
        if phi.ndim==2:
            aE = delx*k_faceX[2:-1, 0]
            aW = delx*k_faceX[1:-2, 0]
            aN = delx*k_faceY[1:-1, 1]
            aS = delx*k_faceY[1:-1, -2]
            aP = aE+aW+aN+aS
            aP_inverse = np.divide(1,aP,out=np.zeros_like(aP),where=aP!=0)
            phi[1:-1,0] = aP_inverse*(
                aE*phi[2:  , 0]+
                aW*phi[:-2 , 0]+
                aN*phi[1:-1, 1]+
                aS*phi[1:-1, -2])
            phi[1:-1,-1] = phi[1:-1,0]
        elif phi.ndim==3:
            aE = k_faceX[2:-1 , 0 , 1:-1]
            aW = k_faceX[1:-2 , 0 , 1:-1]
            aN = k_faceY[1:-1 , 1 , 1:-1]
            aS = k_faceY[1:-1 , -2, 1:-1]
            aT = k_faceZ[1:-1 , 0 , 2:-1]
            aB = k_faceZ[1:-1 , 0 , 1:-2]
            aP = aE+aW+aN+aS+aT+aB
            aP_inverse = np.divide(1,aP,out=np.zeros_like(aP),where=aP!=0)
            phi[1:-1,0,1:-1] = aP_inverse*(
                aE*phi[2:  , 0 , 1:-1]+
                aW*phi[:-2 , 0 , 1:-1]+
                aN*phi[1:-1, 1 , 1:-1]+
                aS*phi[1:-1, -2, 1:-1]+
                aT*phi[1:-1, 0 , 2:  ]+
                aB*phi[1:-1, 0 , :-2 ])
            phi[1:-1,-1,1:-1] = phi[1:-1,0,1:-1]
    elif boundary=='West and East':
        if phi.ndim==2:
            aE = delx*k_faceX[1 , 1:-1]
            aW = delx*k_faceX[-2, 1:-1]
            aN = delx*k_faceY[0 , 2:-1]
            aS = delx*k_faceY[0 , 1:-2]
            aP = aE+aW+aN+aS
            aP_inverse = np.divide(1,aP,out=np.zeros_like(aP),where=aP!=0)
            phi[0,1:-1] = aP_inverse*(
                aE*phi[1 , 1:-1]+
                aW*phi[-2, 1:-1]+
                aN*phi[0 , 2:  ]+
                aS*phi[0 , :-2 ])
            phi[-1,1:-1] = phi[0,1:-1]
        elif phi.ndim==3:
            aE = k_faceX[1 , 1:-1 , 1:-1]
            aW = k_faceX[-2, 1:-1 , 1:-1]
            aN = k_faceY[0 , 2:-1 , 1:-1]
            aS = k_faceY[0 , 1:-2 , 1:-1]
            aT = k_faceZ[0 , 1:-1 , 2:-1]
            aB = k_faceZ[0 , 1:-1 , 1:-2]
            aP = aE+aW+aN+aS+aT+aB
            aP_inverse = np.divide(1,aP,out=np.zeros_like(aP),where=aP!=0)
            phi[0,1:-1,1:-1] = aP_inverse*(
                aE*phi[1 , 1:-1 , 1:-1]+
                aW*phi[-2, 1:-1 , 1:-1]+
                aN*phi[0 , 2:   , 1:-1]+
                aS*phi[0 , :-2  , 1:-1]+
                aT*phi[0 , 1:-1 , 2:  ]+
                aB*phi[0 , 1:-1 , :-2 ])
            phi[-1,1:-1,1:-1] = phi[0,1:-1,1:-1]
    elif boundary=='Bottom and Top':
        aE = k_faceX[2:-1 , 1:-1 , 0 ]
        aW = k_faceX[1:-2 , 1:-1 , 0 ]
        aN = k_faceY[1:-1 , 2:-1 , 0 ]
        aS = k_faceY[1:-1 , 1:-2 , -2]
        aT = k_faceZ[1:-1 , 1:-1 , 1 ]
        aB = k_faceZ[1:-1 , 1:-1 , -2]
        aP = aE+aW+aN+aS+aT+aB
        aP_inverse = np.divide(1,aP,out=np.zeros_like(aP),where=aP!=0)
        phi[1:-1,1:-1,0] = aP_inverse*(
            aE*phi[2:  , 1:-1 , 0 ]+
            aW*phi[:-2 , 1:-1 , 0 ]+
            aN*phi[1:-1, 2:   , 0 ]+
            aS*phi[1:-1, :-2  , 0 ]+
            aT*phi[1:-1, 1:-1 , 1 ]+
            aB*phi[1:-1, 1:-1 , -2])
        phi[1:-1,1:-1,-1] = phi[1:-1,1:-1,0]
    return phi