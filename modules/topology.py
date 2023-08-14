import numpy as np
def create_microstructure_plurigaussian(
        voxels, 
        vol_frac, 
        d_ave, 
        dx,
        mode = "normal", 
        seed = [], 
        gradient_factor = 1,
        periodic = True, 
        display = False, 
        histogram = 'none'):
    """
    This function creates a periodic three phase 2D phase matrix with the given parameters.
    inputs:
    voxels: number of voxels along each axis
    vol_frac: Array of size (2,1) denoting the volume fraction of the first and the 
        second phase. In case of two phase porous media, size of this array is (1,1)
        Volume fraction of the third phase is calculated accordingly. 
        Convention used to label different phases:
        1: ion conductor
        2: pores
        3: electron conductor
    dim: number of dimensions, either 2 or 3
    sigma: variable in Gaussian filter
    mode: microstructure mode, either "normal" or "thin film"
    display: boolean, whether to display the phase matrix
    histogram: type of histogram to display, either "1D", "2D", or "none"

    outputs:
    phase_mat: three phase phase matrix
    """
    # Apparently, skimage has some issues with being called from matlab.
    # This package is used herer to apply Gaussian filter on the random matrices.
    # Instead of using skimage, Gaussian filter implemented in scipy package can be used
    # to circumvent this issue. scipy.ndimage.gaussian_filter
    # import skimage
    from scipy.ndimage import gaussian_filter

    # calculate sigma based on average diameter data
    # This factor should be between 2 and 3
    # it comes from the Gaussian probability density function 
    C = 2.2     
    if type(d_ave) == int or type(d_ave) == float:
        sigma = d_ave/C/dx
    elif type(d_ave) == list:
        sigma = [None]*2
        sigma[0] = d_ave[0]/C/dx
        sigma[1] = d_ave[1]/C/dx

    if sum(vol_frac) - 1 > 1e-2:
        raise NameError('Error! sum of volume fractions must be equal to 1')
    else:
        vol_frac = vol_frac[:-1]

    dim=len(voxels)
    
    Nx = voxels[0]-1 if periodic else voxels[0]
    Ny = voxels[1]-1 if periodic else voxels[1]
    if dim==3:
        Nz = voxels[2]-1 if periodic else voxels[2]
    
    # create the random generator with specified seed
    rng_1 = np.random.default_rng() if len(seed) == 0 else np.random.default_rng(seed=seed[0])
    rng_2 = np.random.default_rng() if len(seed) == 0 else np.random.default_rng(seed=seed[1])

    # create two random matrices (two matrices are necessary
    # for three phase porous media)
    voxels = np.array(voxels)
    dims = (Nx,Ny,Nz) if dim==3 else (Nx,Ny)
    rand_mat_1 = rng_1.random(size=dims)
    rand_mat_2 = rng_2.random(size=dims)

    # create a tile matrix, according to the dimension
    tile_mat = tuple(np.ones(dim,dtype=int)*3) if periodic else tuple(np.ones(dim,dtype=int)*1)

    # apply the Gaussian filter on the tiled random matrices. tiled random matrices
    # are necessary to create a periodic phase matrix
    if type(sigma) == int or type(sigma) == float:
        sigma = np.array([sigma, sigma])
    elif type(sigma) == list:
        sigma = np.array(sigma)

    smooth_mat_1 = gaussian_filter(
        np.tile(rand_mat_1,tile_mat),
        sigma=sigma[0], 
        mode='reflect')
    smooth_mat_2 = gaussian_filter(
        np.tile(rand_mat_2,tile_mat),
        sigma=sigma[1], 
        mode='reflect')

    # extract center matrices from the tiled smooth matrices
    if dim==2 and periodic:
        smooth_mat_1 = smooth_mat_1[Nx:2*Nx+1,Ny:2*Ny+1]
        smooth_mat_2 = smooth_mat_2[Nx:2*Nx+1,Ny:2*Ny+1]
    elif dim==3 and periodic:
        smooth_mat_1 = smooth_mat_1[Nx:2*Nx+1,Ny:2*Ny+1,Nz:2*Nz+1]
        smooth_mat_2 = smooth_mat_2[Nx:2*Nx+1,Ny:2*Ny+1,Nz:2*Nz+1]

    # create different modes of microstructure with thresholding the smooth matrices
    if mode=="normal":
        if dim==3:
            if len(vol_frac)==2:    # phsaes: 1-2-3
                phase_mat = np.zeros_like(smooth_mat_1, dtype=int)+3
                # for gradient three-phase microstructures it is assumed that pore phase has 
                # a uniform distribution, but volume fraction of Ni and YSZ changes
                vf_p = vol_frac[0]
                vf_Ni_nom = vol_frac[1]
                vf_Ni_min = vf_Ni_nom / gradient_factor
                vf_Ni_max = vf_Ni_nom + vf_Ni_nom / gradient_factor * (gradient_factor - 1)
                vf_Ni = np.linspace(vf_Ni_min, vf_Ni_max, voxels[0])
                for i in range(voxels[0]):
                    a = np.quantile(smooth_mat_1[i,:,:].flatten(), q=vf_p)
                    b = np.quantile(smooth_mat_2[i,:,:][smooth_mat_1[i,:,:] >= a].flatten(), q=vf_Ni[i]/(1-vf_p))
                    phase_mat[i,:,:][smooth_mat_1[i,:,:] < a] = 1
                    phase_mat[i,:,:][np.logical_and(smooth_mat_2[i,:,:] < b, smooth_mat_1[i,:,:] > a)] = 2
            if len(vol_frac)==1:    # phases: 1-2
                phase_mat = np.zeros_like(smooth_mat_1, dtype=int)+2
                vf_min = vol_frac[0] / gradient_factor
                vf_max = vol_frac[0] + vol_frac[0] / gradient_factor * (gradient_factor - 1)
                vf = np.linspace(vf_min, vf_max, voxels[0])
                for i in range(voxels[0]):
                    a = np.quantile(smooth_mat_1[i,:,:].flatten(), q=vf[i])
                    phase_mat[i,:,:][smooth_mat_1[i,:,:] < a] = 1
        elif dim==2:
            if len(vol_frac)==2: # phases 1-2-3
                phase_mat = np.zeros_like(smooth_mat_1, dtype=int)+3
                # for gradient three-phase microstructures it is assumed that pore phase has    
                # a uniform distribution, but volume fraction of Ni and YSZ changes
                vf_p = vol_frac[0]
                vf_Ni_nom = vol_frac[1]
                vf_Ni_min = vf_Ni_nom / gradient_factor
                vf_Ni_max = vf_Ni_nom + vf_Ni_nom / gradient_factor * (gradient_factor - 1)
                vf_Ni = np.linspace(vf_Ni_min, vf_Ni_max, voxels[0])
                for i in range(voxels[0]):
                    a = np.quantile(smooth_mat_1[i,:].flatten(), q=vf_p)
                    b = np.quantile(smooth_mat_2[i,:][smooth_mat_1[i,:] >= a].flatten(), q=vf_Ni[i]/(1-vf_p))
                    phase_mat[i,:][smooth_mat_1[i,:] < a] = 1
                    phase_mat[i,:][np.logical_and(smooth_mat_2[i,:] < b, smooth_mat_1[i,:] > a)] = 2
            if len(vol_frac)==1: # phases 1-2
                phase_mat = np.zeros_like(smooth_mat_1, dtype=int)+2
                vf_min = vol_frac[0] / gradient_factor
                vf_max = vol_frac[0] + vol_frac[0] / gradient_factor * (gradient_factor - 1)
                vf = np.linspace(vf_min, vf_max, voxels[0])
                for i in range(voxels[0]):
                    a = np.quantile(smooth_mat_1[i,:].flatten(), q=vf[i])
                    phase_mat[i,:][smooth_mat_1[i,:] < a] = 1
                    
    elif mode=="thin film":
        a = np.quantile(smooth_mat_1.flatten(), q=vol_frac[0])
        b = np.quantile(smooth_mat_1.flatten(), q=1-vol_frac[1])
        phase_mat[smooth_mat_1 < a] = 2
        phase_mat[smooth_mat_1 > b] = 3
    
    # display the phase matrix
    if display==True and dim==2:
        import plotly.express as px
        fig = px.imshow(np.rot90(phase_mat))
        # fig.write_image("fig1.pdf")
        fig.show()
    elif display==True and dim==3:
        # from postprocess import visualize_mesh
        from modules.postprocess import visualize_mesh as vm 
        vm([phase_mat],[(2,3)])

    # display the histogram of the smooth matrices
    if histogram=='1D':
        # fig = px.histogram(smooth_mat_1.flatten())
        # fig.update_traces(xbins=dict( # bins used for histogram
        #     start=0,
        #     end=1,
        #     size=0.001
        # ))
        # fig.show()

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=smooth_mat_2.flatten()))
        fig.add_trace(go.Histogram(x=smooth_mat_2[smooth_mat_1 > a].flatten()))

        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        fig.update_traces(xbins=dict( # bins used for histogram
            start=0,
            end=1,
            size=0.001
        ))
        fig.write_image("hist1D.svg")
        fig.show()
    elif histogram=='2D':
        import plotly.express as px
        # fig = px.density_heatmap(x=smooth_mat_1.flatten(),
        #                           y=smooth_mat_2.flatten(),
        #                           marginal_x='histogram',marginal_y='histogram')
        
        fig = px.scatter(x=smooth_mat_1.flatten(), 
                         y=smooth_mat_2.flatten(),
                         color_discrete_sequence=['black'],
                         marginal_x="histogram",
                         marginal_y="histogram")
        fig.update_traces(marker=dict(size=1,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
        fig.write_image("hist2D.svg")
        fig.show()
    return phase_mat

def measure_TPB(domain, dx):
    """
    this function is used to measure the triple phase boundary (TPB). it is written in 
    vectorized form. so the runtime is extremely faster than non-vectorized version.
    inputs:
    phase_mat: int
        phase matrix of the porous media. a 3D matrix filled with only three values:
        1: ion conductor
        2: pores
        3: electron conductor
        (it's just a convention! could've been anything else)
    outputs:
    TPB_mask: bool
        a boolean mask of the TPB locations.
    TPB_density: float
        the density of the TPB (length per volume).
    vertices:
        vertices in the entire 3D domain. used for visualization in pyvista.
    lines:
        TPB lines in the 3D domain. used for visualization in pyvista.
    """
    # from tqdm import tqdm
    from scipy import ndimage as ndi

    # extract the percolating phase matrix    
    # phase_mat, _, _= percolation_analysis(phase_mat)

    # print("Extracting TPB...", end='')
    N=domain.shape
    TPBs = np.empty((0,6), int)
    TPB_mask = np.zeros_like(domain, dtype=bool)

    # loop through the x axis     
    for i in np.arange(N[0]):
        # create four sliced 2D matrices in each i index
        mat_1 = domain[i , 0:-1 , 0:-1 ].flatten()
        mat_2 = domain[i , 0:-1 , 1:   ].flatten()
        mat_3 = domain[i , 1:   , 0:-1 ].flatten()
        mat_4 = domain[i , 1:   , 1:   ].flatten()

        # concatanate the sliced matrices
        mat_cumulative = np.stack((mat_1,mat_2,mat_3,mat_4), axis=-1)

        # find the number of unique phases in each neighborhood of size (2,2)
        sorted = np.sort(mat_cumulative,axis=1)
        unique_phases = (sorted[:,1:] != sorted[:,:-1]).sum(axis=1)+1
        unique_phases -= np.sum(np.isnan(sorted), axis=1)

        # find the locaiton of TPBs, where all three phases are present in a 
        # (2,2) neighborhood (unique_phases==3).
        TPBs_flatten = np.zeros_like(mat_1).astype(bool)
        TPBs_flatten[unique_phases==3] = True

        # resphase the flattened mask matrix (denoting the location of TPBs) to 2D
        TPBs_plane = TPBs_flatten.reshape((N[1]-1,N[2]-1))

        # find the indices of the TPBs in the 2D plane
        indices = np.argwhere(TPBs_plane==True)+np.array([1,1])

        # construct the coordinates of lines connecting TPB points.
        TPB_location = np.zeros(shape=(np.size(indices,0),6),dtype=int)
        TPB_location[:,0] = i
        TPB_location[:,[1,2]] = indices
        TPB_location[:,3:] = TPB_location[:,0:3] + np.array([1,0,0])
        
        # append the TPB lines in this plane to the TPB matrix of the entire structure
        TPBs = np.append(TPBs, TPB_location, axis=0)
        
        # detect the (2,2) neighborhood of each TPB point in the plane and mask the 
        # corresponding elements to be used as the location of the source matrix 
        # later in the model.
        TPBs_expanded = np.append(TPBs_plane, np.zeros((1,N[2]-1),dtype=bool),axis=0)
        TPBs_expanded = np.append(TPBs_expanded, np.zeros((N[1],1),dtype=bool),axis=1)
        TPBs_expanded[TPBs_expanded != np.roll(TPBs_expanded, True, axis=0)] = True
        TPBs_expanded[TPBs_expanded != np.roll(TPBs_expanded, True, axis=1)] = True   
        TPB_mask[i,:,:] = np.logical_or(TPBs_expanded, TPB_mask[i,:,:])
        # TPB_mask[i,:,:] = TPBs_expanded

    for j in np.arange(N[1]):
        # create four sliced 2D matrices in each j index
        mat_1 = domain[0:-1 , j , 0:-1 ].flatten()
        mat_2 = domain[0:-1 , j , 1:   ].flatten()
        mat_3 = domain[1:   , j , 0:-1 ].flatten()
        mat_4 = domain[1:   , j , 1:   ].flatten()

        # concatanate the sliced matrices
        mat_cumulative = np.stack((mat_1,mat_2,mat_3,mat_4), axis=-1)

        # find the number of unique phases in each neighborhood of size (2,2)
        sorted = np.sort(mat_cumulative,axis=1)
        unique_phases = (sorted[:,1:] != sorted[:,:-1]).sum(axis=1)+1
        unique_phases -= np.sum(np.isnan(sorted), axis=1)

        # find the locaiton of TPBs, where all three phases are present in a 
        # (2,2) neighborhood (unique_phases==3).
        TPBs_flatten = np.zeros_like(mat_1).astype(bool)
        TPBs_flatten[unique_phases==3] = True

        # resphase the flattened mask matrix (denoting the location of TPBs) to 2D
        TPBs_plane = TPBs_flatten.reshape((N[0]-1,N[2]-1))
        
        # find the indices of the TPBs in the 2D plane
        indices = np.argwhere(TPBs_plane==True)+np.array([1,1])

        # construct the coordinates of lines connecting TPB points.
        TPB_location = np.zeros(shape=(np.size(indices,0),6),dtype=int)
        TPB_location[:,1] = j
        TPB_location[:,[0,2]] = indices
        TPB_location[:,3:] = TPB_location[:,0:3] + np.array([0,1,0])
        
        # append the TPB lines in this plane to the TPB matrix of the entire structure
        TPBs = np.append(TPBs, TPB_location, axis=0)

        # detect the (2,2) neighborhood of each TPB point in the plane and mask the 
        # corresponding elements to be used as the location of the source matrix 
        # later in the model.
        TPBs_expanded = np.append(TPBs_plane, np.zeros((1,N[2]-1),dtype=bool),axis=0)
        TPBs_expanded = np.append(TPBs_expanded, np.zeros((N[0],1),dtype=bool),axis=1)
        TPBs_expanded[TPBs_expanded != np.roll(TPBs_expanded, True, axis=0)] = True
        TPBs_expanded[TPBs_expanded != np.roll(TPBs_expanded, True, axis=1)] = True   
        TPB_mask[:,j,:] = np.logical_or(TPBs_expanded, TPB_mask[:,j,:])
        # TPB_mask[:,j,:] = TPBs_expanded
                    
    for k in np.arange(N[2]):
        # create four sliced 2D matrices in each k index
        mat_1 = domain[0:-1 , 0:-1 , k].flatten()
        mat_2 = domain[0:-1 , 1:   , k].flatten()
        mat_3 = domain[1:   , 0:-1 , k].flatten()
        mat_4 = domain[1:   , 1:   , k].flatten()

        # concatanate the sliced matrices
        mat_cumulative = np.stack((mat_1,mat_2,mat_3,mat_4), axis=-1)

        # find the number of unique phases in each neighborhood of size (2,2)
        sorted = np.sort(mat_cumulative,axis=1)
        unique_phases = (sorted[:,1:] != sorted[:,:-1]).sum(axis=1)+1
        unique_phases -= np.sum(np.isnan(sorted), axis=1)
        
        # find the locaiton of TPBs, where all three phases are present in a 
        # (2,2) neighborhood (unique_phases==3).
        TPBs_flatten = np.zeros_like(mat_1).astype(bool)
        TPBs_flatten[unique_phases==3] = True

        # resphase the flattened mask matrix (denoting the location of TPBs) to 2D
        TPBs_plane = TPBs_flatten.reshape((N[0]-1,N[1]-1))

        # find the indices of the TPBs in the 2D plane
        indices = np.argwhere(TPBs_plane==True)+np.array([1,1])

        # construct the coordinates of lines connecting TPB points.
        TPB_location = np.zeros(shape=(np.size(indices,0),6),dtype=int)
        TPB_location[:,2] = k
        TPB_location[:,[0,1]] = indices
        TPB_location[:,3:] = TPB_location[:,0:3] + np.array([0,0,1])
        
        # append the TPB lines in this plane to the TPB matrix of the entire structure
        TPBs = np.append(TPBs, TPB_location, axis=0)

        # detect the (2,2) neighborhood of each TPB point in the plane and mask the 
        # corresponding elements to be used as the location of the source matrix 
        # later in the model.
        TPBs_expanded = np.append(TPBs_plane, np.zeros((1,N[1]-1),dtype=bool),axis=0)
        TPBs_expanded = np.append(TPBs_expanded, np.zeros((N[0],1),dtype=bool),axis=1)
        TPBs_expanded[TPBs_expanded != np.roll(TPBs_expanded, True, axis=0)] = True
        TPBs_expanded[TPBs_expanded != np.roll(TPBs_expanded, True, axis=1)] = True   
        TPB_mask[:,:,k] = np.logical_or(TPBs_expanded, TPB_mask[:,:,k])
        # TPB_mask[:,:,k] = TPBs_expanded

    
    # create the lines connecting TPBs. Since all of the lines has length 2, the first element of 
    # the matrix "lines" must be 2 for all rows.
    lines = np.zeros(shape=(len(TPBs),3))+[2,0,0]
    i = TPBs[:,[0,3]]
    j = TPBs[:,[1,4]]
    k = TPBs[:,[2,5]]
    
    TPB_line = i*(N[1]+1)*(N[2]+1)+j*(N[2]+1)+k

    # reshape the "lines" matrix to be of the size required by pyvista
    lines[:,1:] = TPB_line
    lines = lines.reshape((1,3*len(TPBs)))
    lines = lines.astype(int)
    
    # measure the density of TPBs in the structure. [m/m^3]
    TPB_density = len(TPBs)*dx / (N[0]*N[1]*N[2]*dx**3)
    # print('Done!')

    # measure the length distrubition of TPBs in the X direction (i)
    TPB_mean_location_i = np.floor(np.mean(i,axis=1))
    TPB_dist_x = np.zeros(shape=domain.shape[0])
    for i in range(domain.shape[0]):
        TPB_dist_x[i] = np.sum(TPB_mean_location_i==i) * dx / (N[1]*N[2]*dx**2)     # [m/m^2]

    return TPB_mask, TPB_density, lines, TPB_dist_x

def measure_TPB_notvec(phase_mat):
    """
    obsolete and non-vectorized version of measure_TPB_vec. the runtime of 
    this function is much higher than the vectorized version.
    """
    from tqdm import tqdm
    
    phase_mat, _, _= percolation_analysis(phase_mat)

    N=phase_mat.shape[0]
    TPBs = []

    print("\nMeasuring TPB along X axis")        
    for i in tqdm(np.arange(N)):
        for j in np.arange(N-1):
            for k in np.arange(N-1):
                num = np.unique(phase_mat[i,j:j+2,k:k+2])
                num = num[~np.isnan(num)]
                if len(num)==3:
                    TPB = [[i,j+1,k+1],[i+1,j+1,k+1]]
                    TPBs.append(TPB)
                    
    print("\nMeasuring TPB along Y axis")
    for j in tqdm(np.arange(N)):
        for k in np.arange(N-1):
            for i in np.arange(N-1):
                num = np.unique(phase_mat[i:i+2,j,k:k+2])
                num = num[~np.isnan(num)]
                if len(num)==3:
                    TPB = [[i+1,j,k+1],[i+1,j+1,k+1]]
                    TPBs.append(TPB)
                    
    print("\nMeasuring TPB along Z axis:")
    for k in tqdm(np.arange(N)):
        for j in np.arange(N-1):
            for i in np.arange(N-1):
                num = np.unique(phase_mat[i:i+2,j:j+2,k])
                num = num[~np.isnan(num)]
                if len(num)==3:
                    TPB = [[i+1,j+1,k],[i+1,j+1,k+1]]
                    TPBs.append(TPB)


    # This section needs revision, the final matrix is not aligned with the 
    # phase_mat. In addition, there must be some ways to get rid of the nested
    # loops and decrease the run time.
    count = 0
    vertices = np.ones(shape=(np.power(N+1,3),3))
    for k in np.arange(N+1):
        for i in np.arange(N+1):
            for j in np.arange(N+1):
                vertices[count,:] = [i,j,k]
                count += 1
                
    lines = np.zeros(shape=(len(TPBs),3))+[2,0,0]
    for index in np.arange(len(TPBs)):
        TPBarray = np.asarray(TPBs[index])
        i = TPBarray[:,0]
        j = TPBarray[:,1]
        k = TPBarray[:,2]
        TPB_line = k*(np.power(N+1,2))+i*(N+1)+j
        lines[index,:]=[2,TPB_line[0],TPB_line[1]]

    lines = lines.reshape((1,3*len(TPBs)))
    lines = lines.astype(int)
    
    TPB_density = len(TPBs) / np.power(N,3)
    
    return TPBs, TPB_density, vertices, lines

def percolation_analysis(phase_mat):
    from scipy import ndimage as ndi
    # only the pores (phase_mat==1) must be true for percolation analysis
    phase_1 = np.zeros(shape=phase_mat.shape, dtype=bool)
    phase_1[phase_mat==1] = True
    lw1, num1 = ndi.label(phase_1)
    
    phase_2 = np.zeros(shape=phase_mat.shape, dtype=bool)
    phase_2[phase_mat==2] = True
    lw2, num2 = ndi.label(phase_2)
    
    phase_3 = np.zeros(shape=phase_mat.shape, dtype=bool)
    phase_3[phase_mat==3] = True
    lw3, num3 = ndi.label(phase_3)

    lw1_shuffled = shuffle_labels(lw1)
    lw2_shuffled = shuffle_labels(lw2)
    lw3_shuffled = shuffle_labels(lw3)
    
    # check intersection between left/right, up/down, and top/bottom boundaries 
    # to see if the porous media is percolating. Only x (i) direction is investigated
    # for percolation.
    intersection_x_1 = np.intersect1d(lw1_shuffled[0 , : , :].flatten(), 
                                      lw1_shuffled[-1, : , :].flatten())
    intersection_x_2 = np.intersect1d(lw2_shuffled[0 , : , :].flatten(), 
                                      lw2_shuffled[-1, : , :].flatten())
    intersection_x_3 = np.intersect1d(lw3_shuffled[0 , : , :].flatten(), 
                                      lw3_shuffled[-1, : , :].flatten())

    is_per_x_1 = False
    is_per_x_2 = False
    is_per_x_3 = False
    if intersection_x_1.size>0: is_per_x_1=True 
    if intersection_x_2.size>0: is_per_x_2=True
    if intersection_x_3.size>0: is_per_x_3=True
    
    if (is_per_x_1 and is_per_x_2 and is_per_x_3)==False:
        raise Exception("Error! One of the phases is not percolating!")

    # delete the isolated (non-percolating) pores. format: [1-2-3]
    percolating_label_1 = np.ones(shape = phase_mat.shape)
    percolating_label_2 = np.ones(shape = phase_mat.shape)
    percolating_label_3 = np.ones(shape = phase_mat.shape)
    percolating_label_1[:] = np.nan
    percolating_label_2[:] = np.nan
    percolating_label_3[:] = np.nan
    
    for i in range(len(intersection_x_1)):
        percolating_label_1[lw1_shuffled==intersection_x_1[i]] = 1
    for i in range(len(intersection_x_2)):
        percolating_label_2[lw2_shuffled==intersection_x_2[i]] = 2
    for i in range(len(intersection_x_3)):
        percolating_label_3[lw3_shuffled==intersection_x_3[i]] = 3
    
    phase_mat_nans = np.zeros(shape = phase_mat.shape)
    # phase_mat_nans = np.nan
    phase_mat_nans[percolating_label_1==1] = 1
    phase_mat_nans[percolating_label_2==2] = 2
    phase_mat_nans[percolating_label_3==3] = 3
    phase_mat_nans[phase_mat_nans==0] = np.nan
    
    percolating_labels = np.zeros(shape = np.append(phase_mat.shape,3))
    percolating_labels[:,:,:,0] = percolating_label_1
    percolating_labels[:,:,:,1] = percolating_label_2
    percolating_labels[:,:,:,2] = percolating_label_3
    
    lw_shuffled = np.zeros(shape = np.append(phase_mat.shape,3))
    lw_shuffled[:,:,:,0] = lw1_shuffled
    lw_shuffled[:,:,:,1] = lw2_shuffled
    lw_shuffled[:,:,:,2] = lw3_shuffled
    
    return phase_mat_nans, lw_shuffled, percolating_labels
        
def shuffle_labels(labeled_mat):
    # shuffling the labels without changing the location of "Falses"
    # It's because the label "False" denotes the background phase 
    # we want to delete the background phase from the matrix (replace them with nan)
    num = np.max(labeled_mat)
    b = np.arange(start=1, stop=num+1)
    np.random.shuffle(b)
    b = np.append(0,b)
    shuffled_mat = b[labeled_mat.astype(int)]
    shuffled_mat = shuffled_mat.astype(float)
    shuffled_mat[shuffled_mat==0] = np.nan
    return shuffled_mat

def image_segmentation(phase_mat,sigma=5,display=False):
    """ 
    Segmentation of the phase matrix.
    Inputs:
    phase_mat : float
        Three dimensional matrix describing the phase data.
    
    fp : float
        footprint
        
    Outputs:
    labels:"""
    import concurrent.futures
    
    # import plotly.io as po
    # from plotly.subplots import make_subplots
    # import plotly.graph_objects as go
    
    # po.renderers.default = "browser"
        
    labels = np.zeros(shape = np.append(phase_mat.shape,3))
    dist_mat = np.zeros(shape = np.append(phase_mat.shape,3))
    
    # volumes = np.empty([0,3])
    volumes = [[],[],[]]
    centroids = [[],[],[]]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # results = [executor.submit(segment, phase_mat==i, sigma) for i in [1,2,3]]
        results_1 = executor.submit(segment, phase_mat==1, sigma)
        results_2 = executor.submit(segment, phase_mat==2, sigma)
        results_3 = executor.submit(segment, phase_mat==3, sigma)

        # for i in range(3):
        labels[:,:,:,0], volumes[0], centroids[0], dist_mat[:,:,:,0] = results_1.result()
        labels[:,:,:,1], volumes[1], centroids[1], dist_mat[:,:,:,1] = results_2.result()
        labels[:,:,:,2], volumes[2], centroids[2], dist_mat[:,:,:,2] = results_3.result()
    
    if display:
        from modules.postprocess import visualize_mesh as vm
        vm([labels[:,:,:,0], labels[:,:,:,1], labels[:,:,:,2]], [(), (), ()])

    return labels, dist_mat, volumes, centroids

def remove_thin_boundaries(phase_mat):
    """
    Removes thin boundaries from the phase matrix. It is a good practice 
    to remove thin boundaries before running a simulation with 
    Neumann boundary conditions.

    inputs:
    phase_mat : float
        Three dimensional matrix describing the phase data.
    """
    phase_mat[ : , : , 0][phase_mat[:,:,0]  != phase_mat[:,:,1] ] = np.nan
    phase_mat[ : , : ,-1][phase_mat[:,:,-1] != phase_mat[:,:,-2]] = np.nan
    phase_mat[ : , 0 , :][phase_mat[:,0,:]  != phase_mat[:,1,:] ] = np.nan
    phase_mat[ : ,-1 , :][phase_mat[:,-1,:] != phase_mat[:,-2,:]] = np.nan
    phase_mat[ 0 , : , :][phase_mat[0,:,:]  != phase_mat[1,:,:] ] = np.nan
    phase_mat[-1 , : , :][phase_mat[-1,:,:] != phase_mat[-2,:,:]] = np.nan
    return phase_mat

def remove_edges(phi):
    """
    Remove the edges of the domain
    """
    phi[0,0,:]   = np.nan
    phi[-1,0,:]  = np.nan
    phi[0,-1,:]  = np.nan
    phi[-1,-1,:] = np.nan
    phi[0,:,0]   = np.nan
    phi[0,:,-1]  = np.nan
    phi[-1,:,0]  = np.nan
    phi[-1,:,-1] = np.nan
    phi[:,0,0]   = np.nan
    phi[:,-1,0]  = np.nan
    phi[:,-1,-1] = np.nan
    phi[:,0,-1]  = np.nan

    return phi
    
def create_microstructure(inputs, display=False):
    print("Generating microstructure...", end='')
    flag_lattice = inputs['microstructure']['lattice_geometry']['flag']
    flag_plurigaussian = inputs['microstructure']['plurigaussian']['flag']
    flag_reduced = inputs['microstructure']['plurigaussian']['reduced_geometry']['flag']
    flag_fibrous = inputs['microstructure']['fibrous_bed']['flag']

    dx = inputs['microstructure']['dx']
    scale_factor = inputs['microstructure']['scale_factor']
    
    d_ave = inputs['microstructure']['average_diameter']
    
    voxels = [
        int(inputs['microstructure']['length']['X']/dx), 
        int(inputs['microstructure']['length']['Y']/dx),
        int(inputs['microstructure']['length']['Z']/dx)]
    
    vol_frac = [
        inputs['microstructure']['volume_fractions']['pores'],
        inputs['microstructure']['volume_fractions']['Ni'],
        inputs['microstructure']['volume_fractions']['YSZ']]
    
    infiltration_loading = inputs['microstructure']['infiltration_loading']

    # create the entire domain
    if flag_lattice:
        flag_smallest_lattice = inputs['microstructure']['lattice_geometry']['smallest_lattice']
        domain = create_microstructure_lattice(
            vol_frac, 
            dx, 
            voxels, 
            d_ave,
            offset=True,
            smallest_lattice=flag_smallest_lattice)
    
    elif flag_plurigaussian:
        seed = inputs['microstructure']['plurigaussian']['seed']
        gradient_factor = inputs['microstructure']['plurigaussian']['gradient_factor']
        
        if flag_reduced:
            Lx_extended = inputs['microstructure']['plurigaussian']['reduced_geometry']['Lx_extended']
            if Lx_extended < inputs['microstructure']['length']['X']:
                raise ValueError('Nx_extended must be larger than Nx')
            voxels[0] = int(Lx_extended/dx)

        domain = create_microstructure_plurigaussian(
            voxels = voxels,
            vol_frac = vol_frac,
            d_ave = d_ave,
            dx = dx,
            seed = seed,
            gradient_factor = gradient_factor,
            periodic = False)
    
        if flag_reduced:
            domain = domain[:voxels[0],:,:]
    
    elif flag_fibrous:
        fibre_length = int(inputs['microstructure']['fibrous_bed']['fibre_length']/dx)
        radius = int(d_ave/2/dx)
        freq = inputs['microstructure']['fibrous_bed']['frequency']
        overlap = inputs['microstructure']['fibrous_bed']['overlap']
        rot_max = inputs['microstructure']['fibrous_bed']['rotation_max']
        bend_max = inputs['microstructure']['fibrous_bed']['bend_max']
        target_porosity = vol_frac[0]
        domain = create_fibrous_bed(
            voxels = voxels,
            radius = radius, 
            fibre_length = fibre_length, 
            target_porosity = target_porosity,
            freq = freq, 
            overlap = overlap, 
            rotation_max = rot_max,
            bend_max = bend_max)

    domain = infiltration(domain, infiltration_loading)

    domain = downscale_domain(domain,scale_factor)
    inputs['microstructure']['dx'] = dx*scale_factor

    if display:
        from modules.postprocess import visualize_mesh as vm
        vm([domain], [(2,3)])

    print('Done!')
    return domain

def topological_operations(inputs, domain_old, show_TPB=False, show_TPB_variations=False):
    """
    domain topological operations
    """
    print("Domain topological operations...", end='')
    # removing thin boundaries
    # remove thin boundaries to avoid numerical error for the case of Neumann BCs.
    # don't remove thin boundaries for periodic boundary conditions. it will cause problems.
    # from scipy.io import savemat
    # savemat('domain.mat', {'domain': domain})
    
    domain_old = remove_thin_boundaries(domain_old.astype(float))
    # extract the domain that should be solved. ds is short for Domain for Solver.
    # when periodic boundary condition is used, percolation analysis should not be done.
    domain, _, _ = percolation_analysis(domain_old)

    percolation_percentage =[None] * 3
    for i in range(3):
        percolation_percentage[i] = np.sum(domain==i+1) / np.sum(domain_old==i+1)
    
    # measure covariance
    # cov = measure_covariance(domain, phase=1)
    
    # measure the triple phase boundary and create a mask for source term
    dx = inputs['microstructure']['dx']
    TPB_mask, TPB_density, lines, TPB_dist_x = measure_TPB(domain, dx)

    # convert float to integer
    domain = domain.astype(int)
    lines = lines.astype(int)
    
    print("Done!")

    TPB_dict = {
        'TPB_mask': TPB_mask,
        'TPB_density': TPB_density,
        'lines': lines,
        'TPB_dist_x': TPB_dist_x
    }

    if show_TPB:
        import pyvista as pv
        TPB_mesh = pv.PolyData(create_vertices_in_uniform_grid(domain.shape), lines=lines)
        from modules.postprocess import visualize_mesh as vm
        vm([domain], [(2,3)], TPB_mesh=TPB_mesh, animation='none')

    # plot the variation of TPB density along the X direction
    if show_TPB_variations:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(domain.shape[0])*dx, y=TPB_dist_x, mode='lines+markers'))
        fig.update_layout(
            title="Variation of TPB density along X direction",
            xaxis_title="X [m]",
            yaxis_title="number of TPB elements",
            )
        fig.show()

    if inputs['solver_options']['image_analysis_only']:
        np.savez(
            f'Binary files/image analysis/domain_{inputs["file_options"]["id"]}.npz',
            domain=domain,
            TPB_mask=TPB_mask,
            lines=lines,
            )
        np.savez(
            f'Binary files/image analysis/scalars_{inputs["file_options"]["id"]}.npz',
            TPB_density=TPB_density,
            percolation_percentage=percolation_percentage,
            TPB_dist_x=TPB_dist_x,
            )

    # print percolation percentage and TPB density with three significant digits
    print(f'TPB density: \n{TPB_density/1e12:.3f}')
    print(f'Percolation percentage:\n{percolation_percentage[0]:.3f}, {percolation_percentage[1]:.3f}, {percolation_percentage[2]:.3f}')
    
    # tortuosity_calculator(domain)
    return domain, TPB_dict

# specific functions for entire cell microstructure
def create_microstructure_entire_cell(inputs):
    
    domain_a = create_microstructure_plurigaussian(
        voxels = [inputs['Nx_a'],inputs['Ny'],inputs['Nz']],
        vol_frac = [inputs['vf_pores_a'],inputs['vf_Ni_a'],inputs['vf_YSZ_a']],
        sigma = inputs['sig_gen_a'],
        seed = inputs['seed'],
        display = False,
        )
    
    domain_c = create_microstructure_plurigaussian(
        voxels = [inputs['Nx_c'],inputs['Ny'],inputs['Nz']],
        vol_frac = [inputs['vf_pores_c'],inputs['vf_LSM_c'],inputs['vf_YSZ_c']],
        sigma = inputs['sig_gen_c'],
        seed = inputs['seed'],
        display = False,
        )
    domain_c[domain_c==1] = 4   # pores = 4
    domain_c[domain_c==2] = 5   # LSM = 5
    
    domain_e = np.zeros(shape = (inputs['Nx_e'], inputs['Ny'], inputs['Nz']), dtype = int)+3
    
    domain = np.concatenate((domain_a, domain_e, domain_c), axis=0)

    return domain

def topological_operations_entire_cell(inputs, domain):
    import pyvista as pv

    domain_a = domain[:inputs['Nx_a'],:,:]
    domain_c = domain[-inputs['Nx_c']:,:,:]
    domain_e = domain[inputs['Nx_a']:-inputs['Nx_c'],:,:]

    domain_a, TPB_dict_a = topological_operations(inputs, domain_a)
    domain_c, TPB_dict_c = topological_operations(inputs, domain_c-2)   # normalize between 1 and 3
    domain_c += 2   # re-normalize between 3 and 5
    TPB_dict_c['vertices'][:,0] += inputs['Nx_a'] + inputs['Nx_e']
    domain = np.concatenate((domain_a, domain_e, domain_c), axis=0)
    TPB_dict = {"anode": TPB_dict_a, "cathode": TPB_dict_c}

    if inputs['display_mesh']:
        import modules.postprocess as pp
        pp.visualize_mesh([domain],[()])
        
    return domain, TPB_dict

def create_ideal_microstructure_spheres(voxels, TPB_radius):
    # This function creates an idealized microstructure with two spheres in one
    # third and two thirds of the domain. The TPB length can then be calculated 
    # analytically. This is useful for testing the TPB length calculation.
 
    # voxels = [TPB_radius*4] * 3
    domain = np.zeros(shape = voxels, dtype = int)

    centers = np.array([[voxels[0]//3, voxels[1]//2, voxels[2]//2], 
                        [2*voxels[0]//3, voxels[1]//2, voxels[2]//2]])
    
    # radius of two spheres in the domain
    phase_radius = ((voxels[0]//6)**2 + TPB_radius**2)**0.5
    
    # create the two spheres
    for i in range(voxels[0]):
        for j in range(voxels[1]):
            for k in range(voxels[2]):
                if (i-centers[0,0])**2 + (j-centers[0,1])**2 + (k-centers[0,2])**2 < phase_radius**2 and i<=voxels[0]//2:
                    domain[i,j,k] = 1
                if (i-centers[1,0])**2 + (j-centers[1,1])**2 + (k-centers[1,2])**2 < phase_radius**2 and i>voxels[0]//2:
                    domain[i,j,k] = 2

    TPB_analytical = 2*TPB_radius * np.pi
    _, density, vertices, lines = measure_TPB(domain, 1)
    TPB_measured = density * voxels[0] * voxels[1] * voxels[2]

    from modules.postprocess import visualize_mesh as vm
    import pyvista as pv
    TPB_mesh = pv.PolyData(vertices, lines=lines)
    vm([domain], [(1,1)], TPB_mesh=TPB_mesh)

    return TPB_analytical, TPB_measured

def create_ideal_microstructre_straight_bars(inputs, display=False):
    Nx = inputs['microstructure']['Nx']
    Ny = inputs['microstructure']['Ny']
    Nz = inputs['microstructure']['Nz']
    N = [Nx, Ny, Nz]

    domain = np.ones(shape = N, dtype=int)
    domain[: , :int(Ny/2) , :int(Nz/2)] = 2
    domain[: , int(Ny/2): , :int(Nz/2)] = 3

    if display:
        from modules.postprocess import visualize_mesh as vm 
        vm([domain],[()])
    
    return domain

def tortuosity_calculator(domain):
    # This function uses random walk calculation to measure tortuosity of each phase
    # in the domain. The tortuosity is calculated as the ratio of the average distance
    # traveled by the random walker to the distance between the starting and ending
    # points. The random walker is allowed to move in all 26 directions.
    # The function returns the tortuosity of each phase in the domain.
    
    import random
    import concurrent.futures

    # numbers in the domain should be 0-1-2
    domain -= 1

    # define the directions in which the random walker can move
    directions = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1],
                            [0,0,-1], [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0],
                            [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1], [0,1,1],
                            [0,1,-1], [0,-1,1], [0,-1,-1], [1,1,1], [1,1,-1],
                            [1,-1,1], [1,-1,-1], [-1,1,1], [-1,1,-1], [-1,-1,1],
                            [-1,-1,-1]])

    N = domain.shape
    # define the number of random walks to be performed
    N_walks = 1000

    # define the number of steps in each random walk
    steps = 100

    # initialize the distance array
    distance = np.zeros(shape = (3, N_walks, steps))
    distance[:] = np.nan
    distance_avg = np.zeros(shape = (3, steps))

    # perform the random walk for each phase
    for phase in range(3):        
        # select a random starting point within the phase
        start = np.array([N[0]//2, N[1]//2, N[2]//2])
        f = domain[start[0], start[1], start[2]]
        while f != phase:
            N1 = random.randint(2*N[0]//5, 3*N[0]//5)
            N2 = random.randint(2*N[1]//5, 3*N[1]//5)
            N3 = random.randint(2*N[2]//5, 3*N[2]//5)
            start = np.array([N1, N2, N3])
            f = domain[N1, N2, N3]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res_parallel = [executor.submit(single_random_walk, domain, start, directions, steps) for _ in range(N_walks)]
            
            for Nw, f in enumerate(concurrent.futures.as_completed(res_parallel)):
                distance[phase, Nw, :] = f.result()

        distance_avg[phase,:] = np.nanmean(distance[phase,:,:], axis=0)


    import plotly.graph_objects as go

    for p in range(3):
        # initialize the figure
        fig = go.Figure()
        for Nw in range(N_walks):
            fig.add_scatter(y=distance[p,Nw,:], line=dict(color='grey', width=0.5), showlegend=False)
        fig.add_scatter(y=distance_avg[p,:], line=dict(color='black', width=2), name='Phase '+str(p))
        fig.show()

    return distance

def single_random_walk(domain, start, directions, steps):
    import random

    N = domain.shape

    # initialize the random walker
    walker = [None] * steps
    walker[0] = start

    # the first distance is zero
    distance = np.zeros(steps)
    distance[:] = np.nan
    
    # perform the random walk
    for t in range(1,steps):
        # choose a random direction
        direction = random.choice(directions)
        
        # move the walker in that direction
        walker[t] = walker[t-1] + direction

        # check if the walker has left the domain or has crossed a phase boundary
        if (walker[t][0]<0 or walker[t][0]>=N[0] or 
            walker[t][1]<0 or walker[t][1]>=N[1] or 
            walker[t][2]<0 or walker[t][2]>=N[2]):
            break

        elif (domain[walker[t][0],   walker[t][1],   walker[t][2]] !=
            domain[walker[t-1][0], walker[t-1][1], walker[t-1][2]]):
            walker[t] = walker[t-1]

        # calculate the distance traveled by the walker
        distance[t] = np.linalg.norm(walker[t]-start)
    
    return distance

def segment(phase_mat, sigma):
    from scipy import ndimage as ndi
    from skimage.filters import gaussian
    from skimage.segmentation import watershed

    dist_mat = ndi.distance_transform_edt(phase_mat.astype(int))
    dist_mat = gaussian(dist_mat, sigma=sigma, mode='nearest')
    dist_mat[~phase_mat] = 0
    
    labels = watershed(-dist_mat, mask=phase_mat)
    labels = shuffle_labels(labels)
    
    volumes = np.zeros((np.nanmax(labels).astype(int),1))
    centroids = np.zeros((np.nanmax(labels).astype(int),3))
    
    for particle in np.arange(np.nanmax(labels), dtype=int):
        x = np.where(labels == particle+1)
        centroids[particle,:] = np.average(x,axis=1)
        volumes[particle] = len(x[0])
    
    return labels, volumes, centroids, dist_mat

def infiltration(domain, loading):
    if loading == 0: return domain

    import random
    import scipy.ndimage as ndi

    # find the interfacial surface area of the phases
    isa_12_mat = measure_interface(domain, [1,2])
    isa_31_mat = measure_interface(domain, [1,3])

    # interfacial surface area of pores
    isa_pore = np.logical_or(isa_12_mat, isa_31_mat)

    # find all the indices of the pore interface
    indices_isa_pore = np.where(isa_pore == True)
    length_isa = len(indices_isa_pore[0])

    # randomly select points from the indices
    num_points = int(loading * length_isa)
    indices_infltr = random.sample(range(length_isa), num_points)

    # add the infiltration points to the phase matrix
    phase_inflr = np.zeros(domain.shape, dtype=bool)
    for i in range(num_points):
        phase_inflr[
            indices_isa_pore[0][indices_infltr[i]], 
            indices_isa_pore[1][indices_infltr[i]], 
            indices_isa_pore[2][indices_infltr[i]]] = True
        
    # dilation process to increase the size of the infiltration region
    # iteration parameter should be selected with care
    # it affects the size of infiltrated particles
    dil_inflr = ndi.binary_dilation(phase_inflr, iterations=2).astype(phase_inflr.dtype)

    # assign the second phase (pore=1, Ni=2, YSZ=3) to the dilated region
    domain[np.logical_and(dil_inflr,domain==1)] = 2

    return domain

def create_microstructure_lattice(vol_frac, dx, voxels, d_particle, offset=True, smallest_lattice=True):
    from scipy.optimize import fsolve

    N = voxels

    if sum(vol_frac) - 1 > 1e-2:
        raise ValueError('Sum of volume fractions must be equal to 1!')

    if (vol_frac[1] != vol_frac[2]):
        raise ValueError('For lattice microstructure, volume fractions of Ni and YSZ must be equal!')
    
    vf_particle = vol_frac[1] + vol_frac[2]

    if vf_particle < np.pi/6 or vf_particle > 0.965:
        raise ValueError('For ordered lattice microstructures, particle volume fraction (Ni+YSZ) should be between 0.5236 and 0.965!')
    
    def find_vf(f0):
        f = 4/3*np.pi*(1 - 9/2*f0**2 + 3/2*f0**3)/8/(1-f0)**3 - vf_particle
        return f

    f0 = fsolve(find_vf, 0.1)

    r = d_particle / 2
    L_lattice = 2*r*(1-f0)
    N_lat = int(L_lattice // dx)

    phase_mat_lat = np.zeros((N_lat,2*N_lat,2*N_lat), dtype=int)
    dist_center_mat = dx * np.linalg.norm(
        np.indices((N_lat,N_lat,N_lat), dtype=int) -
        np.array([N_lat//2,N_lat//2,N_lat//2]).reshape((3,1,1,1)), axis=0)
    
    phase_mat_lat[:,:N_lat,:N_lat][dist_center_mat < r] = 1
    phase_mat_lat[:,N_lat:,N_lat:][dist_center_mat < r] = 1
    phase_mat_lat[:,N_lat:,:N_lat][dist_center_mat < r] = 2
    phase_mat_lat[:,:N_lat,N_lat:][dist_center_mat < r] = 2

    if smallest_lattice:
        N[1] = phase_mat_lat.shape[1]
        N[2] = phase_mat_lat.shape[2]

    X_tiles = int(N[0] // phase_mat_lat.shape[0])
    Y_tiles = int(N[1] // phase_mat_lat.shape[1])
    Z_tiles = int(N[2] // phase_mat_lat.shape[2])

    # create the entire microstructure by tiling a single lattice
    phase_mat = np.tile(phase_mat_lat, (X_tiles,Y_tiles,Z_tiles)).astype(int)

    # roll the entire lattice by half the size of a single lattice in all directions
    # This is done to make sure that all TPB are captured in the simulation domain
    if offset:
        phase_mat = np.roll(phase_mat, N_lat//2, axis=0)
        phase_mat = np.roll(phase_mat, N_lat//2, axis=1)
        phase_mat = np.roll(phase_mat, N_lat//2, axis=2)

    # phase numbers are assigned as follows:
    phase_mat += 1      # pore=1, Ni=2, YSZ=3

    # analytical TPB density
    TPB_density = 1/2 * np.pi * np.sqrt(2*f0 - f0**2) / (1-f0)**3 * r / r**3 # [m/m3]
    TPB_density /= 1e12 # [µm/µm3]
    return phase_mat

def compare_circle_circumference(lower_bound, upper_bound):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import pandas as pd

    DIAMETER = np.arange(lower_bound,upper_bound)
    length_voxel = np.zeros(shape=DIAMETER.shape)
    length_real = np.zeros(shape=DIAMETER.shape)

    for i,diameter in enumerate(DIAMETER):
        dx = 1
        Npic =int(diameter*1.4)
        pic = np.zeros((Npic,Npic))
        dist_center_mat = np.linalg.norm(
            np.indices((Npic,Npic), dtype=int) -
            np.array([Npic//2,Npic//2]).reshape((2,1,1)), axis=0)
        
        pic[dist_center_mat < diameter//2] = 1

        # create four sliced 2D matrices in each i index
        mat_1_X = pic[:  , :-1].flatten()
        mat_2_X = pic[:  , 1: ].flatten()
        mat_1_Y = pic[:-1, :  ].flatten()
        mat_2_Y = pic[1: , :  ].flatten()

        # concatanate the sliced matrices
        mat_cumulative_X = np.stack((mat_1_X,mat_2_X), axis=-1)
        mat_cumulative_Y = np.stack((mat_1_Y,mat_2_Y), axis=-1)

        # find the number of unique phases in each neighborhood of size (2,2)
        sorted_X = np.sort(mat_cumulative_X,axis=1)
        unique_phases_X = (sorted_X[:,1:] != sorted_X[:,:-1]).sum(axis=1)+1
        unique_phases_X -= np.sum(np.isnan(sorted_X), axis=1)

        sorted_Y = np.sort(mat_cumulative_Y,axis=1)
        unique_phases_Y = (sorted_Y[:,1:] != sorted_Y[:,:-1]).sum(axis=1)+1
        unique_phases_Y -= np.sum(np.isnan(sorted_Y), axis=1)

        # find the locaiton of DPBs, where two phases are present in a 
        # (1,2) neighborhood (unique_phases==2).
        DPBs_flatten_X = np.zeros_like(mat_1_X).astype(bool)
        DPBs_flatten_X[unique_phases_X==2] = True

        DPBs_flatten_Y = np.zeros_like(mat_1_Y).astype(bool)
        DPBs_flatten_Y[unique_phases_Y==2] = True

        # resphase the flattened mask matrix (denoting the location of TPBs) to 2D
        DPBs_plane_X = DPBs_flatten_X.reshape((Npic,Npic-1))
        DPBs_plane_Y = DPBs_flatten_Y.reshape((Npic-1,Npic))

        length_voxel[i] = (np.sum(DPBs_plane_X) + np.sum(DPBs_plane_Y)) * dx
        length_real[i] = np.pi * diameter * dx

    ratio = length_real/length_voxel
    ratio[1:] = (ratio[1:] + ratio[:-1])/2
    df = pd.DataFrame({
        'diameter [pixels]': DIAMETER, 
        'Circle circumference, voxel)': length_voxel, 
        'Circle circumference, real': length_real,
        'ratio (real to voxel)': ratio})
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    trace1 = go.Scatter(x=df['diameter [pixels]'], y=df['Circle circumference, voxel)'], name='Circle circumference, voxel')
    trace2 = go.Scatter(x=df['diameter [pixels]'], y=df['Circle circumference, real'], name='Circle circumference, real')
    trace3 = go.Scatter(x=df['diameter [pixels]'], y=df['ratio (real to voxel)'], name='ratio (real to voxel)')
    fig.add_trace(trace1, secondary_y=False)
    fig.add_trace(trace2, secondary_y=False)
    fig.add_trace(trace3, secondary_y=True)
    fig.update_xaxes(title_text='diameter [pixels]')
    fig.update_yaxes(title_text="Circle circumference [µm]", secondary_y=False)
    fig.update_yaxes(title_text="ratio (real to voxel)", secondary_y=True)
    fig.show()
    return None

def PDF_analysis(scale, probability_value):
    from scipy.stats import norm
    import scipy.integrate as integrate
    from scipy.optimize import fsolve
    import plotly.graph_objects as go

    distribution_function = lambda x: norm.pdf(x, loc=0, scale=scale)
    probability = lambda x1: integrate.quad(distribution_function, -x1, x1)[0] - probability_value
    
    root = fsolve(probability, 3*scale)[0]

    step = 0.001
    whole_x = np.arange(-3*root, 3*root, step)
    whole_y = list(map(distribution_function, whole_x))

    needed_x = np.arange(-root, root, step)
    needed_y = list(map(distribution_function, needed_x))

    fig = go.Figure()
    trace1 = go.Scatter(x=whole_x, y=whole_y, mode='lines', name='PDF')
    trace2 = go.Scatter(x=needed_x, y=needed_y, mode='lines', fill='tozeroy', name='Needed area')
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.show()
    return root

def downscale_domain(domain, scale=2):
    # scale can only be 2, 4, 8, 16, ...
    if scale == 1:
        return domain

    for _ in range(int(np.log(scale)/np.log(2))):
        if domain.ndim == 2:
            mat1 = domain[:-1:2 , :-1:2].flatten()
            mat2 = domain[1::2  , :-1:2].flatten()
            mat3 = domain[:-1:2 , 1::2 ].flatten()
            mat4 = domain[1::2  , 1::2 ].flatten()
            all_mat = np.stack((mat1,mat2,mat3,mat4), axis=-1)
        
        elif domain.ndim == 3:
            mat1 = domain[:-1:2 , :-1:2 , :-1:2].flatten()
            mat2 = domain[1::2  , :-1:2 , :-1:2].flatten()
            mat3 = domain[:-1:2 , 1::2  , :-1:2].flatten()
            mat4 = domain[1::2  , 1::2  , :-1:2].flatten()
            mat5 = domain[:-1:2 , :-1:2 , 1::2 ].flatten()
            mat6 = domain[1::2  , :-1:2 , 1::2 ].flatten()
            mat7 = domain[:-1:2 , 1::2  , 1::2 ].flatten()
            mat8 = domain[1::2  , 1::2  , 1::2 ].flatten()
            all_mat = np.stack((mat1,mat2,mat3,mat4,mat5,mat6,mat7,mat8), axis=-1)
        
        all_mat_cnt_1 = np.sum(all_mat==1, axis=-1)
        all_mat_cnt_2 = np.sum(all_mat==2, axis=-1)
        all_mat_cnt_3 = np.sum(all_mat==3, axis=-1)

        phase_mat_ups = np.zeros_like(all_mat_cnt_1)
        phase_mat_ups[np.logical_and(all_mat_cnt_1>all_mat_cnt_2, all_mat_cnt_1>all_mat_cnt_3)] = 1
        phase_mat_ups[np.logical_and(all_mat_cnt_2>all_mat_cnt_1, all_mat_cnt_2>all_mat_cnt_3)] = 2
        phase_mat_ups[np.logical_and(all_mat_cnt_3>all_mat_cnt_1, all_mat_cnt_3>all_mat_cnt_2)] = 3

        # find indices of phase_mat_ups = 0
        indices_zero = np.where(phase_mat_ups==0)

        # indices of neghboring voxels to phase_mat_ups = 0
        indices_zero_up = tuple(np.array(indices_zero)-1)
        indices_zero_down = tuple(np.array(indices_zero)+1)

        # generate random number when phase_mat_ups = 0
        # random number is selected based on the value of upper and lower neighboring voxels
        for i in range(len(indices_zero[0])):
            try:
                rand_num = np.random.choice([phase_mat_ups[indices_zero_up[0][i]], phase_mat_ups[indices_zero_down[0][i]]])
                j = 1
                while rand_num == 0:
                    rand_num = np.random.choice([phase_mat_ups[indices_zero_up[0][i]-j], phase_mat_ups[indices_zero_down[0][i]+j]])
                    j += 1
            except:
                rand_num = np.random.choice(np.unique(domain))
            phase_mat_ups[indices_zero[0][i]] = rand_num

        shape_new = np.array(domain.shape)//2
        domain = phase_mat_ups.reshape(shape_new)

    return domain

def upscale_domain(domain, scale=2):
    """
    This function does not upscale the physical resolution of the domain.
    Instead it only makes the computational cells smaller. It does not affect the 
    spatial resolution of the initial image.
    """

    for _ in range(int(np.log(scale)/np.log(2))):
        shape_up = np.array(domain.shape)*scale
        domain_up = np.zeros(shape=shape_up)

        if domain.ndim == 3:
            domain_up[::2,::2,::2] = domain
            domain_up[1::2,::2,::2] = domain
            domain_up[::2,1::2,::2] = domain
            domain_up[1::2,1::2,::2] = domain
            domain_up[::2,::2,1::2] = domain
            domain_up[1::2,::2,1::2] = domain
            domain_up[::2,1::2,1::2] = domain
            domain_up[1::2,1::2,1::2] = domain
        elif domain.ndim == 2:
            domain_up[::2,::2] = domain
            domain_up[1::2,::2] = domain
            domain_up[::2,1::2] = domain
            domain_up[1::2,1::2] = domain

        domain = domain_up

    return domain



def measure_covariance(domain,phase=1):
    import plotly.graph_objects as go

    domain = domain==phase
    domain = domain.astype(bool)

    N = domain.shape
    cov = np.zeros(shape=(N[0]-1))

    # for k in range(N[2]):
    #     for j in range(N[1]):
    for i in range(N[0]-1):
        cov[i] = np.mean(np.logical_and(domain[0,:,:],domain[i+1,:,:]))

    fig = go.Figure()
    trace1 = go.Scatter(x=np.arange(1,N[0]), y=cov, mode='lines + markers', name='Covariance')
    fig.update_xaxes(title_text="Distance [voxels]")
    fig.update_yaxes(title_text="Covariance function")
    fig.add_trace(trace1)
    fig.show()

    return cov

def create_fibre(radius, length):
    fibre = np.zeros(shape=(length,2*radius,2*radius), dtype=int)
    # create i,j,k indices
    i = np.arange(fibre.shape[0])
    j = np.arange(fibre.shape[1])
    k = np.arange(fibre.shape[2])
    # create meshgrid
    I,J,K = np.meshgrid(i,j,k, indexing='ij')
    fibre[(J-radius)**2+(K-radius)**2 < radius**2] = 1

    # delete the first slice of the rod in j and k directions
    fibre = fibre[:,1:,1:]

    return fibre

def create_twisted_fibre(radius, length, amp=1, freq=1, phase=0, overlap=1):

    # create the rod
    # rod radius is increased by overlap
    new_radius = int(radius*overlap)
    fibre = create_fibre(new_radius, length)

    # create the wave function
    # amp is the amplitude of the wave function relative to the radius of the fibre
    AMP = amp * radius
    phi_1 = phase           # phase shift in the i direction
    phi_2 = phi_1 + np.pi/2 # phase shift in the j direction
    wave_1 = np.sin(np.arange(length)*freq*2*np.pi/length + phi_1)*AMP
    wave_2 = np.sin(np.arange(length)*freq*2*np.pi/length + phi_2)*AMP

    # make the rod bigger in j and k directions
    fibre = np.pad(fibre, 
                 ((0,0),(int(AMP),int(AMP)),(int(AMP),int(AMP))), 
                 'constant', constant_values=0)

    # create the twisted fibre
    for i in range(fibre.shape[0]):
        fibre[i,:,:] = np.roll(fibre[i,:,:], wave_1[i].astype(int), axis=0)
        fibre[i,:,:] = np.roll(fibre[i,:,:], wave_2[i].astype(int), axis=1)

    return fibre

def create_twisted_multifibre(radius, fibre_length, amp=1, freq=1, overlap=1, n_fibres=2):
    # if amp>1 then two twisted fibres are not touching each other.

    fibres = [None] * n_fibres
    phase = 0
    for n in range(n_fibres):
        fibres[n] = create_twisted_fibre(radius, fibre_length, amp, freq, phase=phase, overlap=overlap)
        fibres[n][fibres[n]==1] = n+1 # change the material of the new fibre
        phase += 2*np.pi/n_fibres

    # fibres = np.sum(fibres, axis=0) # add all the fibres together
    # fibres[fibres>n_fibres] = 1 # change the material of the overlapping region
    multi_fibre = np.zeros_like(fibres[0])
    for n in range(n_fibres):
        multi_fibre[fibres[n]==n+1] = n+1

    return multi_fibre

def rotate_3D_image(image, rotation=(0,0,0)):
    from scipy.ndimage import rotate

    image = rotate(image, rotation[0], axes=(1,2), order=0, prefilter=True)
    image = rotate(image, rotation[1], axes=(0,2), order=0, prefilter=True)
    image = rotate(image, rotation[2], axes=(0,1), order=0, prefilter=True)
    image = image.astype(int)

    return image

def create_fibrous_bed(
        voxels,
        radius, 
        fibre_length, 
        target_porosity,
        amp = 1, 
        freq = 1, 
        overlap = 1, 
        rotation_max = (0,0,0),
        bend_max = 1.5):

    # initialize the bed
    bed = np.zeros(shape=voxels, dtype=int)
    por_bed = 1

    # place each rod randomly in the bed
    while por_bed > target_porosity:
        fibre = create_twisted_multifibre(radius, fibre_length, amp, freq, overlap)
        if bend_max > 1:
            fibre = bend_fibre(fibre, bending_factor=np.random.uniform(low=1, high=bend_max), flip=np.random.choice([True, False]))
        fibre = rotate_3D_image(fibre, rotation=np.random.uniform(-1,1,3)*rotation_max)

        # find a random position in the bed
        i = np.random.randint(0, bed.shape[0])
        j = np.random.randint(0, bed.shape[1])
        k = np.random.randint(0, bed.shape[2])
        
        # identify start and end indices of the fibre domain in the bed
        i_start = i-fibre.shape[0]//2
        j_start = j-fibre.shape[1]//2
        k_start = k-fibre.shape[2]//2
        i_end = i_start + fibre.shape[0]
        j_end = j_start + fibre.shape[1]
        k_end = k_start + fibre.shape[2]

        # create a mask for the fibre domain
        fibre_mask = np.zeros_like(bed, dtype=int)
        
        if i_start < 0 or i_end > bed.shape[0]:
            i_start = i_start % bed.shape[0]
            i_end = i_end % bed.shape[0]
            fibre_mask[i_start:,:,:] += 1
            fibre_mask[:i_end,:,:] += 1
        else:
            fibre_mask[i_start:i_end, :, :] += 1

        if j_start < 0 or j_end > bed.shape[1]:
            j_start = j_start % bed.shape[1]
            j_end = j_end % bed.shape[1]
            fibre_mask[:,j_start:,:] += 1
            fibre_mask[:,:j_end,:] += 1
        else:
            fibre_mask[:, j_start:j_end, :] += 1

        if k_start < 0 or k_end > bed.shape[2]:
            k_start = k_start % bed.shape[2]
            k_end = k_end % bed.shape[2]
            fibre_mask[:,:,k_start:] += 1
            fibre_mask[:,:,:k_end] += 1
        else:
            fibre_mask[:, :, k_start:k_end] += 1
        
        # put the fibre in the bed in a random specific location
        fibre_mask = fibre_mask == 3
        start_eff = np.array([i_start, j_start, k_start])
        end_eff = np.array([i_end, j_end, k_end])
        put_fibre_in_bed(fibre, fibre_mask, bed, start_eff, end_eff)

        # calculate the new porosity of the bed
        por_bed = np.sum(bed==0)/bed.size
    
    # values of the bed are 1,2,3
    bed += 1
    return bed

def put_fibre_in_bed(fibre, f_mask, bed, start_eff, end_eff):
    cof = correct_overlapping_fibres
    # These are effective start and end points of the fibre
    i1, j1, k1 = start_eff
    i2, j2, k2 = end_eff
    N = bed.shape
    N_fibre = fibre.shape

    if np.any(np.array(N_fibre) > np.array(N)):
        # warning message and skip this fibre
        print('Fibre is too large for the bed. Skipping this fibre.')
        return

    if i1<i2 and j1<j2 and k1<k2:
        bed[f_mask] += fibre.flatten()

    elif i1>i2 and j1<j2 and k1<k2:
        bed[i1:,j1:j2,k1:k2][f_mask[i1:,j1:j2,k1:k2]] = cof(bed[i1:,j1:j2,k1:k2][f_mask[i1:,j1:j2,k1:k2]], fibre[:N[0]-i1,:,:].flatten())
        bed[:i2,j1:j2,k1:k2][f_mask[:i2,j1:j2,k1:k2]] = cof(bed[:i2,j1:j2,k1:k2][f_mask[:i2,j1:j2,k1:k2]], fibre[N[0]-i1:,:,:].flatten())
    
    elif i1<i2 and j1>j2 and k1<k2:
        bed[i1:i2,j1:,k1:k2][f_mask[i1:i2,j1:,k1:k2]] = cof(bed[i1:i2,j1:,k1:k2][f_mask[i1:i2,j1:,k1:k2]], fibre[:,:N[1]-j1,:].flatten())
        bed[i1:i2,:j2,k1:k2][f_mask[i1:i2,:j2,k1:k2]] = cof(bed[i1:i2,:j2,k1:k2][f_mask[i1:i2,:j2,k1:k2]], fibre[:,N[1]-j1:,:].flatten())
    
    elif i1<i2 and j1<j2 and k1>k2:
        bed[i1:i2,j1:j2,k1:][f_mask[i1:i2,j1:j2,k1:]] = cof(bed[i1:i2,j1:j2,k1:][f_mask[i1:i2,j1:j2,k1:]], fibre[:,:,:N[2]-k1].flatten())
        bed[i1:i2,j1:j2,:k2][f_mask[i1:i2,j1:j2,:k2]] = cof(bed[i1:i2,j1:j2,:k2][f_mask[i1:i2,j1:j2,:k2]], fibre[:,:,N[2]-k1:].flatten())
    
    elif i1>i2 and j1>j2 and k1<k2:
        bed[i1:,j1:,k1:k2][f_mask[i1:,j1:,k1:k2]] = cof(bed[i1:,j1:,k1:k2][f_mask[i1:,j1:,k1:k2]], fibre[:N[0]-i1,:N[1]-j1,:].flatten())
        bed[:i2,:j2,k1:k2][f_mask[:i2,:j2,k1:k2]] = cof(bed[:i2,:j2,k1:k2][f_mask[:i2,:j2,k1:k2]], fibre[N[0]-i1:,N[1]-j1:,:].flatten())
        bed[i1:,:j2,k1:k2][f_mask[i1:,:j2,k1:k2]] = cof(bed[i1:,:j2,k1:k2][f_mask[i1:,:j2,k1:k2]], fibre[:N[0]-i1,N[1]-j1:,:].flatten())
        bed[:i2,j1:,k1:k2][f_mask[:i2,j1:,k1:k2]] = cof(bed[:i2,j1:,k1:k2][f_mask[:i2,j1:,k1:k2]], fibre[N[0]-i1:,:N[1]-j1,:].flatten())
    
    elif i1>i2 and j1<j2 and k1>k2:
        bed[i1:,j1:j2,k1:][f_mask[i1:,j1:j2,k1:]] = cof(bed[i1:,j1:j2,k1:][f_mask[i1:,j1:j2,k1:]], fibre[:N[0]-i1,:,:N[2]-k1].flatten())
        bed[:i2,j1:j2,:k2][f_mask[:i2,j1:j2,:k2]] = cof(bed[:i2,j1:j2,:k2][f_mask[:i2,j1:j2,:k2]], fibre[N[0]-i1:,:,N[2]-k1:].flatten())
        bed[i1:,j1:j2,:k2][f_mask[i1:,j1:j2,:k2]] = cof(bed[i1:,j1:j2,:k2][f_mask[i1:,j1:j2,:k2]], fibre[:N[0]-i1,:,N[2]-k1:].flatten())
        bed[:i2,j1:j2,k1:][f_mask[:i2,j1:j2,k1:]] = cof(bed[:i2,j1:j2,k1:][f_mask[:i2,j1:j2,k1:]], fibre[N[0]-i1:,:,:N[2]-k1].flatten())
    
    elif i1<i2 and j1>j2 and k1>k2:
        bed[i1:i2,j1:,k1:][f_mask[i1:i2,j1:,k1:]] = cof(bed[i1:i2,j1:,k1:][f_mask[i1:i2,j1:,k1:]], fibre[:,:N[1]-j1,:N[2]-k1].flatten())
        bed[i1:i2,:j2,:k2][f_mask[i1:i2,:j2,:k2]] = cof(bed[i1:i2,:j2,:k2][f_mask[i1:i2,:j2,:k2]], fibre[:,N[1]-j1:,N[2]-k1:].flatten())
        bed[i1:i2,:j2,k1:][f_mask[i1:i2,:j2,k1:]] = cof(bed[i1:i2,:j2,k1:][f_mask[i1:i2,:j2,k1:]], fibre[:,N[1]-j1:,:N[2]-k1].flatten())
        bed[i1:i2,j1:,:k2][f_mask[i1:i2,j1:,:k2]] = cof(bed[i1:i2,j1:,:k2][f_mask[i1:i2,j1:,:k2]], fibre[:,:N[1]-j1,N[2]-k1:].flatten())
    
    elif i1>i2 and j1>j2 and k1>k2:
        bed[i1:,j1:,k1:][f_mask[i1:,j1:,k1:]] = cof(bed[i1:,j1:,k1:][f_mask[i1:,j1:,k1:]], fibre[:N[0]-i1,:N[1]-j1,:N[2]-k1].flatten())
        bed[:i2,:j2,:k2][f_mask[:i2,:j2,:k2]] = cof(bed[:i2,:j2,:k2][f_mask[:i2,:j2,:k2]], fibre[N[0]-i1:,N[1]-j1:,N[2]-k1:].flatten())
        bed[i1:,:j2,:k2][f_mask[i1:,:j2,:k2]] = cof(bed[i1:,:j2,:k2][f_mask[i1:,:j2,:k2]], fibre[:N[0]-i1,N[1]-j1:,N[2]-k1:].flatten())
        bed[:i2,j1:,k1:][f_mask[:i2,j1:,k1:]] = cof(bed[:i2,j1:,k1:][f_mask[:i2,j1:,k1:]], fibre[N[0]-i1:,:N[1]-j1,:N[2]-k1].flatten())
        bed[i1:,j1:,:k2][f_mask[i1:,j1:,:k2]] = cof(bed[i1:,j1:,:k2][f_mask[i1:,j1:,:k2]], fibre[:N[0]-i1,:N[1]-j1,N[2]-k1:].flatten())
        bed[:i2,:j2,k1:][f_mask[:i2,:j2,k1:]] = cof(bed[:i2,:j2,k1:][f_mask[:i2,:j2,k1:]], fibre[N[0]-i1:,N[1]-j1:,:N[2]-k1].flatten())
        bed[i1:,:j2,k1:][f_mask[i1:,:j2,k1:]] = cof(bed[i1:,:j2,k1:][f_mask[i1:,:j2,k1:]], fibre[:N[0]-i1,N[1]-j1:,:N[2]-k1].flatten())
        bed[:i2,j1:,:k2][f_mask[:i2,j1:,:k2]] = cof(bed[:i2,j1:,:k2][f_mask[:i2,j1:,:k2]], fibre[N[0]-i1:,:N[1]-j1,N[2]-k1:].flatten())

    return bed

def bend_fibre(fibre, bending_factor, flip=False):
    
    bf = int(bending_factor*fibre.shape[1])
    # pad the fibre domain
    fibre = np.pad(
        fibre,((0,0),(0,bf),(0,0)),
        'constant', constant_values=0)
    
    # bend the fibre using np.roll
    length = fibre.shape[0]
    wave = np.sin(np.arange(length)*np.pi/length)*bf
    for i in range(length):
        fibre[i,...] = np.roll(fibre[i,...], wave[i].astype(int), axis=0)

    if flip:
        fibre = np.flip(fibre, axis=1)
        
    return fibre

def add_roughness(domain_smooth, d_rough, labels=[1,2]):
    from scipy.ndimage.morphology import binary_dilation

    if d_rough==0:
        Warning('d_rough is 0, no roughness added')
        return domain_smooth
    
    prob = 1/(d_rough*1.5)      # probability of initial roughness points
    R = d_rough//2            # radius of roughness points

    rng_0 = np.random.default_rng(seed=16)
    rng_1 = np.random.default_rng(seed=25)
    
    # extract the label values from domain
    domain_rough_0 = np.zeros_like(domain_smooth, dtype=bool)
    domain_rough_0[domain_smooth==labels[0]] = True 

    domain_rough_1 = np.zeros_like(domain_smooth, dtype=bool)
    domain_rough_1[domain_smooth==labels[1]] = True 

    domain_both_labels = np.zeros_like(domain_smooth, dtype=bool)
    domain_both_labels[np.logical_or(domain_rough_0,domain_rough_1)] = True
    
    if domain_smooth.ndim==2:
        struct = np.zeros((2*R+1,2*R+1), dtype=bool)
        I,J = np.indices(struct.shape)
        struct[(I-R)**2+(J-R)**2<=R**2] = True
    elif domain_smooth.ndim==3:
        struct = np.zeros((2*R+1,2*R+1,2*R+1), dtype=bool)
        I,J,K = np.indices(struct.shape)
        struct[(I-R)**2+(J-R)**2+(K-R)**2<=R**2] = True
    
    surf_thin = measure_interface(domain_smooth, labels = labels)

    # randomly add roughness on the thin surface
    added_pts_0 = np.zeros_like(surf_thin, dtype=bool)
    added_pts_0[surf_thin] = rng_0.choice([False,True], size=surf_thin[surf_thin].size, p=[1-prob/2,prob/2])
    added_pts_0 = binary_dilation(added_pts_0, structure=struct)
    domain_rough_0 = np.logical_or(domain_rough_0,added_pts_0)

    added_pts_1 = np.zeros_like(surf_thin, dtype=bool)
    added_pts_1[surf_thin] = rng_1.choice([False,True], size=surf_thin[surf_thin].size, p=[1-prob/2,prob/2])
    added_pts_1 = binary_dilation(added_pts_1, structure=struct)
    domain_rough_1 = np.logical_or(domain_rough_1,added_pts_1)
    
    final_rough = np.zeros(shape = domain_rough_0.shape, dtype=int)
    final_rough[np.logical_and(domain_rough_0,~domain_rough_1)] = labels[0]
    final_rough[np.logical_and(domain_rough_1,~domain_rough_0)] = labels[1] 
    final_rough[np.logical_and(added_pts_0,final_rough==0)] = labels[0]
    final_rough[np.logical_and(added_pts_1,final_rough==0)] = labels[1]

    final_rough[~domain_both_labels] = domain_smooth[~domain_both_labels]

    return final_rough

def add_roughness_all_phases(domain_smooth, labels=None, iteration=1, d_rough=6):
    if labels is None:

        labels = [[1,2],[1,3],[2,3]] if len(np.unique(domain_smooth))==3 else [[1,2]]
    else:
        labels = [labels]

    domain_rough = domain_smooth
    D = d_rough

    for _ in range(iteration):
        
        for l in labels:
            domain_rough = add_roughness(
                domain_rough, 
                labels=l,  
                d_rough=D)
            
        D = D//2
    
    return domain_rough

def measure_interface(domain, labels=None, output='binary'):
    from scipy.ndimage.morphology import binary_dilation
    
    if labels is None:
        labels = list(np.unique(domain)[:-1])
    
    interface = np.logical_xor(
        binary_dilation(domain==labels[0]), 
        domain==labels[0])
    
    if len(labels) == 2:
        interface = np.logical_and(
            interface, 
            domain==labels[1]) 

    if output == 'binary':
        pass
    elif output == 'nans':
        interface = interface.astype(float)
        interface[interface==0] = np.nan

    return interface

def create_circle(diameter, output='total'):
    I,J = np.indices((diameter,diameter))
    circle = np.zeros((diameter,diameter))+1
    circle[(I-diameter//2)**2+(J-diameter//2)**2 <= (diameter//2)**2] = 2
    
    # pad the circle
    circle = np.pad(circle,
                    ((diameter//10,diameter//10),(diameter//10,diameter//10)),
                    'constant', constant_values=1)
    
    N = circle.shape[0]
    if output == 'total':
        return circle
    elif output == 'half':
        return circle[N//2:,:]
    elif output == 'quarter':
        return circle[N//2:,:N//2]
    
def correct_overlapping_fibres(location_in_bed, new_fibre):
    location_in_bed += new_fibre
    location_in_bed[location_in_bed==4] = 2
    location_in_bed[location_in_bed==3] = new_fibre[location_in_bed==3]
    location_in_bed[np.logical_and(location_in_bed==2, new_fibre!=0)] = new_fibre[np.logical_and(location_in_bed==2, new_fibre!=0)]

    return location_in_bed

def create_vertices_in_uniform_grid(N):
    x = np.arange(N[0]+1)
    y = np.arange(N[1]+1)
    z = np.arange(N[2]+1)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return vertices