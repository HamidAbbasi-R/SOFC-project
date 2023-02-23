def create_phase_data(voxels, vol_frac, sigma, mode="normal", seed=[], periodic=True, display=False, histogram='none'):
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
    import numpy as np
    # Apparently, skimage has some issues with being called from matlab.
    # This package is used herer to apply Gaussian filter on the random matrices.
    # Instead of using skimage, Gaussian filter implemented in scipy package can be used
    # to circumvent this issue. scipy.ndimage.gaussian_filter
    # import skimage
    from scipy.ndimage import gaussian_filter
    import plotly.express as px

    print("Generating microstructure...", end='')
    if sum(vol_frac) - 1 > 1e-2:
        raise NameError('Error! sum of volume fractions must be equal to 1')
    else:
        vol_frac = vol_frac[:-1]

    dim=len(voxels)
    
    Nx = voxels[0]-1
    Ny = voxels[1]-1
    if dim==3:
        Nz = voxels[2]-1
    
    # create the random generator with specified seed
    if len(seed) == 0:
        rng_1 = np.random.default_rng()
        rng_2 = np.random.default_rng()
    else:
        rng_1 = np.random.default_rng(seed=seed[0])
        rng_2 = np.random.default_rng(seed=seed[1])

    # create two random matrices (two matrices are necessary
    # for three phase porous media)
    voxels = np.array(voxels)
    rand_mat_1 = rng_1.random(size=tuple(voxels-1))
    rand_mat_2 = rng_2.random(size=tuple(voxels-1))

    # rand_mat_1 = np.random.normal(size=tuple(voxels-1))
    # rand_mat_2 = np.random.normal(size=tuple(voxels-1))

    # create a tile matrix, according to the dimension
    if periodic==True:
        tile_mat = tuple(np.ones(dim,dtype=int)*3)
    else:
        tile_mat = tuple(np.ones(dim,dtype=int)*1)

    # apply the Gaussian filter on the tiled random matrices. tiled random matrices
    # are necessary to create a periodic phase matrix
    # smooth_mat_1 = skimage.filters.gaussian(np.tile(rand_mat_1,tile_mat),\
    #      sigma=sigma, mode='reflect')
    # smooth_mat_2 = skimage.filters.gaussian(np.tile(rand_mat_2,tile_mat),\
    #      sigma=sigma, mode='reflect')
    smooth_mat_1 = gaussian_filter(np.tile(rand_mat_1,tile_mat),\
            sigma=sigma, mode='reflect')
    smooth_mat_2 = gaussian_filter(np.tile(rand_mat_2,tile_mat),\
            sigma=sigma, mode='reflect')

    # extract center matrices from the tiled smooth matrices
    if dim==2 and periodic==True:
        smooth_mat_1 = smooth_mat_1[Nx:2*Nx+1,Ny:2*Ny+1]
        smooth_mat_2 = smooth_mat_2[Nx:2*Nx+1,Ny:2*Ny+1]
    elif dim==3 and periodic==True:
        smooth_mat_1 = smooth_mat_1[Nx:2*Nx+1,Ny:2*Ny+1,Nz:2*Nz+1]
        smooth_mat_2 = smooth_mat_2[Nx:2*Nx+1,Ny:2*Ny+1,Nz:2*Nz+1]

    # create different modes of microstructure with thresholding the smooth matrices
    if mode=="normal":
        if len(vol_frac)==2:    # phsaes: 1-2-3
            phase_mat = np.zeros_like(smooth_mat_1, dtype=int)+3
            a = np.quantile(smooth_mat_1.flatten(), q=vol_frac[0])
            b = np.quantile(smooth_mat_2[smooth_mat_1 >= a].flatten(), q=vol_frac[1]/(1-vol_frac[0]))
            phase_mat[smooth_mat_1 < a] = 1        
            phase_mat[np.logical_and(smooth_mat_2 < b, smooth_mat_1 > a)] = 2
        if len(vol_frac)==1:    # phases: 0-1
            phase_mat = np.zeros_like(smooth_mat_1, dtype=int)
            a = np.quantile(smooth_mat_1.flatten(), q=vol_frac)
            phase_mat[smooth_mat_1 < a] = 1
    elif mode=="thin film":
        a = np.quantile(smooth_mat_1.flatten(), q=vol_frac[0])
        b = np.quantile(smooth_mat_1.flatten(), q=1-vol_frac[1])
        phase_mat[smooth_mat_1 < a] = 2
        phase_mat[smooth_mat_1 > b] = 3
    
    # display the phase matrix
    if display==True and dim==2:
        fig = px.imshow(np.rot90(phase_mat))
        fig.write_image("fig1.pdf")
        fig.show()
    elif display==True and dim==3:
        # from postprocess import visualize_mesh
        from modules.postprocess import visualize_mesh as vm 
        vm([phase_mat],[()])

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

    print('Done!')
    return phase_mat

def measure_TPB(phase_mat, dx):
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
    import numpy as np
    from tqdm import tqdm
    from scipy import ndimage as ndi

    # extract the percolating phase matrix    
    # phase_mat, _, _= percolation_analysis(phase_mat)

    # print("Extracting TPB...", end='')
    N=phase_mat.shape
    TPBs = np.empty((0,6), int)
    TPB_mask = np.zeros_like(phase_mat, dtype=bool)

    # loop through the x axis     
    for i in np.arange(N[0]):
        # create four sliced 2D matrices in each i index
        mat_1 = phase_mat[i , 0:-1 , 0:-1 ].flatten()
        mat_2 = phase_mat[i , 0:-1 , 1:   ].flatten()
        mat_3 = phase_mat[i , 1:   , 0:-1 ].flatten()
        mat_4 = phase_mat[i , 1:   , 1:   ].flatten()

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
        mat_1 = phase_mat[0:-1 , j , 0:-1 ].flatten()
        mat_2 = phase_mat[0:-1 , j , 1:   ].flatten()
        mat_3 = phase_mat[1:   , j , 0:-1 ].flatten()
        mat_4 = phase_mat[1:   , j , 1:   ].flatten()

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
        mat_1 = phase_mat[0:-1 , 0:-1 , k].flatten()
        mat_2 = phase_mat[0:-1 , 1:   , k].flatten()
        mat_3 = phase_mat[1:   , 0:-1 , k].flatten()
        mat_4 = phase_mat[1:   , 1:   , k].flatten()

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

    # This section needs revision, there must be some ways to get rid of the nested
    # loops and make the run faster. It is not a big deal however.
    count = 0
    vertices = np.ones(shape=((N[0]+1)*(N[1]+1)*(N[2]+1),3))
    for i in np.arange(N[0]+1):
        for j in np.arange(N[1]+1):
            for k in np.arange(N[2]+1):
                vertices[count,:] = [i,j,k]
                count += 1
    
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
    return TPB_mask, TPB_density, vertices, lines

def measure_TPB_notvec(phase_mat):
    """
    obsolete and non-vectorized version of measure_TPB_vec. the runtime of 
    this function is much higher than the vectorized version.
    """
    import numpy as np
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
    import numpy as np
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
    percolating_label_1 = np.ones(shape = lw1_shuffled.shape)
    percolating_label_2 = np.ones(shape = lw2_shuffled.shape)+1
    percolating_label_3 = np.ones(shape = lw3_shuffled.shape)+2
    
    percolating_label_1[lw1_shuffled!=intersection_x_1[0]] = np.nan
    percolating_label_2[lw2_shuffled!=intersection_x_2[0]] = np.nan
    percolating_label_3[lw3_shuffled!=intersection_x_3[0]] = np.nan
    
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
    import numpy as np
    num = np.max(labeled_mat)
    b = np.arange(start=1, stop=num+1)
    np.random.shuffle(b)
    b = np.append(0,b)
    shuffled_mat = b[labeled_mat.astype(int)]
    shuffled_mat = shuffled_mat.astype(float)
    shuffled_mat[shuffled_mat==0] = np.nan
    return shuffled_mat

def ISA(phase_mat):
    """
    Measures the interfacial surface between different phases.

    Parameters
    ----------
    phase_mat : float
        Three dimensional matrix describing the phase data.

    Returns
    -------
    isa_12_mat : integer
        Three dimensional matrix describing the interfacial 
        surface area (ISA) between the 1st and the 2nd phases.
    isa_23_mat : integer
        Same as above. But for 2nd and 3rd phases.
    isa_31_mat : integer
        Same as above. But for 3rd and 1st phases.
    isa_12 : float
        Normalized interfacial surface area (m2/m3)
    """
    from scipy import ndimage as ndi
    import numpy as np
    
    A1 = phase_mat == 1
    A2 = phase_mat == 2
    A3 = phase_mat == 3
    
    B1 = ndi.binary_dilation(A1).astype(A1.dtype)
    B2 = ndi.binary_dilation(A2).astype(A2.dtype)
    B3 = ndi.binary_dilation(A3).astype(A3.dtype)
    
    C1 = B1 - A1.astype(int)
    C2 = B2 - A2.astype(int)
    C3 = B3 - A3.astype(int)
    
    isa_12_mat = C1 * A2.astype(int)
    isa_23_mat = C2 * A3.astype(int)
    isa_31_mat = C3 * A1.astype(int)
    
    isa_12 = np.sum(isa_12_mat) / np.power(phase_mat.shape[0],3)
    isa_23 = np.sum(isa_23_mat) / np.power(phase_mat.shape[0],3)
    isa_31 = np.sum(isa_31_mat) / np.power(phase_mat.shape[0],3)
    
    return isa_12_mat, isa_23_mat, isa_31_mat, isa_12, isa_23, isa_31

def image_segmentation(phase_mat,sigma=5):
    """ 
    Segmentation of the phase matrix.
    Inputs:
    phase_mat : float
        Three dimensional matrix describing the phase data.
    
    fp : float
        footprint
        
    Outputs:
    labels:"""
    from tqdm import tqdm
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.filters import gaussian
    import numpy as np
    
    # import plotly.io as po
    # from plotly.subplots import make_subplots
    # import plotly.graph_objects as go
    
    # po.renderers.default = "browser"
    
    phase_mat_nans, _, percolating_labels = percolation_analysis(phase_mat)
        
    labels = np.zeros(shape = np.append(phase_mat.shape,3))
    dist_mat = np.zeros(shape = np.append(phase_mat.shape,3))
    
    # volumes = np.empty([0,3])
    volumes = [[],[],[]]
    centroids = [[],[],[]]
    
    for i in [1,2,3]:
        phase = phase_mat_nans == i
        dist_mat[:,:,:,i-1] = ndi.distance_transform_edt(phase.astype(int))
        dist_mat[:,:,:,i-1] = gaussian(dist_mat[:,:,:,i-1], sigma=sigma, mode='nearest')
        dist_mat[:,:,:,i-1][~phase] = 0
        # visualize_mesh([dist_mat[:,:,:,i-1]], [(0.0001,1000)])
        # coords = peak_local_max(dist_mat[:,:,:,i-1], footprint=fp, min_distance=1)
        # mask = np.zeros(dist_mat[:,:,:,i-1].shape, dtype=bool)
        # mask[tuple(coords.T)] = True
        # markers, _ = ndi.label(mask)
        labels[:,:,:,i-1] = watershed(-dist_mat[:,:,:,i-1], mask=phase)
        labels[:,:,:,i-1] = shuffle_labels(labels[:,:,:,i-1])
        
        volumes[i-1] = np.zeros((np.nanmax(labels[:,:,:,i-1]).astype(int),1))
        centroids[i-1] = np.zeros((np.nanmax(labels[:,:,:,i-1]).astype(int),3))
        
        for particle in tqdm(np.arange(np.nanmax(labels[:,:,:,i-1]), dtype=int)):
            x = np.where(labels[:,:,:,i-1] == particle+1)
            centroids[i-1][particle,:] = np.average(x,axis=1)
            volumes[i-1][particle] = len(x[0])
    
    return labels, dist_mat, phase_mat_nans, percolating_labels, volumes, centroids

def remove_thin_boundaries(phase_mat):
    """
    Removes thin boundaries from the phase matrix. It is a good practice 
    to remove thin boundaries before running a simulation with 
    Neumann boundary conditions.

    inputs:
    phase_mat : float
        Three dimensional matrix describing the phase data.
    """
    import numpy as np
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
    import numpy as np
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
    
def create_microstructure(inputs):
    # create the entire domain
    Nx = inputs['microstructure']['Nx']
    Ny = inputs['microstructure']['Ny']
    Nz = inputs['microstructure']['Nz']

    vol_frac = [inputs['microstructure']['volume_fractions']['pores'],
                inputs['microstructure']['volume_fractions']['Ni'],
                inputs['microstructure']['volume_fractions']['YSZ']]

    domain = create_phase_data(
        voxels = [Nx,Ny,Nz],
        vol_frac = vol_frac,
        sigma = inputs['microstructure']['sig_gen'],
        seed = inputs['microstructure']['seed'],
        display = True,
        )
    return domain

def topological_operations(inputs, domain):
    """
    domain topological operations
    """
    print("Domain topological operations...", end='')
    # removing thin boundaries
    # remove thin boundaries to avoid numerical error for the case of Neumann BCs.
    # don't remove thin boundaries for periodic boundary conditions. it will cause problems.
    domain = remove_thin_boundaries(domain.astype(float))
    # extract the domain that should be solved. ds is short for Domain for Solver.
    # when periodic boundary condition is used, percolation analysis should not be done.
    domain, _, _ = percolation_analysis(domain)

    # measure the triple phase boundary and create a mask for source term
    TPB_mask, TPB_density, vertices, lines = measure_TPB(domain, inputs['microstructure']['dx'])
    print("Done!")

    TPB_dict = {
        'TPB_mask': TPB_mask,
        'TPB_density': TPB_density,
        'vertices': vertices,
        'lines': lines
    }
    return domain, TPB_dict

# specific functions for entire cell microstructure
def create_microstructure_entire_cell(inputs):
    import numpy as np
    
    domain_a = create_phase_data(
        voxels = [inputs['Nx_a'],inputs['Ny'],inputs['Nz']],
        vol_frac = [inputs['vf_pores_a'],inputs['vf_Ni_a'],inputs['vf_YSZ_a']],
        sigma = inputs['sig_gen_a'],
        seed = inputs['seed'],
        display = False,
        )
    
    domain_c = create_phase_data(
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
    import numpy as np
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
