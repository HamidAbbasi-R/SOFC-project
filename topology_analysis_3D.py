if __name__ == '__main__':
    from modules import topology as tpl
    import numpy as np

    N_walks = [    # Number of voxels in each direction
        100,
        100,
        100,
        ]

    sigma_gen = [2,6]   # Generation parameter
    sigma_seg = 1.5   # Segmegation parameter
    dx = 90e-9      # Voxel size [m]
    show_TPB = False

    phase_mat = tpl.create_phase_data(
        voxels = N_walks,
        vol_frac = [0.28, 0.28, 0.44],
        sigma = sigma_gen,
        seed = [50,10],
        display = False)

    import numpy as np
    phase_mat_nans, labeled_pores, percolating_labels\
        = tpl.percolation_analysis(phase_mat)
    
    isa_12_mat, isa_23_mat, isa_31_mat, isa_12, isa_23, isa_31 = tpl.interfacial_surface(phase_mat_nans)

    # triple phase boundary analysis
    TPBs, TPB_density, vertices, lines = tpl.measure_TPB(phase_mat_nans, dx)
    TPB_density = TPB_density / 1e12      # [μm^-2]
    if show_TPB:
        import pyvista as pv
        TPB_mesh = pv.PolyData(vertices, lines=lines)
        from modules.postprocess import visualize_mesh as vm
        vm([phase_mat_nans], [(3,3)], clip_widget=False, TPB_mesh=TPB_mesh)


    # image segmentation 
    labels, dist_mat, phase_mat_nans, percolating_labels, volumes, centroids\
        = tpl.image_segmentation(phase_mat,sigma_seg, display=True)
    volumes[0] = volumes[0] * dx**3 * 1e18      # [μm^3]
    volumes[1] = volumes[1] * dx**3 * 1e18      # [μm^3]
    volumes[2] = volumes[2] * dx**3 * 1e18      # [μm^3]

    radius = [0,0,0]
    radius[0] = (volumes[0] / np.pi * 3/4)**(1/3)    # [μm]
    radius[1] = (volumes[1] / np.pi * 3/4)**(1/3)    # [μm]
    radius[2] = (volumes[2] / np.pi * 3/4)**(1/3)    # [μm]

    d_avg = [0,0,0]
    d_avg[0] = np.mean(radius[0])*2    # [μm]
    d_avg[1] = np.mean(radius[1])*2    # [μm]
    d_avg[2] = np.mean(radius[2])*2    # [μm]
    print(f'd_avg = {d_avg[0]:.2}, {d_avg[1]:.2}, {d_avg[2]:.2} [μm]')
    print(f'TPB_density = {TPB_density:.4} [μm^-2]')
    print(f'Length to pore/particle diameter ratio = {np.max(N_walks)*dx*1e6/d_avg[0]:.2f}, {np.max(N_walks)*dx*1e6/d_avg[1]:.2f}, {np.max(N_walks)*dx*1e6/d_avg[2]:.2f}')

    # volumes = [0,0,0,0]
    # for i,sigma in enumerate([1,3,5,7]):
    #     _, _, _, _, volumes[i], _\
    #         = ms.image_segmentation(phase_mat,sigma)

    import plotly.graph_objects as go
    fig = go.Figure()
    max0 = np.max(volumes[0])
    max1 = np.max(volumes[1])
    max2 = np.max(volumes[2])
    maxt = np.max([max0,max1,max2])
    Nhist = 100

    fig.add_trace(go.Histogram(
        x=2*radius[0][:,0], 
        xbins=dict( # bins used for histogram
            start=0,
            end=maxt,
            size=maxt/Nhist)
            ))

    fig.add_trace(go.Histogram(
        x=2*radius[1][:,0], 
        xbins=dict( # bins used for histogram
            start=0,
            end=maxt,
            size=maxt/Nhist)
            ))

    fig.add_trace(go.Histogram(
        x=2*radius[2][:,0], 
        xbins=dict( # bins used for histogram
            start=0,
            end=maxt,
            size=maxt/Nhist)
            ))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.5)
    fig.show()

    # export to matlab for mesh generation via iso2mesh toolbox
    # import scipy.io
    # mdic = {"phase_mat": phase_mat,
    #         "percolating_label_1": percolating_labels[:,:,:,0],
    #         "percolating_label_2": percolating_labels[:,:,:,1],
    #         "percolating_label_3": percolating_labels[:,:,:,2]}
    # scipy.io.savemat('microstructure.mat', mdic)

    #%% VTK visualisation

    # mat1 = phase_mat
    # mat1 = percolating_labels[:,:,:,0]
    # mat2 = labels[:,:,:,0]
    # mat3 = isa_31_mat

    # mat_list = [mat1]

    # thd1 = ()
    # thd1 = (1,np.nanmax(dist_mat[:,:,:,0]))
    # thd2 = ()
    # thd3 = ()

    # thd_list = [thd1]

    # blocks = ms.visualize_network(volumes, centroids)

    # ms.visualize_mesh(mat_list, thd_list)


    #%% VTK visualisation [TPB]
    # mesh_TPB = pv.PolyData(vertices, lines=lines)

    # p = pv.Plotter()

    # p.subplot(0, 0)
    # p.add_mesh(mesh_TPB_all, line_width=2, color='k')
    # p.add_title('Before')

    # p.subplot(0, 1)
    # p.add_mesh(mesh_TPB, line_width=2, color='k')
    # p.add_title('After')

    # p.link_views()
    # p.show()
    # p.renderer.add_border(color='Black')
