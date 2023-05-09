if __name__ == '__main__':
    from modules import topology as tpl
    import numpy as np

    inputs = {
        "microstructure": 
        {
            "dx": 10e-09,
            "volume_fractions": {
                "pores": 0.44,
                "Ni": 0.28,
                "YSZ": 0.28
            },
            "length": {
                "X": 5e-6,
                "Y": 1e-6,
                "Z": 1e-6
            },
            "lattice_geometry": {
                "flag": True,
                "particle_diameter": 0.4e-06
            },
            "plurigaussian": {
                "flag": False,
                "sig_gen": 3,
                "gradient_factor": 1,
                "seed": [30,20],
                "reduced_geometry": {
                    "flag": False,
                    "Nx_extended": 1000
                }
            },
            "infiltration_loading": 0.01
        }}
    
    # tpl.compare_circle_circumference(5,100)
    sigma_seg = 1.5   # Segmegation parameter
    show_microstructure = False
    show_TPB = True
    show_segmentation = True
    show_histograms = True

    domain = tpl.create_microstructure(inputs, display=show_microstructure)

    domain, labeled_pores, percolating_labels\
        = tpl.percolation_analysis(domain)

    # triple phase boundary analysis
    dx = inputs['microstructure']['dx']
    TPBs, TPB_density, vertices, lines = tpl.measure_TPB(domain, dx)
    TPB_density = TPB_density / 1e12      # [μm^-2]

    if show_TPB:
        import pyvista as pv
        TPB_mesh = pv.PolyData(vertices, lines=lines)
        from modules.postprocess import visualize_mesh as vm
        vm([domain], [(2,3)], clip_widget=False, TPB_mesh=TPB_mesh)


    # image segmentation 
    labels, dist_mat, volumes, centroids\
        = tpl.image_segmentation(domain,sigma_seg, display=show_segmentation)
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

    # print results
    N = [inputs['microstructure']['Nx'], inputs['microstructure']['Ny'], inputs['microstructure']['Nz']]
    print(f'd_avg = {d_avg[0]:.2}, {d_avg[1]:.2}, {d_avg[2]:.2} [μm]')
    print(f'TPB_density = {TPB_density:.4} [μm^-2]')
    print(f'Length to pore/particle diameter ratio = {N[0]*dx*1e6/d_avg[0]:.2f}, {N[1]*dx*1e6/d_avg[1]:.2f}, {N[2]*dx*1e6/d_avg[2]:.2f}')

    if show_histograms:
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