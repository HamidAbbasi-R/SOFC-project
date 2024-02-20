if __name__ == '__main__':
    from modules import topology as tpl
    import numpy as np

    inputs = {
        "microstructure": 
        {
            "dx": 50e-09,
            "average_diameter": 500e-9,
            "volume_fractions": {
                "pores": 0.4,
                "Ni": 0.3,
                "YSZ": 0.3
            },
            "type": "plurigaussian",
            "length": {
                "X": 3e-6,
                "Y": 3e-6,
                "Z": 3e-6
            },
            "plurigaussian": {
                "gradient_factor": 1,
                "seed": [50,40],
                "reduced_geometry": {
                    "flag": False,
                    "Lx_extended": 10e-6
                }
            },
            "roughness_flag": False,
            "scale_factor": 1,
            "infiltration_loading": 0.00
        }}
    
    # tpl.compare_circle_circumference(5,100)
    sigma_seg = 1.5   # Segmegation parameter
    show_microstructure = False
    show_TPB = False
    show_segmentation = True
    show_histograms = True

    domain = tpl.create_microstructure(inputs, display=show_microstructure)

    domain, labeled_pores, percolating_labels\
        = tpl.percolation_analysis(domain)

    # triple phase boundary analysis
    dx = inputs['microstructure']['dx']
    TPB_mask, TPB_density, lines, TPB_dist_x = tpl.measure_TPB(domain, dx)
    TPB_density = TPB_density / 1e12      # [μm^-2]

    if show_TPB:
        import pyvista as pv
        TPB_mesh = pv.PolyData(tpl.create_vertices_in_uniform_grid(domain.shape()), lines=lines)
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
    L = [
        inputs['microstructure']['length']['X'], 
        inputs['microstructure']['length']['Y'], 
        inputs['microstructure']['length']['Z']]
    print(f'd_avg = {d_avg[0]:.2}, {d_avg[1]:.2}, {d_avg[2]:.2} [μm]')
    print(f'TPB_density = {TPB_density:.4} [μm^-2]')
    print(f'Length to pore/particle diameter ratio = {L[0]*1e6/d_avg[0]:.2f}, {L[1]*1e6/d_avg[1]:.2f}, {L[2]*1e6/d_avg[2]:.2f}')

    if show_histograms:
        import plotly.graph_objects as go
        max0 = np.max(volumes[0])
        max1 = np.max(volumes[1])
        max2 = np.max(volumes[2])
        maxt = np.max([max0,max1,max2])
        Nhist = 50

        fig = go.Figure()
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