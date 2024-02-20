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

def visualize_residuals(residuals):
    import pandas as pd
    import plotly.express as px

    # visualize the error
    x = np.arange(len(residuals[2]))
    r = [None]*3
    df = [None]*3
    titles = ['Hydrogen concentration', 'Electron potential', 'Ion potential']

    phase_names = ['gas', 'elec', 'ion']
    for p in [0,1,2]:
        if phase_names[p] not in inputs['solver_options']['transport_eqs']:
            continue
        r[p] = np.stack((x, residuals[p]), axis=-1)
        df[p] = pd.DataFrame(r[p], columns=['iteration', 'residual'])
        df[p].insert(2, 'Variable', titles[p])

    df = pd.concat(df)
    fig = px.line(df, x='iteration', y='residual', color='Variable', log_y=True)
    fig.update_yaxes(exponentformat="e")
    fig.show()

def postprocessing(
        phi,
        masks_dict,
        indices,
        field_functions,
        TPB_dict,
        K,
        K_array_ion,
        plots,
        ):
    # visualize the solution
    
    dense_m = create_dense_matrices(
        phi,
        masks_dict,
        indices,
        field_functions,
        TPB_dict,
        K_array_ion,
        )
    
    ds = masks_dict['ds']
    N = [ds[i].shape for i in range(len(ds))]
    Lx_a = N[0][0]
    Lx_e = N[1][0] - N[0][0] - N[3][0]
    Lx_c = N[3][0]
    Lx_tot = Lx_a + Lx_e + Lx_c

    img_flag = inputs['output_options']['img_output']
    csv_flag = inputs['output_options']['csv_output']
    plot1D_flag = inputs['output_options']['show_1D_plots']
    plot3D_flag = inputs['output_options']['show_3D_plots']

    if plot3D_flag:
        import pyvista as pv
        # lines = TPB_dict['lines']
        pv.set_plot_theme("document")
        from modules.topology import create_vertices_in_uniform_grid as cvug
        # TPB_mesh = pv.PolyData(cvug(dense_m['phi_dense'].shape), lines=lines)
        TPB_mesh_a = pv.PolyData(
            cvug(N[0]), 
            lines=TPB_dict['anode']['lines'],
            )
        vertices_c = cvug(N[3])
        vertices_c[:,0] += Lx_a + Lx_e
        TPB_mesh_c = pv.PolyData(
            vertices_c, 
            lines=TPB_dict['cathode']['lines'],
            )
    
    dx = inputs['microstructure']['dx']
    mats = []
    thds = []
    log_scale = []
    titles = []

    # 1D phi plots
    x = np.arange(Lx_tot)*dx*1e6
    vars = ['rhoH2', 'Vel', 'Vio', 'rhoO2']

    for var in vars:
        if plots[f'{var}_1D'] and (plot1D_flag or csv_flag):
            var_avg,var_max,var_min,var_c_down,var_c_up = [np.zeros(Lx_tot)] * 5
            for i in range(Lx_tot):
                a = dense_m[var][i, :, :][~np.isnan(dense_m[var][i, :, :])]
                var_avg[i] = np.average(a) if a.size>0 else np.nan
                var_max[i] = np.max(a) if a.size>0 else np.nan
                var_min[i] = np.min(a) if a.size>0 else np.nan
                var_c_down[i], var_c_up[i] = mean_confidence_interval(a) if a.size>0 else (np.nan, np.nan)
            if csv_flag: create_csv_output(x, var_avg, var_min, var_max, var_c_down, var_c_up, f'{var}_{inputs["file_options"]["id"]}')
            if plot1D_flag: plot_with_continuous_error(x, var_avg, var_c_down, var_c_up, x_title='Distance from anode (µm)', y_title=var, title=f'{var}_{inputs["file_options"]["id"]}', save_img=img_flag)
    
    # 1D volumetric charge transfer rate plot [NEW]
    if True:
        I_vol = np.zeros(Lx_tot) * np.nan
        I_vol[1:Lx_a-1] = np.nanmean(dense_m['I'][1:Lx_a-1, ...], axis=(1,2)) # [A/m3]
        I_vol[Lx_a+Lx_e+1:-1] = np.nanmean(dense_m['I'][Lx_a+Lx_e+1:-1, ...], axis=(1,2)) # [A/m3]
        if csv_flag: create_csv_output(x, I_vol, title=f'I_vol_{inputs["file_options"]["id"]}')
        if plot1D_flag and plots['IV_1D']: plot_with_continuous_error(x, I_vol, x_title='Distance from anode (µm)', y_title='Charge transfer rate (A/m3)', title=f'I_vol_{inputs["file_options"]["id"]}', save_img=img_flag)

    # 1D area current density plot [OLD] 
    # not needed anymore since we have the volumetric charge transfer rate
    # also, this is not a good way to calculate the area-specific current density
    # area-specific current density is actually the flux of ion potential through YZ plane
    if False:
        area = N[0][1]*N[0][2]*dx**2 # [m2]
        I_avg = np.zeros(Lx_tot)
        # j_buch = 0         # area specific current density as defined in eq 21 by Buchaniec et al. 2019 [A/m2]
        j_a_prok = 0         # area specific current density as defined in eq 3.31 in Prokop's thesis 2020 [A/m2]
        for i in range(Lx_a):
            if i == 0 or i == Lx_a-1:
                I_avg[i] = np.nan
            else:
                a = dense_m['I'][i, ...]     # [A/m3]
                j_a_prok += np.nansum(a)*dx**3        # [A]

        j_c_prok = 0
        for i in range(Lx_c-1, -1, -1):
            if i == 0 or i == Lx_c-1:
                I_avg[i+Lx_a+Lx_e] = np.nan
            else:
                a = dense_m['I'][i+Lx_a+Lx_e, ...]
                j_c_prok += np.nansum(a)*dx**3        # [A]
                I_avg[i+Lx_a+Lx_e] = j_c_prok/area      # [A/m2]
                
        if csv_flag: create_csv_output(x, I_avg, title=f'I_{inputs["file_options"]["id"]}')
        if plot1D_flag: plot_with_continuous_error(x, I_avg, x_title='Distance from anode (µm)', y_title='Area-specific current density (A/m2)', title=f'Ia_A_{inputs["file_options"]["id"]}', save_img=img_flag)
    
    # 1D ion flux plot - sigma*dVio/dx*area [A] [NEW]
    # This value should be devided by the area of electrolyte to get the area-specific current density [A/m2]
    if False:
        # fluxion_matrix is [A/m2]
        fluxion_matrix = np.zeros(N[1])*np.nan
        fluxion_matrix[:Lx_a,...]           = -K[2] * dense_m['grad_phi'][2][0][:Lx_a, ...]             #* vol_frac_YSZ[:Lx_a]
        fluxion_matrix[Lx_a:Lx_a+Lx_e,...]  = -K[6] * dense_m['grad_phi'][2][0][Lx_a:Lx_a+Lx_e, ...]    #* vol_frac_YSZ[Lx_a:Lx_a+Lx_e]
        fluxion_matrix[Lx_a+Lx_e:,...]      = -K[5] * dense_m['grad_phi'][2][0][Lx_a+Lx_e:, ...]        #* vol_frac_YSZ[Lx_a+Lx_e:]
        # fluxion_matrix[~ds[2]] = np.nan
        # fluxion_matrix[:Lx_a,...][TPB_dict['anode']['TPB_mask']] = np.nan
        # fluxion_matrix[Lx_a+Lx_e:,...][TPB_dict['cathode']['TPB_mask']] = np.nan

        dx = inputs['microstructure']['dx']
        Ia = np.nansum(fluxion_matrix, axis=(1,2)) / (N[0][1]*N[0][2])      # 1D [A/m2]
            
        if plots['Ia_1D'] and plot1D_flag: plot_with_continuous_error(x, Ia, x_title='Distance from anode (µm)', y_title='Current density (A/m2)', title=f'Ia_{inputs["file_options"]["id"]}', save_img=img_flag)
        if csv_flag: create_csv_output(x, Ia, title=f'Ia_{inputs["file_options"]["id"]}')

    # 1D ion flux calculated from matrix of coefficients [A]
    if True:
        # flux matrix is [A/m2]
        flux_W, flux_E = dense_m['flux_W'], dense_m['flux_E']

        # Ia_W & Ia_E are [A/cm2] d
        Ia_W = np.nansum(flux_W, axis=(1,2)) / (N[0][1]*N[0][2]) / 1e4
        # Ia_E = np.nansum(flux_E, axis=(1,2)) / (N[0][1]*N[0][2]) / 1e4

        if plots['flux_1D'] and plot1D_flag: 
            # plot_with_continuous_error(x, Ia_E, x_title='Distance from anode (µm)', y_title='Current density, East (A/cm2)', title=f'flux_E_{inputs["file_options"]["id"]}', save_img=img_flag)
            plot_with_continuous_error(x, Ia_W, x_title='Distance from anode (µm)', y_title='Current density, West (A/cm2)', title=f'flux_W_{inputs["file_options"]["id"]}', save_img=img_flag)
        if csv_flag: 
            # create_csv_output(x, Ia_E, title=f'Ia_E_{inputs["file_options"]["id"]}')
            create_csv_output(x, Ia_W, title=f'Ia_W_{inputs["file_options"]["id"]}')


    if inputs['output_options']['show_3D_plots']:
        vars = ['rhoH2', 'Vel', 'Vio','rhoO2', 'I', 'eta_act', 'eta_con']
        for var in vars:
            if plots[f'{var}_3D']:
                mats.append(dense_m[var])
                thds.append(())
                titles.append(var)
                log_scale.append(True if var=='eta_act' else False)

        # TPB_mats = [[TPB_mesh]*len(mats)]
        TPB_mats = [[TPB_mesh_a, TPB_mesh_c]] * len(mats)
        if mats != []:
            visualize_mesh(
                mat = mats,
                thd = thds,
                titles = titles,
                clip_widget = False, 
                TPB_mesh = TPB_mats,
                log_scale = log_scale,
                link_views=True,
                show_axis=True,
                entire_domain=[True]*len(mats),
                )

def plot_with_continuous_error(x, y, y_min=None, y_max=None, y_c_down=None, y_c_up=None, 
                               x_title='x', y_title='y', title=None, save_img=False, log_type="linear"):
    import plotly.graph_objects as go

    x = [x] if type(x) is not list else x 
    y = [y] if type(y) is not list else y
    y_min = [y_min] if type(y_min) is not list else y_min
    y_max = [y_max] if type(y_max) is not list else y_max
    y_c_down = [y_c_down] if type(y_c_down) is not list else y_c_down
    y_c_up = [y_c_up] if type(y_c_up) is not list else y_c_up

    fig = go.Figure()
    for i in range(len(y)):
        if y_max[i] is not None:
            fig.add_trace(
                go.Scatter(
                    name='Upper Bound',
                    x=x[i],
                    y=y_max[i],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                ))
            fig.add_trace(
                go.Scatter(
                    name='Lower Bound',
                    x=x[i],
                    y=y_min[i],
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=False,
                ))

        # add y curve
        if y[i] is not None:
            fig.add_trace(go.Scatter(
                name='y',
                x=x[i],
                y=y[i],
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                showlegend=False,
            ))

        # add confidence level
        if y_c_down[i] is not None:
            fig.add_trace(go.Scatter(
                name='Continuous Lower Bound',
                x=x[i],
                y=y_c_down[i],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                # fillcolor='rgba(68, 68, 68, 0.7)',
                # fill='tonexty',
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                name='Continuous Upper Bound',
                x=x[i],
                y=y_c_up[i],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.7)',
                fill='tonexty',
                showlegend=False,
            ))


    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title=x_title,
        title=title,
        hovermode=None,
    )

    fig.update_xaxes(exponentformat="SI") 
    fig.update_yaxes(exponentformat="e")
    fig.update_yaxes(type=log_type)
    fig.show()
    if save_img:
        dir = 'Binary files/1D plots/graphs/'
        create_directory(dir)
        file_dir = dir + f'{title if title is not None else "fig"}.html'
        fig.write_html(file_dir)

def mean_confidence_interval(data, confidence=0.95):
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def save_image(phase_mat):
    
    from tqdm import tqdm
    import imageio
    
    phase_1 = np.zeros(shape=phase_mat.shape)
    phase_1[phase_mat==1] = 1
    
    phase_2 = np.zeros(shape=phase_mat.shape)
    phase_2[phase_mat==2] = 2
    
    phase_3 = np.zeros(shape=phase_mat.shape)
    phase_3[phase_mat==3] = 3
    
    N = phase_mat.shape[0]
    for k in tqdm(np.arange(N)):
        str_1 = "images\phase_1."+f"{k:03}"
        imageio.imwrite(str_1, phase_1[:,:,k], format='png')     

def visualize_mesh(
        mat, 
        thd=None, 
        blocks=[], 
        titles=[], 
        clip_widget=False, 
        TPB_mesh=[], 
        entire_domain=None,
        log_scale=None, 
        animation=None,
        save_graphics=False,
        elevation=0,
        azimuth=0,
        isometric=False,
        link_views=False,
        cmap='viridis',
        save_vtk=False,
        show_axis=False,
        link_colorbar=False,
        edge_width=0,
        opacity=1
        ):
    """
    Visualizes the mesh via PyVista.
    inputs:
    mat : float
        Three dimensional matrix describing the phase data. It could be list 
    """
    import pyvista as pv
    import matplotlib.pyplot as plt
    
    pv.set_plot_theme("document")
    # check if colormap is qualitative or not
    is_qualitative = False
    if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                'tab20c']:
        is_qualitative = True
    
    cmap = plt.cm.get_cmap(cmap)     
     
    if thd is None: thd = [()]*len(mat)
    if entire_domain is None: entire_domain = [False]*len(mat)

    if link_colorbar:
        zmin = min([np.nanmin(m) for m in mat])
        zmax = max([np.nanmax(m) for m in mat])
        clim = [zmin, zmax]
        if is_qualitative:
            from matplotlib.colors import LinearSegmentedColormap
            colors = cmap(range(zmax-zmin+1))  # Get the first N unique colors
            cmap = LinearSegmentedColormap.from_list("", colors)
    else:
        clim = None
    
    subplts = len(mat)
    
    if bool(blocks): 
        p = pv.Plotter(
            shape=(1, subplts+1), 
            border=True, 
            notebook=False,
            )
    else:
        p = pv.Plotter(
            shape=(1, subplts), 
            border=True, 
            notebook=False, 
            off_screen=True if save_graphics else False,
            )
        if show_axis: p.show_axes()
    
    for i in np.arange(subplts):
        scale = log_scale[i] if log_scale is not None else False
        sargs = dict(
            title=f'P{i+1}', 
            height=0.15, 
            vertical=True, 
            position_x=0.01, 
            position_y=0.8,
            )
        sub_mat = mat[i]
        N=sub_mat.shape
        # Initializing grids
        mesh = pv.ImageData(dimensions=(N[0]+1,N[1]+1,N[2]+1))
        
        # Assigning values to grids and thresholding
        if type(sub_mat[0,0,0]) is np.bool_:
            sub_mat = sub_mat.astype(int) 
            mesh.cell_data["data"] = sub_mat.T.flatten()
            mesh = mesh.threshold((1,1))
        else:
            mesh.cell_data["data"] = sub_mat.T.flatten()
            sub_thd = thd[i]
            if save_vtk: mesh.save('mesh.vtk')
            if len(sub_thd)==0:
                mesh = mesh.threshold()
            if len(sub_thd)>0:
                mesh = mesh.threshold(sub_thd)
        
        p.subplot(0, i)
        # mesh.save(f"mesh{i}.vtk")
        if clip_widget:
            p.add_mesh_clip_plane(
                mesh,
                scalar_bar_args=sargs, 
                cmap=cmap, 
                # cmap=cmap_mat[i] if is_qualitative else cmap, 
                clim=clim, 
                show_edges=False if edge_width==0 else True,
                line_width=edge_width,
                normal_rotation = 0.0,
                normal = (0,1,0),
                origin = (0,1,0),
                opacity=opacity,
                )
        else:
            p.add_mesh(
                mesh, 
                scalar_bar_args=sargs, 
                log_scale=scale, 
                cmap=cmap, 
                # cmap=cmap_mat[i] if is_qualitative else cmap, 
                clim=clim, 
                show_edges=False if edge_width==0 else True,
                line_width=edge_width,
                opacity=opacity,
                )
            p.add_bounding_box(line_width=1, color='black')
            if bool(titles):
                p.add_text(titles[i], font_size=20, position='lower_edge')
        if bool(TPB_mesh):
            if save_vtk: TPB_mesh[i].save(f'TPB_mesh{i}.vtk')
            if entire_domain[i]:
                p.add_mesh(TPB_mesh[i][0], line_width=10, color='r')
                p.add_mesh(TPB_mesh[i][1], line_width=10, color='r')
            else:
                p.add_mesh(TPB_mesh[i], line_width=10, color='r')

        
        if isometric: p.view_isometric()
        else: 
            p.camera_position = 'xy'
            # p.camera.clipping_range = (1e-2, 1e3)
            p.camera.elevation = elevation
            p.camera.azimuth = azimuth
        # p.show_grid()
        # p.remove_scalar_bar()
        # p.enable_parallel_projection()


    if bool(blocks):
        p.subplot(0, subplts)
        p.add_mesh(blocks)
        
    if link_views: p.link_views()

    if animation is not None:
        p.open_movie("Binary files/animation.mp4")
        p.camera_position = 'xz'
        if animation == 'zoom':
            frames = 200
            final_zoom = 0.6
            initial_zoom = 0.1
            p.camera.zoom(initial_zoom)
            p.camera.clipping_range = (1e-2, 1e3)
            for value in np.linspace(0, 1, int(frames)):
                x = (final_zoom / initial_zoom)**(1/(frames))
                p.camera.zoom(x)
                p.camera.azimuth = value * 40
                p.camera.elevation = value * 20
                p.write_frame()
            frames = 400
            for value in np.linspace(0, 1, int(frames)):
                p.camera.azimuth = value * 360 + 40
                p.write_frame()
            p.close()
        elif animation == 'rotate':
            frames = 200
            p.camera.elevation = 20
            p.camera.zoom(1)
            for value in np.linspace(0, 1, int(frames)):
                p.camera.azimuth = value * 360
                p.write_frame()
            p.close()
    elif animation is None:
        if save_graphics:
            scale_factor = 1
            p.window_size = [scale_factor*p.window_size[0],scale_factor*p.window_size[1]]
            p.show()
            p.update()
            p.screenshot('screenshot.png')
            # file_name = 'img.' + save_graphics
            # if save_graphics=='pdf' or save_graphics=='svg':
            #     p.save_graphic(file_name, raster=False, painter=False)
            # elif save_graphics=='html':
            #     p.export_html("img.html")
        else:
            p.show()

    return None
    
def visualize_network(volumes, centroids, M=1):
    import pyvista as pv
    
    blocks = pv.MultiBlock()
    
    for i in np.arange(len(volumes[M-1])):
        radius = np.power(volumes[M-1][i,:]*3/4/np.pi,1/3)
        center = np.flip(centroids[M-1][i,:])
        blocks.append(pv.Sphere(radius = radius, center = center))
                      
    return blocks

def visualize_contour(mat, n_levels=5):
    import pyvista as pv
    import matplotlib.pyplot as plt

    pv.set_plot_theme("document")
    cmap_contur = plt.cm.get_cmap("viridis")

    N=mat.shape

    # Initializing grids
    mesh = pv.ImageData(dimensions=(N[0]+1,N[1]+1,N[2]+1))
    mesh.cell_data["data"] = mat.flatten()
    mesh = mesh.threshold()
    mesh = mesh.cell_data_to_point_data()

    p = pv.Plotter(notebook=False)
    contours = mesh.contour(np.linspace(np.nanmin(mat), np.nanmax(mat), n_levels))    

    # show the solid phase [needs improvement]
    solid = np.zeros_like(mat)
    solid[np.isnan(mat)] = np.nan
    solid[~np.isnan(mat)] = 1
    mesh_s = pv.ImageData(dimensions=(N[0]+1,N[1]+1,N[2]+1))
    mesh_s.cell_data["data"] = solid.flatten()
    mesh_s = mesh_s.threshold()
    cmap_solid = plt.cm.get_cmap("Greys")
    
    # p.add_mesh(mesh_s, cmap=cmap_solid, opacity=0.1)
    p.add_mesh(contours, cmap=cmap_contur)
    p.add_bounding_box(line_width=1, color='black')
    p.show_axes()
    p.show()
    
    return None
    
def is_edge(shape, i, j, k):
    """
    Checks if a given set of i,j,k indices lie on the edge of a 3D domain.

    Args:
        shape: The shape of the 3D domain as a tuple (i_dim, j_dim, k_dim).
        i: The i-th index.
        j: The j-th index.
        k: The k-th index.

    Returns:
        True if the indices lie on the edge, False otherwise.
    """
    i_dim, j_dim, k_dim = shape
    edges = {(0, 0), (0, j_dim - 1), (i_dim - 1, 0), (i_dim - 1, j_dim - 1)} | \
            {(0, k_dim - 1), (i_dim - 1, k_dim - 1), (j_dim - 1, 0), (j_dim - 1, k_dim - 1)}
    return (i, j) in edges or (i, k) in edges or (j, k) in edges

# specific functions for entire cell
# def visualize_3D_matrix_entire_cell(inputs, phi_dense, masks_dict, TPB_dict, titles, rhoH2=None, Vel_a=None, Vio=None, rhoO2=None, Vel_c=None, field_mat=None, TPB_mat=None):
#     # visualize the solution
#     import pyvista as pv
#     import pandas as pd
#     import plotly.express as px
#     pv.set_plot_theme("document")

#     N = [inputs['Nx_a']+inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
#     ds = masks_dict['ds']
#     ds_entire_cell = [None]*5
#     ds_entire_cell[0] = np.concatenate((ds[0], np.zeros((inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']),dtype=bool)), axis=0)
#     ds_entire_cell[1] = np.concatenate((ds[1], np.zeros((inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']),dtype=bool)), axis=0)
#     ds_entire_cell[2] = ds[2]
#     ds_entire_cell[3] = np.concatenate((np.zeros((inputs['Nx_a']+inputs['Nx_e'], inputs['Ny'], inputs['Nz']),dtype=bool),ds[3]),axis=0)
#     ds_entire_cell[4] = np.concatenate((np.zeros((inputs['Nx_a']+inputs['Nx_e'], inputs['Ny'], inputs['Nz']),dtype=bool),ds[4]),axis=0)
#     mats = []
#     thds = []
#     log_scale = []
    
#     if Vio is not None:
#         sol_Vio = np.copy(phi_dense)
#         sol_Vio[ds_entire_cell[2] == False] = np.nan
#         mats.append(sol_Vio)
#         thds.append(())
#         log_scale.append(False)

#     import pyvista as pv
#     vertices_a = TPB_dict['anode']['vertices']
#     lines_a = TPB_dict['anode']['lines']

#     vertices_c = TPB_dict['cathode']['vertices']
#     lines_c = TPB_dict['cathode']['lines']

#     TPB_mesh_a = pv.PolyData(vertices_a, lines=lines_a)
#     TPB_mesh_c = pv.PolyData(vertices_c, lines=lines_c)
    
    
#     if inputs['show_3D_plots']:
#         visualize_mesh(
#             mat = mats,
#             thd = thds,
#             titles = titles,
#             clip_widget = False, 
#             TPB_mesh = [TPB_mesh_a, TPB_mesh_c],
#             log_scale = log_scale)

#     if inputs['show_1D_plots']:
#         import plotly.graph_objects as go
#         from plotly.subplots import make_subplots

#         Nx = inputs['Nx_a'] + inputs['Nx_e'] + inputs['Nx_c']
#         x = np.arange(Nx)*inputs['dx']*1e6

#         Vio_lin = np.zeros(Nx)
#         Vio_min = np.zeros_like(Vio_lin)
#         Vio_max = np.zeros_like(Vio_lin)
#         Vio_c_up = np.zeros_like(Vio_lin)
#         Vio_c_down = np.zeros_like(Vio_lin)

#         for i in range(Nx):
#             a = sol_Vio[i, :, :][~np.isnan(sol_Vio[i, :, :])]
#             Vio_lin[i] = np.average(a)
#             Vio_max[i] = np.max(a)
#             Vio_min[i] = np.min(a)
#             Vio_c_down[i], Vio_c_up[i] = mean_confidence_interval(a)

#         plot_with_continuous_error(x, Vio_lin, Vio_max, Vio_min, Vio_c_down, Vio_c_up, x_title='Distance from anode (µm)', y_title='Hydrogen concentration (kg/m3)', title='Hydrogen concentration (kgm-3)')

def create_dense_matrices(
        phi, 
        masks_dict, 
        indices, 
        field_functions, 
        TPB_dict,
        K_array_ion
        ):
    
    def determine_fluxes():
        
        # from tqdm import tqdm
        from modules.preprocess import harmonic_mean as hm
        dx = inputs['microstructure']['dx']
        ds = masks_dict['ds']
        flux_W = [None]*len(phi)
        flux_E = [None]*len(phi)

        # determine the fluxes
        flux_W = np.zeros(shape=ds[2].shape)
        flux_E = np.zeros(shape=ds[2].shape)
        K_array_ion[~ds[2]] = np.nan

        for n in range(len(indices[2]['all_points'])):
            i,j,k = indices[2]['all_points'][n]
            if i==0 or i==N[2][0]-1: continue
            # aW and aE are [S] or [A/V]
            # aW = J_scl[p][n,indices[p]['west_nb'][n]] if indices[p]['west_nb'][n]!=-1 else 0
            # aE = J_scl[p][n,indices[p]['east_nb'][n]] if indices[p]['east_nb'][n]!=-1 else 0

            # kw and ke are [S/m]
            Kw = hm(K_array_ion[i,j,k], K_array_ion[i-1,j,k])
            Ke = hm(K_array_ion[i,j,k], K_array_ion[i+1,j,k])

            # flux_W and flux_E are [A/m2]
            phi_nb = phi[2][indices[2]['west_nb'][n]] if indices[2]['west_nb'][n]!=-1 else np.nan
            flux_W[i,j,k] = Kw * (phi_nb - phi[2][n]) / dx
            
            phi_nb = phi[2][indices[2]['east_nb'][n]] if indices[2]['east_nb'][n]!=-1 else np.nan
            flux_E[i,j,k] = Ke * (phi_nb - phi[2][n]) / dx

        flux_W[~ds[2]] = np.nan
        flux_E[~ds[2]] = np.nan
            
        return flux_W, flux_E
    
    def create_field_variable(func, phase):
        field_mat = np.zeros(shape = N[1])

        for n in indices[phase]['interior']:
            i,j,k = indices[phase]['all_points'][n]
            if phase==3: i += Lx_a + Lx_e

            field_mat[i,j,k] = func(phi_dense[i,j,k])

        return field_mat

    def create_TPB_field_variable(func_name, location):
        if func_name == 'current_density':
            func = field_functions['f'][2] if location=='anode' else field_functions['f'][5]
        elif func_name == 'eta_act':
            func = field_functions['eta_a_act'] if location=='anode' else field_functions['eta_c_act']
        elif func_name == 'eta_con':
            func = field_functions['eta_a_con'] if location=='anode' else field_functions['eta_c_con']

        TPB_mat = np.zeros(N[1])

        for phase in range(len(phi)):
            if phase==0 and location=='cathode': continue
            elif phase==3 and location=='anode': continue

            for n in indices[phase]['source']:
                i,j,k = indices[phase]['all_points'][n]
                if is_edge(N[phase], i,j,k): continue
                
                flag_a, flag_e, flag_c = False, False, False
                if phase==0: flag_a = True
                elif phase==3:
                    i += Lx_a + Lx_e
                    flag_c = True
                elif phase==1 or phase==2:
                    if i < Lx_a: flag_a = True
                    elif i >= Lx_a and i < Lx_a + Lx_e: flag_e = True       # it should never happen [no source points in the electrolyte]
                    elif i >= Lx_a + Lx_e: flag_c = True

                if flag_a and location=='cathode': continue
                elif flag_c and location=='anode': continue
                elif flag_a and location=='anode':
                    cH2_i = phi_dense[i,j,k] if phase==0 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds_entire[0][i-1:i+2,j-1:j+2,k-1:k+2]])
                    cO2_i = None
                elif flag_c and location=='cathode':
                    cH2_i = None
                    cO2_i = phi_dense[i,j,k] if phase==3 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds_entire[3][i-1:i+2,j-1:j+2,k-1:k+2]])
                
                Vel_i = phi_dense[i,j,k] if phase==1 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds_entire[1][i-1:i+2,j-1:j+2,k-1:k+2]])
                Vio_i = phi_dense[i,j,k] if phase==2 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds_entire[2][i-1:i+2,j-1:j+2,k-1:k+2]])

                TPB_mat[i,j,k] = func(
                    cH2_i if flag_a else cO2_i,
                    Vel_i, 
                    Vio_i,
                    )
                
        return TPB_mat
    
    write_arrays = inputs['output_options']['write_arrays']
    ds = masks_dict['ds']
    N = [ds[i].shape for i in range(len(phi))]
    
    flux_W, flux_E = determine_fluxes()
    
    if len(phi) == 4:    # anode, electrolyte, cathode
        Lx_a = N[0][0]
        Lx_e = N[1][0] - N[0][0] - N[3][0]
        Lx_c = N[3][0]
        ds_entire = [None]*4
        # gas phase (anode) - H2
        ds_entire[0] = np.concatenate((ds[0], np.zeros((Lx_e + Lx_c, N[0][1], N[0][2]),dtype=bool)), axis=0)
        # electron phase (anode & cathode)
        ds_entire[1] = ds[1]
        # ion phase (anode & cathode)
        ds_entire[2] = ds[2]
        # gas phase (cathode) - O2
        ds_entire[3] = np.concatenate((np.zeros((Lx_a+Lx_e, N[0][1], N[0][2]),dtype=bool),ds[3]),axis=0)
    elif len(phi) == 3:      # anode 
        ds_entire = ds
        
    phi_dense = np.zeros(N[1])
    for i in range(len(phi)):
        phi_dense[ds_entire[i]] = phi[i]

    Ia_mat = create_TPB_field_variable('current_density', 'anode')
    if len(phi) == 4: Ic_mat = create_TPB_field_variable('current_density', 'cathode')
    I_mat = Ia_mat + Ic_mat if len(phi) == 4 else Ia_mat
    I_mat[I_mat==0] = np.nan

    eta_a_act_mat = create_TPB_field_variable('eta_act', 'anode')
    if len(phi) == 4: eta_c_act_mat = create_TPB_field_variable('eta_act', 'cathode')
    eta_act_mat = eta_a_act_mat + eta_c_act_mat if len(phi) == 4 else eta_a_act_mat
    eta_act_mat[eta_act_mat==0] = np.nan
    
    eta_a_con_mat = create_field_variable(field_functions['eta_a_con'],0)
    if len(phi) == 4: eta_c_con_mat = create_field_variable(field_functions['eta_c_con'],3)
    eta_con_mat = eta_a_con_mat + eta_c_con_mat if len(phi) == 4 else eta_a_con_mat
    ds_gas = np.zeros_like(ds[1],dtype=bool)
    ds_gas[:N[0][0],...] = ds[0]
    ds_gas[N[1][0]-N[0][0]:,...] = ds[3]
    eta_con_mat[~ds_gas] = np.nan
    
    sol_phi = [None] * len(phi)
    grad_phi = [None] * len(phi)

    # from findiff import FinDiff
    # d_d = [None]*3
    # d_d[0] = FinDiff(0, dx, acc=4)
    # d_d[1] = FinDiff(1, dx, acc=4)
    # d_d[2] = FinDiff(2, dx, acc=4)

    for i in range(len(phi)):
        sol_phi[i] = np.copy(phi_dense)
        sol_phi[i][ds_entire[i] == False] = np.nan

        # numpy approach
        grad_phi[i] = np.gradient(sol_phi[i], inputs['microstructure']['dx'])
        
    #     # FinDiff approach
    #     # grad_phi[i] = [None]*3
    #     # for dir in range(3):
    #     #     grad_phi[i][dir] = d_d[dir](sol_phi[i]) 

    dense_m = {
        'phi_dense': phi_dense,
        'rhoH2': sol_phi[0], 
        'Vel': sol_phi[1], 
        'Vio': sol_phi[2], 
        'rhoO2': sol_phi[3] if len(sol_phi)==4 else None,
        'grad_phi': grad_phi,
        'I': I_mat, 
        'eta_act': eta_act_mat, 
        'eta_con': eta_con_mat,
        'flux_W': flux_W,
        'flux_E': flux_E,
        }
    
    if write_arrays:
        type = 'entire' if len(phi)==4 else 'anode'
        np.savez(
            f'Binary files/arrays/matrices_{type}_{inputs["file_options"]["id"]}.npz', 
            dx = inputs['microstructure']['dx'],
            phi = dense_m['phi_dense'], 
            rhoH2 = dense_m['rhoH2'], 
            Vel = dense_m['Vel'], 
            Vio = dense_m['Vio'], 
            rhoO2 = dense_m['rhoO2'],
            # grad_phi = dense_m['grad_phi'],
            I   = dense_m['I'], 
            eta_act = dense_m['eta_act'],
            eta_con = dense_m['eta_con'],
            TPB_dict = TPB_dict,
            )

    return dense_m

def create_csv_output(x, y_avg, y_min=None, y_max=None, y_c_down=None, y_c_up=None, title='y'):
    import pandas as pd

    avg_title = title + '_avg'
    min_title = title + '_min'
    max_title = title + '_max'
    c_down_title = title + '_c_down'
    c_up_title = title + '_c_up'

    df = pd.DataFrame({'x': x, 
                       avg_title: y_avg, 
                       min_title: y_min, 
                       max_title: y_max, 
                       c_down_title: y_c_down, 
                       c_up_title: y_c_up})
    
    dir = 'Binary files/1D plots/csv/'
    create_directory(dir)
    df.to_csv(dir + title + '.csv', index=False)

def create_directory(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

def plot_domain(
        domains, 
        gap=0, 
        qualitative=True, 
        file_name=None, 
        renderer='browser', 
        template='plotly',
        link_views=True,
        colormap='plasma',
        show_figure=True,
        not_pores=False):
    
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.renderers.default = renderer

    # number of domains
    N = len(domains)

    for n in range(N):
        domains[n] = np.rot90(domains[n]).astype(float)
        if not_pores:
            domains[n][domains[n]==1] = np.nan
    # range of colorscale
    zmin = np.nanmin([np.nanmin(d) for d in domains]).astype(int)
    zmax = np.nanmax([np.nanmax(d) for d in domains]).astype(int)

    # choose a qualitiative colormap from zmin to zmax
    if qualitative:
        pass
        # if zmin == zmax:
        #     colorscale = px.colors.qualitative.Bold
        # else:
        #     colorscale = [px.colors.qualitative.Bold[i] for i in np.arange(zmax-zmin+1)]
    else:
        pass
        # colorscale = 'Viridis'
    colorscale = colormap
        
    # create figure
    fig = make_subplots(
        rows=1, 
        cols=N, 
        shared_yaxes=link_views)
    
    for n in range(N):
        fig.append_trace(
            go.Heatmap(
                z=domains[n], 
                xgap=gap, 
                ygap=gap, 
                zmin=zmin, 
                zmax=zmax, 
                colorscale=colorscale, 
                showscale=False
                ), 
            row=1, 
            col=n+1)
        
        # make every pixel in every figure square
        scaleanchor = "x" if N==1 else f"x{n+1}"
        fig.update_yaxes(
            scaleanchor=scaleanchor,
            scaleratio=1,
            row=1, 
            col=n+1,)

        # update matches for each axis (x2, x3, ... match x1)
        if link_views:
            fig.update_xaxes(
                scaleanchor='x' if N==1 else 'x1',
                scaleratio=1,
                matches='x1',
                row=1, 
                col=n+1)
        
    fig.update_layout(template=template)

    if show_figure: fig.show()
    if file_name is not None: fig.write_image(file_name+'.svg')

def create_fig_from_1D_model(color):
    # read csv file
    import csv
    import numpy as np
    # import plotly.express as px
    import plotly.graph_objects as go

    # read csv file
    with open('Vio','r') as file:
        reader = csv.reader(file)
        # ignore the first 8 lines
        for _ in range(8):
            next(reader)
        data = list(reader)
        list_nums = [[float(k) for k in data[m][0].split()] for m in range(len(data))]
        x_Vio = np.array([list_nums[k][0] for k in range(len(list_nums))])
        Vio = np.array([list_nums[k][1] for k in range(len(list_nums))])
    
    x = [None] * 3
    flux = [None] * 3
    for i,location in enumerate(['anode', 'electrolyte', 'cathode']):
        if location == 'anode':
            filename = 'flux_a'
        elif location == 'electrolyte':
            filename = 'flux_e'
        elif location == 'cathode':
            filename = 'flux_c'
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            # ignore the first 8 lines
            for _ in range(8):
                next(reader)
            data = list(reader)
            list_nums = [[float(k) for k in data[m][0].split()] for m in range(len(data))]
            x[i] = np.array([list_nums[k][0] for k in range(len(list_nums))])
            flux[i] = np.array([list_nums[k][1] for k in range(len(list_nums))])

    flux_tot = np.concatenate([flux[i] for i in range(3)])
    x_tot = np.concatenate([x[i] for i in range(3)])
    
    fig = [None]*2
    fig[0] = go.Scatter(
        x=x_Vio,
        y=Vio,
        name='Vio_1D_model',
        mode='lines',
        yaxis='y2',
        line=dict(
            color=color, 
            width=2, 
            dash='dash',
            ),
        )
    
    fig[1] = go.Scatter(
        x=x_tot,
        y=flux_tot,
        name='Ia_1D_model',
        mode='lines',
        yaxis='y1',
        line=dict(
            color=color, 
            width=2, 
            dash='solid',
            ),
        )

    return fig