def visualize_residuals(inputs, residuals):
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # visualize the error
    x = np.arange(len(residuals[2]))
    r = [None]*3
    df = [None]*3
    titles = ['Hydrogen concentration', 'Electron potential', 'Ion potential']

    for p in [0,1,2]:
        if inputs['solver_options']['ion_only'] and p!=2:
            continue
        r[p] = np.stack((x, residuals[p]), axis=-1)
        df[p] = pd.DataFrame(r[p], columns=['iteration', 'residual'])
        df[p].insert(2, 'Variable', titles[p])

    df = pd.concat(df)
    fig = px.line(df, x='iteration', y='residual', color='Variable', log_y=True)
    fig.update_yaxes(exponentformat="e")
    fig.show()

def visualize_3D_matrix(inputs, dense_m, masks_dict, TPB_dict, plots, vol_fac=1, save_file=False):
    # visualize the solution
    import numpy as np

    csv_flag = inputs['output_options']['csv_output']
    plot1D_flag = inputs['output_options']['show_1D_plots']
    plot3D_flag = inputs['output_options']['show_3D_plots']

    if plot3D_flag:
        import pyvista as pv
        pv.set_plot_theme("document")
        TPB_mesh = pv.PolyData(vertices, lines=lines)

    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]
    
    dx = inputs['microstructure']['dx']
    ds = masks_dict['ds']
    vertices = TPB_dict['vertices']
    lines = TPB_dict['lines']
    mats = []
    thds = []
    log_scale = []
    titles = []

    dense_phi = dense_m['phi_dense']
    dense_cH2 = dense_m['cH2']
    dense_Vel = dense_m['Vel']
    dense_Vio = dense_m['Vio']
    dense_Ia = dense_m['Ia']
    dense_eta_act = dense_m['eta_act']
    dense_eta_con = dense_m['eta_con']

    x = np.arange(N[0])*dx*1e6

    if plots['cH2_1D'] and (plot1D_flag or csv_flag):
        cH2_avg = np.zeros(N[0])
        cH2_max = np.zeros(N[0])
        cH2_min = np.zeros(N[0])
        cH2_c_down = np.zeros(N[0])
        cH2_c_up = np.zeros(N[0])
        for i in range(N[0]):
            a = dense_cH2[i, :, :][~np.isnan(dense_cH2[i, :, :])]
            cH2_avg[i] = np.average(a)
            cH2_max[i] = np.max(a)
            cH2_min[i] = np.min(a)
            cH2_c_down[i], cH2_c_up[i] = mean_confidence_interval(a)
        if csv_flag: create_csv_output(x, cH2_avg, cH2_min, cH2_max, cH2_c_down, cH2_c_up, 'cH2')
        if plot1D_flag: plot_with_continuous_error(x, cH2_avg, cH2_c_down, cH2_c_up, x_title='Distance from anode (µm)', y_title='Hydrogen concentration (kg/m3)', title='Hydrogen concentration', save_file=save_file)
    
    if plots['Vel_1D'] and (plot1D_flag or csv_flag):
        Vel_avg = np.zeros(N[0])
        Vel_max = np.zeros(N[0])
        Vel_min = np.zeros(N[0])
        Vel_c_down = np.zeros(N[0])
        Vel_c_up = np.zeros(N[0])
        for i in range(N[0]):
            a = dense_Vel[i, :, :][~np.isnan(dense_Vel[i, :, :])]
            Vel_avg[i] = np.average(a)
            Vel_max[i] = np.max(a)
            Vel_min[i] = np.min(a)
            Vel_c_down[i], Vel_c_up[i] = mean_confidence_interval(a)
        if csv_flag: create_csv_output(x, Vel_avg, Vel_min, Vel_max, Vel_c_down, Vel_c_up, 'Vel')
        if plot1D_flag: plot_with_continuous_error(x, Vel_avg, Vel_c_down, Vel_c_up, x_title='Distance from anode (µm)', y_title='Electron potential (V)', title='Electron potential', save_file=save_file)

    if plots['Vio_1D'] and (plot1D_flag or csv_flag):
        Vio_avg = np.zeros(N[0])
        Vio_max = np.zeros(N[0])
        Vio_min = np.zeros(N[0])
        Vio_c_down = np.zeros(N[0])
        Vio_c_up = np.zeros(N[0]) 
        for i in range(N[0]):
            a = dense_Vio[i, :, :][~np.isnan(dense_Vio[i, :, :])]
            Vio_avg[i] = np.average(a)
            Vio_max[i] = np.max(a)
            Vio_min[i] = np.min(a)
            Vio_c_down[i], Vio_c_up[i] = mean_confidence_interval(a)
        if csv_flag: create_csv_output(x, Vio_avg, Vio_min, Vio_max, Vio_c_down, Vio_c_up, 'Vio')
        if plot1D_flag: plot_with_continuous_error(x, Vio_avg, Vio_min, Vio_max, Vio_c_down, Vio_c_up, x_title='Distance from anode (µm)', y_title='Ion potential (V)', title='Ion potential', save_file=save_file)

    if plots['Ia_1D'] and (plot1D_flag or csv_flag):
        Ia_avg = np.zeros(N[0])
        Ia_max = np.zeros(N[0])
        Ia_min = np.zeros(N[0])
        Ia_c_down = np.zeros(N[0])
        Ia_c_up = np.zeros(N[0])
        Ia_A_avg = np.zeros(N[0])
        Ia_A_max = np.zeros(N[0])
        Ia_A_min = np.zeros(N[0])
        Ia_A_c_down = np.zeros(N[0])
        Ia_A_c_up = np.zeros(N[0])

        vol = np.zeros(N[0])
        area = N[1]*N[2]*dx**2 # [m2]
        for i in range(N[0]):
            if i == 0 or i == N[0]-1:
                Ia_avg[i] = np.nan
                Ia_max[i] = np.nan
                Ia_min[i] = np.nan
                Ia_c_down[i], Ia_c_up[i] = np.nan, np.nan
                Ia_A_avg[i] = np.nan
                Ia_A_max[i] = np.nan
                Ia_A_min[i] = np.nan
                Ia_A_c_down[i], Ia_A_c_up[i] = np.nan, np.nan
                vol[i] = np.nan
            else:
                a = dense_Ia[i, :, :][~np.isnan(dense_Ia[i, :, :])]
                Ia_avg[i]  = np.average(a)
                Ia_max[i]  = np.max(a)
                Ia_min[i]  = np.min(a)
                Ia_c_down[i], Ia_c_up[i] = mean_confidence_interval(a)
                
                vol[i] = vol_fac*len(a)*dx**3 # [m3]
                Ia_A_avg[i] = Ia_avg[i]*vol[i]/area # [A/m2] 
                # minimum and maximum value for area-specific current density in each slice 
                # does not have any physical meaning
                # Ia_A_max[i] = Ia_max[i]*vol[i]/area # [A/m2]
                # Ia_A_min[i] = Ia_min[i]*vol[i]/area # [A/m2]
                # Ia_A_c_down[i] = Ia_c_down[i]*vol[i]/area # [A/m2]
                # Ia_A_c_up[i] = Ia_c_up[i]*vol[i]/area # [A/m2]

        if csv_flag: create_csv_output(x, Ia_avg, Ia_min, Ia_max, Ia_c_down, Ia_c_up, 'Ia')
        if csv_flag: create_csv_output(x, Ia_A_avg, title='Ia_A')
        if plot1D_flag: plot_with_continuous_error(x, Ia_A_avg, x_title='Distance from anode (µm)', y_title='Area-specific current density (A/m2)', title='Area-specific current density', save_file=save_file)
        if plot1D_flag: plot_with_continuous_error(x, Ia_avg, Ia_min, Ia_max, Ia_c_down, Ia_c_up, x_title='Distance from anode (µm)', y_title='Volumetric current density (A/m3)', title='Volumetric current density', save_file=save_file)
    

    if inputs['output_options']['show_3D_plots']:
        if plots['cH2_3D'] and plot3D_flag:
            mats.append(dense_cH2)
            thds.append(())
            titles.append('cH2')
            log_scale.append(False)

        if plots['Vel_3D'] and plot3D_flag:
            mats.append(dense_Vel)
            thds.append(())
            titles.append('Vel')
            log_scale.append(False)
        
        if plots['Vio_3D'] and plot3D_flag:
            mats.append(dense_Vio)
            thds.append(())
            titles.append('Vio')
            log_scale.append(False)

        if plots['Ia_3D'] and plot3D_flag:
            mats.append(dense_Ia)
            thds.append(())
            titles.append('Ia')
            log_scale.append(True)

        if plots['eta_act_3D']and plot3D_flag:
            mats.append(dense_eta_act)
            thds.append(()) 
            titles.append('e_act')
            log_scale.append(False)

        if plots['eta_con_3D'] and plot3D_flag:
            mats.append(dense_eta_con)
            thds.append(())
            titles.append('e_con')
            log_scale.append(False)

        visualize_mesh(
            mat = mats,
            thd = thds,
            titles = titles,
            clip_widget = False, 
            TPB_mesh = TPB_mesh,
            log_scale = log_scale)

def create_TPB_field_variable_individual(inputs, phi_dense, indices, masks_dict, func):
    # visualize a function on the TPB
    import numpy as np
    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]
    ds = masks_dict['ds']


    TPB_mat = np.zeros(shape = N)

    for p in [0,1,2]:
        for n in indices[p]['source']:
            i,j,k = indices[p]['all_points'][n]
            if close_to_edge(inputs, i,j,k): continue

            cH2_i = phi_dense[i,j,k] if p==0 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[0][i-1:i+2,j-1:j+2,k-1:k+2]])
            Vel_i = phi_dense[i,j,k] if p==1 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[1][i-1:i+2,j-1:j+2,k-1:k+2]])
            Vio_i = phi_dense[i,j,k] if p==2 else np.average(phi_dense[i-1:i+2,j-1:j+2,k-1:k+2][ds[2][i-1:i+2,j-1:j+2,k-1:k+2]])

            TPB_mat[i,j,k] = func(cH2_i, Vel_i, Vio_i)
    
    TPB_mat[TPB_mat==0] = np.nan
    return TPB_mat

def plot_with_continuous_error(x, y, y_min=None, y_max=None, y_c_down=None, y_c_up=None, 
                               x_title='x', y_title='y', title=None, save_file=False, log_type="linear"):
    import plotly.graph_objects as go

    x = [x] if type(x) is not list else x 
    y = [y] if type(y) is not list else y
    y_min = [y_min] if type(y_min) is not list else y_min
    y_max = [y_max] if type(y_max) is not list else y_max
    y_c_down = [y_c_down] if type(y_c_down) is not list else y_c_down
    y_c_up = [y_c_up] if type(y_c_up) is not list else y_c_up

    fig = go.Figure()
    for i in range(len(y)):
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
        if y is not None:
            fig.add_trace(go.Scatter(
                name='y',
                x=x[i],
                y=y[i],
                mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
                showlegend=False,
            ))

        # add confidence level
        if y_c_down is not None:
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
    if save_file:
        file_dir = f'Binary files/1D plots/{title if title is not None else "fig"}.html'
        fig.write_html(file_dir)

def mean_confidence_interval(data, confidence=0.95):
    import numpy as np
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h

def create_field_variable_individual(inputs, phi, indices, func):
    import numpy as np
    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]
    field_mat = np.zeros(shape = N)

    for n in indices['interior']:
        i,j,k = indices['all_points'][n]
        field_mat[i,j,k] = func(phi[i,j,k])
    
    field_mat[field_mat==0] = np.nan
    return field_mat

def save_image(phase_mat):
    
    from tqdm import tqdm
    import numpy as np
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

def visualize_mesh(mat, thd=[()], blocks=[], titles=[], clip_widget=False, TPB_mesh=[], log_scale=None):
    """
    Visualizes the mesh via PyVista.
    inputs:
    mat : float
        Three dimensional matrix describing the phase data. It could be list 
    """
    import numpy as np
    import pyvista as pv
    import matplotlib.pyplot as plt
    
    pv.set_plot_theme("document")
    cmap = plt.cm.get_cmap("jet")

    subplts = len(mat)
    
    if bool(blocks): 
        p = pv.Plotter(shape=(1, subplts+1), border=True, notebook=False)
    else:
        p = pv.Plotter(shape=(1, subplts), border=True, notebook=False)
    
    for i in np.arange(subplts):
        scale = log_scale[i] if log_scale is not None else False
        sargs = dict(title=f'P{i+1}', height=0.15, vertical=True, position_x=0.01, position_y=0.8)
        sub_mat = mat[i]
        N=sub_mat.shape
        # Initializing grids
        mesh = pv.UniformGrid(dims=(N[0]+1,N[1]+1,N[2]+1))
        
        # Assigning values to grids
        mesh.cell_data["data"] = sub_mat.T.flatten()
        
        sub_thd = thd[i]
        if len(sub_thd)==0:
            mesh = mesh.threshold()
        if len(sub_thd)>0:
            mesh = mesh.threshold(sub_thd)
            
        p.subplot(0, i)
        # mesh.save(f"mesh{i}.vtk")
        if clip_widget:
            p.add_mesh_clip_plane(mesh, scalar_bar_args={'title': f'Phase{i+1}'}, cmap=cmap)
        else:
            p.add_mesh(mesh, scalar_bar_args=sargs, log_scale=scale, cmap=cmap)#, show_edges=True)
            if bool(titles):
                p.add_text(titles[i], font_size=20, position='lower_edge')
            if bool(TPB_mesh):
                p.add_mesh(TPB_mesh, line_width=10, color='k')
                # for j in range(len(TPB_mesh)):
                    # p.add_mesh(TPB_mesh[j], line_width=3, color='k')
        p.add_bounding_box(line_width=1, color='black')
        # p.add_axes(line_width=5)

    if bool(blocks):
        p.subplot(0, subplts)
        p.add_mesh(blocks)

    # if bool(TPB_mesh):
    #     p.add_mesh(TPB_mesh, line_width=3, color='k')
        
    p.link_views()
    # p.camera_position = 'xz'
    # p.camera.azimuth = 90
    # p.camera.roll += 45
    # p.camera.elevation = -45
    p.view_isometric()
    # p.save_graphic("img.eps",raster=False, painter=True)
    p.show()
    return None
    
def visualize_network(volumes, centroids, M=1):
    import numpy as np
    import pyvista as pv
    
    blocks = pv.MultiBlock()
    
    for i in np.arange(len(volumes[M-1])):
        radius = np.power(volumes[M-1][i,:]*3/4/np.pi,1/3)
        center = np.flip(centroids[M-1][i,:])
        blocks.append(pv.Sphere(radius = radius, center = center))
                      
    return blocks

def visualize_contour(mat, n_levels=5):
    import numpy as np
    import pyvista as pv
    import matplotlib.pyplot as plt

    pv.set_plot_theme("document")
    cmap = plt.cm.get_cmap("jet")

    N=mat.shape[0]

    # Initializing grids
    mesh = pv.UniformGrid(dims=(N+1,N+1,N+1))
    mesh.cell_data["data"] = mat.flatten()
    mesh = mesh.threshold()
    mesh = mesh.cell_data_to_point_data()

    p = pv.Plotter(notebook=False)
    contours = mesh.contour(np.linspace(np.nanmin(mat), np.nanmax(mat), n_levels))    
    p.add_mesh(contours, cmap=cmap)

    # show the solid phase [needs improvement]
    # solid = np.zeros_like(mat)
    # solid[np.isnan(mat)] = 1
    # solid[~np.isnan(mat)] = np.nan
    # mesh_s = pv.UniformGrid(dims=(N+1,N+1,N+1))
    # mesh_s.cell_data["data"] = solid.flatten()
    # mesh_s = mesh_s.threshold()
    # cmap = plt.cm.get_cmap("Greys")
    # p.add_mesh(mesh_s, cmap=cmap,opacity=0.5)

    p.show()
    
    return None

def close_to_edge(inputs, i,j,k):
    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]
    if \
    (i==1 and j==1) or\
    (i==1 and k==1) or\
    (j==1 and k==1) or\
    (i==N[0]-2 and j==1) or\
    (i==N[0]-2 and k==1) or\
    (j==N[1]-2 and k==1) or\
    (i==1 and j==N[1]-2) or\
    (i==1 and k==N[2]-2) or\
    (j==1 and k==N[2]-2) or\
    (i==N[0]-2 and j==N[1]-2) or\
    (i==N[0]-2 and k==N[2]-2) or\
    (j==N[1]-2 and k==N[2]-2):
        return True

# specific functions for entire cell
def visualize_3D_matrix_entire_cell(inputs, phi_dense, masks_dict, TPB_dict, titles, cH2=None, Vel_a=None, Vio=None, cO2=None, Vel_c=None, field_mat=None, TPB_mat=None):
    # visualize the solution
    import numpy as np
    import pyvista as pv
    import pandas as pd
    import plotly.express as px
    pv.set_plot_theme("document")

    N = [inputs['Nx_a']+inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']]
    ds = masks_dict['ds']
    ds_entire_cell = [None]*5
    ds_entire_cell[0] = np.concatenate((ds[0], np.zeros((inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']),dtype=bool)), axis=0)
    ds_entire_cell[1] = np.concatenate((ds[1], np.zeros((inputs['Nx_e']+inputs['Nx_c'], inputs['Ny'], inputs['Nz']),dtype=bool)), axis=0)
    ds_entire_cell[2] = ds[2]
    ds_entire_cell[3] = np.concatenate((np.zeros((inputs['Nx_a']+inputs['Nx_e'], inputs['Ny'], inputs['Nz']),dtype=bool),ds[3]),axis=0)
    ds_entire_cell[4] = np.concatenate((np.zeros((inputs['Nx_a']+inputs['Nx_e'], inputs['Ny'], inputs['Nz']),dtype=bool),ds[4]),axis=0)
    mats = []
    thds = []
    log_scale = []
    
    if Vio is not None:
        sol_Vio = np.copy(phi_dense)
        sol_Vio[ds_entire_cell[2] == False] = np.nan
        mats.append(sol_Vio)
        thds.append(())
        log_scale.append(False)

    import pyvista as pv
    vertices_a = TPB_dict['anode']['vertices']
    lines_a = TPB_dict['anode']['lines']

    vertices_c = TPB_dict['cathode']['vertices']
    lines_c = TPB_dict['cathode']['lines']

    TPB_mesh_a = pv.PolyData(vertices_a, lines=lines_a)
    TPB_mesh_c = pv.PolyData(vertices_c, lines=lines_c)
    
    
    if inputs['show_3D_plots']:
        visualize_mesh(
            mat = mats,
            thd = thds,
            titles = titles,
            clip_widget = False, 
            TPB_mesh = [TPB_mesh_a, TPB_mesh_c],
            log_scale = log_scale)

    if inputs['show_1D_plots']:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        Nx = inputs['Nx_a'] + inputs['Nx_e'] + inputs['Nx_c']
        x = np.arange(Nx)*inputs['dx']*1e6

        Vio_lin = np.zeros(Nx)
        Vio_min = np.zeros_like(Vio_lin)
        Vio_max = np.zeros_like(Vio_lin)
        Vio_c_up = np.zeros_like(Vio_lin)
        Vio_c_down = np.zeros_like(Vio_lin)

        for i in range(Nx):
            a = sol_Vio[i, :, :][~np.isnan(sol_Vio[i, :, :])]
            Vio_lin[i] = np.average(a)
            Vio_max[i] = np.max(a)
            Vio_min[i] = np.min(a)
            Vio_c_down[i], Vio_c_up[i] = mean_confidence_interval(a)

        plot_with_continuous_error(x, Vio_lin, Vio_max, Vio_min, Vio_c_down, Vio_c_up, x_title='Distance from anode (µm)', y_title='Hydrogen concentration (kg/m3)', title='Hydrogen concentration (kgm-3)')

def create_dense_matrices(phi, inputs, masks_dict, indices, field_functions):
    import numpy as np
    N = [
        inputs['microstructure']['Nx'], 
        inputs['microstructure']['Ny'], 
        inputs['microstructure']['Nz']]
    ds = masks_dict['ds']
    phi_dense = np.zeros(N)
    phi_dense[ds[0]] = phi[0]
    phi_dense[ds[1]] = phi[1]
    phi_dense[ds[2]] = phi[2]
    Ia_mat = create_TPB_field_variable_individual(inputs, phi_dense, indices, masks_dict, field_functions['Ia'])
    eta_act_mat = create_TPB_field_variable_individual(inputs, phi_dense, indices, masks_dict, field_functions['eta_act'])
    eta_conc_mat = create_field_variable_individual(inputs, phi_dense, indices[0], field_functions['eta_con'])
    sol_cH2 = np.copy(phi_dense)
    sol_cH2[ds[0] == False] = np.nan
    sol_Vel = np.copy(phi_dense)
    sol_Vel[ds[1] == False] = np.nan
    sol_Vio = np.copy(phi_dense)
    sol_Vio[ds[2] == False] = np.nan

    dense_m = {
        'phi_dense': phi_dense,
        'cH2': sol_cH2, 
        'Vel': sol_Vel, 
        'Vio': sol_Vio, 
        'Ia': Ia_mat, 
        'eta_act': eta_act_mat, 
        'eta_con': eta_conc_mat
        }
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
    
    df.to_csv('Binary files/1D plots/' + title + '.csv', index=False)