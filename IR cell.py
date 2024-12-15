#%% Libraries and functions
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from modules.postprocess import visualize_mesh as vm
from scipy.ndimage import zoom
import plotly.io as pio
pio.renderers.default = 'vscode'
width = None
height = None

def plot_arrow(fig, x, y, color, axis='y'):
    fig.add_trace(go.Scatter(
    x=[x, x-15] if axis=='y' else [x, x+15],
    y=[y, y],
    yaxis=axis,
    showlegend=False,
    marker=dict(
        size=10,
        color = color,
        symbol='arrow-bar-up',
        angleref='previous',
        ),
    ))

def plot_bounds(fig, x, y_min, y_max, color, spec):
    fig.add_trace(go.Scatter(
        x=x, 
        y=y_min, 
        mode='lines', 
        name='Lower Bound', 
        line=dict(width=0),
        showlegend=False,
        yaxis='y2' if spec in ['Vio', 'R_SMR','H2_normal'] else 'y',
        ))
    fig.add_trace(go.Scatter(
        x=x, 
        y=y_max, 
        fill='tonexty', 
        fillcolor=color,
        mode='none', 
        yaxis='y2' if spec in ['Vio', 'R_SMR','H2_normal'] else 'y',
        showlegend=False,
        ))
    
def add_higer_lower_bounds():
    bound_1 = np.linspace(0, max_bound[i], len(y[i])//3)
    bound_2 = np.linspace(max_bound[i], max_bound[i]/2, len(y[i])//3)
    bound_3 = np.linspace(max_bound[i]/2, max_bound[i]/1.5, len(y[i]) - len(bound_1) - len(bound_2))
    bound = np.concatenate((bound_1, bound_2, bound_3))

    x_noise[i], y_max[i] = add_white_noise_to_curve(
        x[i], y[i]*(1+bound),
        sigma = sigma_bound[i],
        )

    x_noise[i], y_min[i] = add_white_noise_to_curve(
        x[i], y[i]*(1-bound),
        sigma = sigma_bound[i],
        )

    # update the lower bound if it is lower than the noise
    y_min[i] = np.minimum(y_min[i], y_noise[i])
    y_max[i] = np.maximum(y_max[i], y_noise[i])
    
def get_colormap_colors(cmap_name, N):
    cmap = plt.get_cmap(cmap_name)
    colors_array = [cmap(i / N) for i in range(N)]  # Get RGBA colors for N evenly spaced values
    # convert to rgba format
    colors = [f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1)' for color in colors_array]
    # colors opacified
    colors_opac = [f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.5)' for color in colors_array]
    return colors, colors_opac

def add_white_noise_to_curve(
        x,y,
        sigma=1,
        sigma_distribution=None,
        ):

    # noise distribution is the standard deviation of the noise across the curve (should be the same length as y)
    if sigma_distribution == 'linear':
        sigma = sigma * y
    elif sigma_distribution == 'quadratic':
        sigma = sigma * y**2
    elif sigma_distribution == 'cubic':
        sigma = sigma * y**3
    elif sigma_distribution == 'falling-rising':
        sigma_l = sigma     # left
        sigma_m = sigma/5   # middle
        sigma_r = sigma/2     # right
        sigma_1 = np.linspace(sigma_l, sigma_m, len(y)//2)
        sigma_2 = np.linspace(sigma_m, sigma_r, len(y)//2+1)
        sigma = np.concatenate((sigma_1, sigma_2))
    elif sigma_distribution == 'rising-falling':
        sigma_l = sigma/3
        sigma_m = sigma
        sigma_r = sigma/3
        sigma_1 = np.linspace(sigma_l, sigma_m, len(y)//2)
        sigma_2 = np.linspace(sigma_m, sigma_r, len(y)//2+1)
        sigma = np.concatenate((sigma_2, sigma_1))
    elif sigma_distribution == 'falling':
        sigma = np.linspace(sigma, sigma/5, len(y))
    elif sigma_distribution == 'rising':
        sigma_l = sigma/10
        sigma_m = sigma/5
        sigma_r = sigma
        sigma = np.linspace(sigma/5, sigma, len(y))
    elif sigma_distribution is None:
        sigma = np.ones(len(y)) * sigma
    
    y_noise = [y[i] + np.random.randn() * sigma[i] for i in range(len(y))]
    y_noise = np.array(y_noise)
    # x_noise = np.linspace(x[0], x[-1], len(y_noise))
    x_noise = x

    return x_noise, y_noise

def R_SMR(y_CH4, y_H2O):
    # cut the longer array to the same length
    min_len = min(len(y_CH4), len(y_H2O))
    y_CH4, y_H2O = y_CH4[:min_len], y_H2O[:min_len]

    # y_CH4 = np.array(y_CH4)
    # y_H2O = np.array(y_H2O)

    R_SMR = 2.035e5 * np.exp(-9.93e4 / R / T) * (y_CH4*100)**(1.29) * (y_H2O*100)**(-0.32)
    return R_SMR

def read_and_adjust_data():
    # read from folder named "internal reforming data" in the same direcoty
    path = 'Internal reforming data/'
    for i,spec in enumerate(species):
        data = np.genfromtxt(f'{path}{spec}.txt', delimiter=',')
        x[i] = np.array(data[:,0])
        y[i] = np.array(data[:,1])
    
    max_len = max([len(x[i]) for i in range(N_species)])
    for i,spec in enumerate(species):
        zoom_rate = max_len / len(x[i]) 
        x[i] = zoom(x[i], zoom_rate)
        y[i] = zoom(y[i], zoom_rate)

        # sort the data by x from smallest to largest
        x[i], y[i] = zip(*sorted(zip(x[i], y[i])))

        # use scipy.ndimage.zoom to make all the arrays the same size

        # scale x to [0, 250]
        x[i] = (x[i] - x[i][0]) / (x[i][-1] - x[i][0]) * 250


        ### between each two points, add one more point (to make the curve smoother)
        for _ in range(N_pts):
            x_new = []
            y_new = []
            for j in range(len(x[i])-1):
                x_new.append(x[i][j])
                y_new.append(y[i][j])
                # the new point is the average of the two points
                y_new.append((y[i][j]+y[i][j+1])/2)
                x_new.append((x[i][j]+x[i][j+1])/2)
            x_new.append(x[i][-1])
            y_new.append(y[i][-1])
            x[i] = np.array(x_new)
            y[i] = np.array(y_new)

def make_3D_data(x_noise, y_noise, y_min, y_max, y):
    from scipy.ndimage import median_filter
    from modules.topology import measure_TPB as mt

    phase_mat = np.load('phase_mat.npy')
    TPB_mask, _, lines, _ = mt(phase_mat, 50e-9)

    flag_pore  = phase_mat == 1
    flag_Ni = phase_mat == 2
    flag_YSZ = phase_mat == 3

    if phase_mat.shape[0] != len(x_noise[0]):
        raise ValueError('The shape of the phase matrix should be the same as the x_noise')
    
    H2O_3D = np.zeros_like(phase_mat, dtype=float)
    H2_3D = np.zeros_like(phase_mat, dtype=float)

    H2O_3D[np.where(~flag_pore)] = np.nan
    H2_3D[np.where(~flag_pore)] = np.nan
    for i in range(len(x_noise[0])):
        # # make a random matrix the same size as H2O_3D[i,...][np.where(flag_pore[i,...])]
        # mat_surf = np.zeros_like(H2O_3D[i,...][np.where(flag_pore[i,...])])
        # # assign random values between y_min and y_max to the matrix
        # mat_surf = np.random.uniform(y_min[species.index('H2O')][i], y_max[species.index('H2O')][i], mat_surf.shape)
        # H2O_3D[i,...][np.where(flag_pore[i,...])] = mat_surf
        
        # use a median filter to smooth the data
        H2O_3D[i,...][np.where(flag_pore[i,...])] = y[species.index('H2O')][i]
        H2_3D[i,...][np.where(flag_pore[i,...])] = y[species.index('H2')][i]

    sol = [H2_3D]
    return sol, TPB_mask, lines

#%% Constants and parameters
species = [     # species to read from the data
    'CO2',
    'CO',
    'CH4',
    'H2O',
    'H2',
    'Vio',
    'Ia',
    'R_SMR',
    'H2_normal',
    ]

titles = [      # titles of the plots
    'CO<sub>2',
    'CO',
    'CH<sub>4',
    'H<sub>2</sub>O',
    'H<sub>2',
    'V<sub>io',
    'I<sub>a',
    'R<sub>SMR',
    'H<sub>2</sub> No IR',
    ]

sigma = [       # standard deviation of the noise
    0.002, 
    0.002, 
    0.002, 
    0.002, 
    0.002, 
    0.001,
    0.002,
    5,
    0.0005,
    ]

N_pts=3     # number of points to add between each two points

max_bound = [   # maximum bound of the error
    0.05,
    0.05,
    0.05,
    0.03,
    0.07,
    0.001,
    0.03,
    0.07,
    0.001,
    ]

sigma_bound = np.array(sigma)/2

# TPB_d = 3.63e12     # TPB density [m/m^3]

# Visualization options
flag_2D_plots       = True
show_original       = False
add_bounds          = True
flag_comparison     = True
save_svg_2D_plots   = True
flag_3D_plot        = False
flag_TPB            = False
save_3D_arrays      = False

#%% Calculations
# constants
R = 8.3144598
T = 1073.15

N_species = len(species)
x, y = [None] * N_species, [None] * N_species
x_noise, y_noise = [None]*N_species, [None]*N_species
y_min, y_max = [None]*N_species, [None]*N_species

fig_conc = go.Figure()
fig_reac = go.Figure()
colors, colors_opac = get_colormap_colors('tab10', N_species)

read_and_adjust_data()
# replace Ia with np.nan for x<200
i = species.index('Ia') 
y[i] = np.where(x[i] < 200, np.nan, y[i])

# replace R_SMR with np.nan for x>200
i = species.index('R_SMR')
y[i] = np.where(x[i] > 200, np.nan, y[i])

for i,spec in enumerate(species):
    ### add white noise to the y data
    x_noise[i], y_noise[i] = add_white_noise_to_curve(
        x[i], y[i],
        sigma = sigma[i],
        # sigma_distribution = 'falling-rising',
        sigma_distribution = 'falling' if spec not in ['Vio','Ia','CO2'] else 'rising',
        )

    if spec == 'H2_normal': continue

    fig = fig_reac if spec in ['Ia', 'R_SMR'] else fig_conc
    
    ### add the original and noise data to the plot
    if show_original:
        fig.add_trace(go.Scatter(
            x=x[i], 
            y=y[i], 
            mode='lines', 
            name = f'{titles[i]} Original',
            line=dict(
                color='red', 
                width=2,
                dash='dash',
                ),
            # secondary y axis
            yaxis='y2' if spec in ['Vio', 'R_SMR'] else 'y',
            ))
    
    fig.add_trace(go.Scatter(
        x=x_noise[i], 
        y=y_noise[i], 
        mode='lines',
        # use the ith color in the color cycle
        line=dict(color=colors[i]),
        name=titles[i],
        # secondary y axis
        yaxis='y2' if spec in ['Vio', 'R_SMR'] else 'y',
        ))

    ### add higher and lower bounds to the curve
    if add_bounds:
        add_higer_lower_bounds()
        plot_bounds(
            fig,
            x_noise[i], 
            y_min[i], 
            y_max[i], 
            colors_opac[i], 
            spec,
            )

# compare the H2_normal curve with H2 curve
if flag_comparison:
    fig_compare = go.Figure()
    
    i = species.index('H2')
    fig_compare.add_trace(go.Scatter(
        x=x_noise[i], 
        y=y_noise[i], 
        mode='lines',
        line=dict(color=colors[i]),
        name=f'{titles[i]}</sub> IR',
        ))
    if add_bounds:
        plot_bounds(
            fig_compare,
            x_noise[i], 
            y_min[i], 
            y_max[i], 
            colors_opac[i], 
            'H2',
            )
    

    # adjust the macro model
    i = species.index('H2_normal')
    add_higer_lower_bounds()
    x_noise[i] = np.where(x_noise[i] < 150, x[i], x_noise[i])
    y_noise[i] = np.where(x_noise[i] < 150, y[i], y_noise[i])
    y_max[i] = np.where(x_noise[i] < 150, y[i], y_max[i])
    y_min[i] = np.where(x_noise[i] < 150, y[i], y_min[i])

    fig_compare.add_trace(go.Scatter(
        x=x_noise[i], 
        y=y_noise[i], 
        mode='lines',
        line=dict(color=colors[i]),
        name = titles[i],
        # secondary y axis
        yaxis='y2',
        ))
    if add_bounds:
        plot_bounds(
            fig_compare,
            x_noise[i], 
            y_min[i], 
            y_max[i], 
            colors_opac[i], 
            'H2_normal',
            )

figs = [fig_conc, fig_reac, fig_compare] if flag_comparison else [fig_conc, fig_reac]
for fig in figs:
    if fig == fig_conc:
        yaxis_title = 'Mole fraction'
        yaxis_title_secondary = 'Voltage (V)'
        y_line =0.2
        y_text = 0.1
    elif fig == fig_reac:
        yaxis_title = 'Current density (A/m²)'
        yaxis_title_secondary = 'Reaction rate (mol/m²/s)'
        y_line = 0.25
        y_text = 0.2
    elif fig == fig_compare:
        yaxis_title = 'H2 Mole fraction (IR)'
        yaxis_title_secondary = 'H2 Mole fraction (no IR)'
        y_line = 0.08
        y_text = 0.04
    
    fig.update_layout(
        xaxis_title='Distance from anode surface (µm)',
        yaxis_title=yaxis_title,
        template='simple_white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False),

        xaxis_tickvals=np.linspace(0, 250, 6),

        legend=dict(
            x=0.0,
            y=1.2,
            orientation='h',
            ),
        width=width,
        height=height,
        # change the order of legend
        legend_traceorder='reversed',
        # secondary y axis
        yaxis2=dict(
            title=yaxis_title_secondary,
            overlaying='y',
            side='right',
            ),
    )
    # draw a vertical line at x=150
    fig.add_shape(
        dict(
            type='line',
            x0=210,
            y0=0,
            x1=210,
            y1=y_line,
            line=dict(
                color='RoyalBlue',
                width=3,
                dash='dash',
                ),
            )
        )

    # add a text label on the left and right of the vertical line
    fig.add_annotation(
        dict(
            x=230,
            y=y_text,
            xref='x',
            yref='y',
            text='Electrochemical<br>active layer',
            showarrow=False,
            font=dict(
                size=10,
                color='black',
                ),
            ),
            # rotate the text
            textangle=-90,
        )

    if fig == fig_compare:
        fig.add_shape(
            dict(
                type='line',
                x0=150,
                y0=0,
                x1=150,
                y1=y_line,
                line=dict(
                    color=colors[species.index('H2_normal')],
                    width=5,
                    # dash='dash',
                    ),
                )
            )
        fig.add_annotation(
            dict(
                x=165,
                y=0.01,
                xref='x',
                yref='y',
                text='Microscale<br>model',
                showarrow=False,
                font=dict(
                    size=12,
                    color=colors[species.index('H2_normal')],
                    ),
                ),
                # rotate the text
                textangle=-90,
            )
        fig.add_annotation(
            dict(
                x=135,
                y=0.01,
                xref='x',
                yref='y',
                text='Macroscale<br>model',
                showarrow=False,
                font=dict(
                    size=12,
                    color=colors[species.index('H2_normal')],
                    ),
                ),
                # rotate the text
                textangle=-90,
            )

### add arrows to the plots
plot_arrow(fig_conc, 150, 0.06, colors[species.index('H2')])
plot_arrow(fig_conc, 150, 0.16, colors[species.index('H2O')])
plot_arrow(fig_conc, 150, 0.025, colors[species.index('CH4')])
plot_arrow(fig_conc, 150, 0.1, colors[species.index('CO')])
plot_arrow(fig_conc, 150, 0.0, colors[species.index('CO2')])
plot_arrow(fig_conc, 150, 1.05, colors[species.index('Vio')], axis='y2')

plot_arrow(fig_reac, 240, 0.1, colors[species.index('Ia')])
plot_arrow(fig_reac, 60, 300, colors[species.index('R_SMR')], axis='y2')

plot_arrow(fig_compare, 30, 0.04, colors[species.index('H2')])
plot_arrow(fig_compare, 30, 0.947, colors[species.index('H2_normal')], axis='y2')

if save_svg_2D_plots:
    fig_conc.write_image('conc.svg')
    fig_reac.write_image('reac.svg')
    fig_compare.write_image('compare.svg')

if flag_2D_plots:
    fig_conc.show()
    fig_reac.show()
    fig_compare.show()

if flag_3D_plot:
    from modules.topology import create_vertices_in_uniform_grid as cvug
    import pyvista as pv
    sol, TPB_mask, lines = make_3D_data(x_noise, y_noise, y_min, y_max, y)
    TPB_dict = {'TPB_mask': TPB_mask, 'lines': lines}
    titles = ['H2']
    TPB_mesh = pv.PolyData(
        cvug(sol[0].shape), 
        lines=lines,
        )
    vm(sol, TPB_mesh = [TPB_mesh] if flag_TPB else None, titles = titles)
    if save_3D_arrays:
        import pickle
        with open(f'test.pkl', 'wb') as file:
            pickle.dump(
                {
                    # "dx":       50e-9,
                    # "phi":      dense_m['phi_dense'],
                    # "rhoH2":    dense_m['rhoH2'],
                    # "Vel":      dense_m['Vel'],
                    "Vio":      sol[0],
                    # "rhoO2":    dense_m['rhoO2'],
                    # "I":        dense_m['I'],
                    # "eta_act":  dense_m['eta_act'],
                    # "eta_con":  dense_m['eta_con'],
                    "TPB_dict":   TPB_dict,
                    # "mask":     ,
                    }, 
                file)
