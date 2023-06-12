import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import os
import itertools

# Different configurations
folder = 'D39 - 0.93'

# props = [
#     ['dx', [30]],
#     ['d', [0.4]],
#     ['vfp', [44]],
#     ['lx', [50]],
#     ['V', [0.05]],
# ]

props = [
    ['Nx', [300]],
    ['Nyz', [100]],
    ['dx', [100]],
    ['V', [0.03,0.05]],
]

# Create file names
parameter = 'Vio'
show_max_min = True
save_svg = False

length_props = [None] * len(props)
for i in range(len(props)):
    length_props[i] = len(props[i][1])
n_curves = max(length_props)

for i in range(len(props)):
    if len(props[i][1]) < n_curves: props[i][1] *= n_curves
        
filename = [None]*n_curves
filepath = [None]*n_curves
df = [None]*n_curves
max_x = [None]*n_curves
trace = [None]*n_curves
cwd = os.getcwd()
col_pal = px.colors.qualitative.Bold
col_pal_iterator = itertools.cycle(col_pal) 
color = [None]*n_curves
for i in range(n_curves):
    name_str = ''
    for j in range(len(props)):
        name_str +=  '_' + props[j][0] + str(props[j][1][i])

    if folder == 'lattice': name_str = '_lat' + name_str
    name_str = parameter + name_str + '.csv'
    filename[i] = name_str
    filepath[i] = os.path.join(cwd, 'Binary files', '1D plots', 'csv', folder, filename[i])
    df[i] = pd.read_csv(filepath[i])
    max_x[i] = df[i].iloc[-1,0] + df[i].iloc[1,0] - df[i].iloc[0,0]

if len(set(max_x)) > 1:
    for i in range(n_curves): df[i].iloc[:,0] += max(max_x) - max_x[i]

for i in range(n_curves):
    color[i] = next(col_pal_iterator)
    trace[i] = go.Scatter(x=df[i].iloc[:,0], y=df[i].iloc[:,1], name=filename[i][:-4], line=dict(color=color[i], width=4))

if parameter == 'Vio' and show_max_min:
    for i in range(n_curves):
        trace.append(go.Scatter(x=df[i].iloc[:,0], y=df[i].iloc[:,2],
                                    line=dict(width=0),
                                    showlegend=False))
        color_RGB = px.colors.convert_colors_to_same_type(color[i], colortype='rgb')
        color_alpha = 'rgba(' + color_RGB[0][0][4:-1] + ', 0.2)'
        trace.append(go.Scatter(x=df[i].iloc[:,0], y=df[i].iloc[:,3],
                                    fillcolor = color_alpha,
                                    fill='tonexty',
                                    line=dict(width=0),
                                    showlegend=False))

# Add the traces to the figure
fig = go.Figure(data=trace)

# Set the x and y axes
fig.update_xaxes(title_text='x [micro meter]')
y_title = 'Current density [A/m2]' if parameter == 'Ia_A' else 'Ion potential [V]'
fig.update_yaxes(title_text=y_title)

# set the thickness of the lines
# fig.update_traces(line_width=4)

# save svg file
if save_svg:
    fig.write_image("plot.svg")

# Display the plot
fig.show()