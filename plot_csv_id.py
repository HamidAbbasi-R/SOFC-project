# def plot_csv(param,ids):
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import os
import itertools

# ids = [111,112,113,114]
ids = [65, 70, 21, 75, 80, 85]
# Create file names
# if param == 'Ion potential':
#     parameter = 'Vio'
# elif param == 'Current density':
#     parameter = 'Ia_A'
parameter = 'Ia_A'
show_max_min = False
save_svg = False

n_curves = len(ids)
        
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
    # name_str is id with three ditits
    filename[i] = parameter + '_' + str(ids[i]).zfill(3) + '.csv'
    filepath[i] = os.path.join(cwd, 'Binary files', '1D plots', 'csv', filename[i])
    df[i] = pd.read_csv(filepath[i])
    max_x[i] = df[i].iloc[-1,0] + df[i].iloc[1,0] - df[i].iloc[0,0]

if len(set(max_x)) > 1:
    for i in range(n_curves): df[i].iloc[:,0] += max(max_x) - max_x[i]

for i in range(n_curves):
    color[i] = next(col_pal_iterator)
    trace[i] = go.Scatter(x=df[i].iloc[:,0], 
                        y=df[i].iloc[:,1], 
                        name=filename[i][:-4], 
                        mode='lines+markers',
                        line=dict(color=color[i], width=2),
                        marker=dict(size=5))

if parameter == 'Vio' and show_max_min:
    for i in range(n_curves):
        trace.append(go.Scatter(x=df[i].iloc[:,0], 
                                y=df[i].iloc[:,2],
                                line=dict(width=0),
                                showlegend=False))
        color_RGB = px.colors.convert_colors_to_same_type(color[i], colortype='rgb')
        color_alpha = 'rgba(' + color_RGB[0][0][4:-1] + ', 0.2)'
        trace.append(go.Scatter(x=df[i].iloc[:,0], 
                                y=df[i].iloc[:,3],
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
# fig.show(renderer="notebook")