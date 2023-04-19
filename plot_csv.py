import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import os
import itertools

# Different configurations
Nx = [
    100,
    200,
    300,
    400,
    500]

Nyz = [
    100] * len(Nx)

dx = [
    50] * len(Nx)

V = [
    0.05] * len(Nx)

# Create file names
parameter = 'Vio'
show_max_min = True
filename = [None]*len(Nx)
filepath = [None]*len(Nx)
df = [None]*len(Nx)
max_x = [None]*len(Nx)
trace = [None]*len(Nx)
cwd = os.getcwd()
col_pal = px.colors.qualitative.Bold
col_pal_iterator = itertools.cycle(col_pal) 
color = [None]*len(Nx)
for i in range(len(Nx)):
    Vstr = str(V[i])
    filename[i] =  parameter + f'_Nx{Nx[i]}_Nyz{Nyz[i]}_dx{dx[i]}_V{Vstr[2:]}.csv'
    filepath[i] = os.path.join(cwd, 'Binary files', '1D plots', 'csv', filename[i])
    df[i] = pd.read_csv(filepath[i])
    max_x[i] = df[i].iloc[-1,0] + df[i].iloc[1,0] - df[i].iloc[0,0]

if len(set(max_x)) > 1:
    for i in range(len(Nx)): df[i].iloc[:,0] += max(max_x) - max_x[i]

for i in range(len(Nx)):
    color[i] = next(col_pal_iterator)
    trace[i] = go.Scatter(x=df[i].iloc[:,0], y=df[i].iloc[:,1], name=filename[i][:-4], line=dict(color=color[i], width=4))

if parameter == 'Vio' and show_max_min:
    for i in range(len(Nx)):
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

# Display the plot
fig.show()