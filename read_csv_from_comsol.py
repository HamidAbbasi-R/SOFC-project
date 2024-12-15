#%%
# read csv file
import csv
import numpy as np
import plotly.express as px

# read csv file
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

fig = px.line(x=x_tot, y=flux_tot, title='Flux')



