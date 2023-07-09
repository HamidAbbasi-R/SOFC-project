import plotly.express as px
import pandas as pd

# setting the colormap
# colormap = 
df = pd.read_excel('microstructure overpotential.xlsx' , sheet_name='Sheet3', engine='openpyxl')
fig = px.line(df, x="J [A/m2]", y="eta [V]", markers=True, color_discrete_sequence=px.colors.qualitative.Bold, color='legend')
fig.update_layout(
    title="Overpotential vs. Current Density for different microstructures and different hydrogen partial pressures",
    xaxis_title="Current Density [A/m2]",
    yaxis_title="Activation Overpotential [V]",
    legend_title="Topology of Microstructure and Hydrogen Partial Pressure",
)
fig.write_image("microstructure_overpotential.svg")
fig.show()