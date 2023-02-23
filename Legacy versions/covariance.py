#%%
import microstructure as ms
import numpy as np
import math
import plotly.express as px
from scipy.special import erfinv
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.fft import fft, ifft
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# create the entire domain
Nx, Ny = 401, 401
p = [0.5,0.5]
domain = ms.create_phase_data(
    voxels = [Nx,Ny], 
    vol_frac = p,
    sigma = 4, 
    seed = [70,30], 
    display = True,
    periodic = False,
    )

d1 = np.copy(domain)
d1[d1!=1] = 0
C1 = np.zeros(Ny-1)     # covariance of the real microstructure (eq.1 Moussaoui et al. 2018)
C1[0] = p[0]
for h in range(1,Ny-1):
    C1[h] = np.mean(np.logical_and(d1[:,:-h], d1[:,h:]))
# C1bar = 1- 2*C1[0] + C1     # covariance of the complementary set. eq.13 Abdallah et al. 2016

# d2 = np.copy(domain)
# d2[d2!=2] = 0
# C2 = np.zeros(Ny-1)     # covariance of the real microstructure (eq.1 Moussaoui et al. 2018)
# C2[0] = p[1]
# for h in range(1,Ny-1):
#     C2[h] = np.mean(np.logical_and(d2[:,:-h], d2[:,h:]))
# C2bar = 1- 2*C2[0] + C2    # covariance of the complementary set. eq.13 Abdallah et al. 2016

CX = C1             # covariance of the 1st random set X. eq.31 Abdallah et al. 2016
# CY = C2/C1bar       # covariance of the 2nd random set Y. eq.32 Abdallah et al. 2016

epsx = 0.5
lambdax = math.sqrt(2)*erfinv(2*epsx-1)     # inverse of eq.6 Moussaoui et al. 2018

def func1(x):
    # Eq.10 Moussaoui et al. 2018
    f = 1/2/np.pi*integrate.quad(lambda t: 1/np.sqrt(1-t**2)*np.exp(-lambdax**2/(1+t)), 1, x)[0] - CX[h] + CX[0]
    return f

def func2(x):
    # Eq.10 Moussaoui et al. 2018
    f = 1/2/np.pi*integrate.quad(lambda t: 1/np.sqrt(1-t**2)*np.exp(-lambdax**2/(1+t)), 0, x)[0] - CX[h]
    return f


rhoX1 = np.ones(Ny-1)
rhoX2 = np.ones(Ny-1)
x0 = 0
for h in tqdm(range(0,Ny-1)):
    rhoX1[h] = fsolve(func1, x0)
    rhoX2[h] = fsolve(func2, x0)
    # rho_x[h] = fsolve(func, [1])

rhoX1[rhoX1==x0] = 1         # set the value of rhoX to 1 if the solution cannot be found (in few first points)
rhoX2[rhoX2==x0] = 1   
wX = ifft(np.sqrt(fft(rhoX1)))       # Eq. 30 Abdallah et al. 2016
rhoX_test = np.convolve(wX.real,wX.real, mode='same')      # Eq. 9 Moussaoui et al. 2018 - to check with rhoX to see if the ifft and fft are correct

fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=np.arange(len(rhoX1.real)),y=rhoX1.real), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(len(rhoX1.real)),y=rhoX_test.real), row=2, col=1)
fig.show()

UZ = np.random.normal(size=[Nx-1,Ny-1])
GZ = ifft(fft(UZ) * np.sqrt(fft(rhoX1)).real) 
GZ = np.real(GZ)
GZ[GZ> lambdax] = 1
GZ[GZ<=lambdax] = 0

fig = px.imshow(GZ)
fig.show()

# fig = px.scatter(rhoX)
# fig.show()

# fig = px.scatter(CY)
# fig.show()


# fig = px.imshow(filtered_field)
# fig.show()