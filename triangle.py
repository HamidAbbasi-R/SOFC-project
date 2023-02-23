def triangle():
  # import numpy as np
  # from scipy.sparse import linalg
  # from scipy.sparse import csr_matrix
  from skimage.filters import gaussian
  # import plotly.express as px
  # from tqdm import tqdm
  # from scipy import ndimage as ndi
  # import plotly.express as px
  # import pyvista as pv
  # import os
  # from dill import dump
  # from dill import load
  # import warnings
  # from sympy import diff, exp, lambdify, log, simplify, symbols
  # import matplotlib.pyplot as plt

  # # from sympy import diff, exp, lambdify, log, simplify, symbols

  # if pv.rcParams['use_ipyvtk']:
  #     print('Using ipyvtk-simple')
  # for i in tqdm(range(100)):
  #   a = i*2
  # A = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]])
  # fig = px.imshow(A)
  # # fig.show()
  # fig = plt.imshow(A)
  # A = csr_matrix(A)
  # b = np.array([1, 2, 4])
  # x = linalg.gmres(A, b)
  # A = gaussian(A.todense(), sigma=1)
  # A = ndi.gaussian_filter(A, sigma=1)
  # return A,x

  # # create a equilateral triangle
  # p = [[],[],[]]
  # p[0] = np.array([0,0])
  # p[1] = np.array([1,0])
  # p[2] = np.array([0.5,0.8660254037844386])
  # # p[2] = np.array([1,2])

  # # first point
  # coord = first_point

  # # create N random points inside the triangle
  # pts = [p[0], p[1], p[2], coord]
  # for i in range(int(N)):
  #     a = np.random.choice(3,1)
  #     coord = (coord + p[a[0]])/2
  #     pts.append(coord)

  # # plot the points
  # import plotly.express as px
  # fig = px.scatter(x=[p[0] for p in pts], y=[p[1] for p in pts])
  # fig.update_traces(marker=dict(size=4, color='DarkSlateGrey'))
  # fig.update_yaxes(
  #     scaleanchor = "x",
  #     scaleratio = 1,
  #   )
  # fig.show()