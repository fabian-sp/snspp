"""
@author: Fabian Schaipp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
import time

from ssnsp.helper.data_generation import lasso_test
from ssnsp.solver.opt_problem import problem


#%% generate data

gammas = np.linspace(0.1, 1, 5)
n = 5000
k = 100
l1 = .15 
kappa = 1e4

for g in gammas:

    N = int(g*n)
      
    xsol, A, b, f, phi = lasso_test(N, n, k, l1, block = False, kappa = kappa)
    
    print("Matrix shape: ", f.A.shape)
      
    #sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-9, max_iter = 20000, selection = 'cyclic')
    
    #start = time.time()
    #sk.fit(A,b)
    #end = time.time()
    
    #print(f"Computing time: {end-start} sec")
    
    #x_sk = sk.coef_.copy().squeeze()
    
    #print(f.eval(x_sk) +phi.eval(x_sk))


    #%% solve with SAGA
    
    params = {'n_epochs' : 300, 'reg': 1e-4}
    
    Q = problem(f, phi, tol = 1e-9, params = params, verbose = True, measure = True)
    
    Q.solve(solver = 'saga')
    
    print(f.eval(Q.x) +phi.eval(Q.x))
    

#%% solve with SSNSP

    params = {'max_iter' : 15, 'batch_size': f.N, 'sample_style': 'fast_increasing', 'alpha_C' : 10., 'n_epochs': 5}
    
    P = problem(f, phi, tol = 1e-7, params = params, verbose = True, measure = True)
    
    P.solve(solver = 'ssnsp')
    
    
    fig,ax = plt.subplots(figsize = (6,4))
    Q.plot_objective(ax = ax, ls = '--', marker = '<')
    P.plot_objective(ax = ax)
    ax.set_yscale('log')
    
    
    all_x = pd.DataFrame(np.vstack((xsol, Q.x, P.x)).T, columns = ['true', 'saga', 'spp'])

    print(np.linalg.norm(xsol-P.x)/np.linalg.norm(xsol))
    
