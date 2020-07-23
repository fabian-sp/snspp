"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .spp_solver import stochastic_prox_point
from .saga import saga
from .warm_spp import warm_spp
from .saga_fast import saga_fast

class problem:
    
    def __init__(self, f, phi, x0 = None, tol = 1e-3, params = dict(), verbose = False, measure = False):
        self.f = f
        self.phi = phi
        self.A = f.A.copy()
        self.n = self.A.shape[1]
        
        self.x0 = x0
        self.tol = tol
        self.params = params
        self.verbose = verbose
        self.measure = measure
        
    
    def solve(self, solver = 'ssnsp'):
        
        if self.x0 is None:
            self.x0 = np.zeros(self.n)
        
        if solver == 'ssnsp':
            self.x, self.xavg, self.info = stochastic_prox_point(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                         verbose = self.verbose, measure = self.measure)
        elif solver == 'saga':
            self.x, self.xavg, self.info =  saga(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                                                 verbose = self.verbose, measure = self.measure)
        elif solver == 'saga_fast':
            self.x, self.xavg, self.info =  saga_fast(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                                                 verbose = self.verbose, measure = self.measure)        
        elif solver == 'warm_ssnsp':
            self.x, self.xavg, self.info = warm_spp(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                                                    verbose = self.verbose, measure = self.measure)
        else:
            raise ValueError("Not a known solver option")
            
        return
    
    def plot_path(self):
        
        if self.n <= 100:
            fig, axs = plt.subplots(1,2)
            ax = axs[0]
            sns.heatmap(self.info['iterates'], cmap = 'coolwarm', vmin = -1, vmax = 1, ax = ax)
            
            ax = axs[1]
            ax.plot(self.info['iterates'])
            ax.set_xlabel('iteration number')
            ax.set_ylabel('coefficient')
        else:
            fig, ax = plt.subplots(1,1)
            
            coeffs = self.info['iterates'][-1,:]
            c = plt.cm.Blues(abs(coeffs)/max(abs(coeffs)))
            
            for j in range(len(coeffs)):
                ax.plot(self.info['iterates'][:,j], color = c[j])
            
            ax.set_xlabel('iteration number')
            ax.set_ylabel('coefficient')
        return
    
    def plot_objective(self):
        fig, ax = plt.subplots()
        ax.plot(self.info['objective'])
        ax.plot(self.info['objective_mean'])
        ax.legend(['obj(x_t)', 'obj(x_star)'])
        
        return
    
    def plot_samples(self):
        assert 'samples' in self.info.keys(), "No sample information"
        
        tmpfun = lambda x: np.isin(np.arange(self.f.N), x)
        
        tmp = np.apply_along_axis(tmpfun, axis = 1, arr = self.info['samples'])
        tmp2 = tmp.sum(axis=0)
        
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(1, 10, wspace=0.4, hspace=0.3)
        ax1 = fig.add_subplot(grid[:, :-3])
        ax2 = fig.add_subplot(grid[:, -3:])
        
        sns.heatmap(tmp.T, square = False, annot = False, cmap = 'Blues', vmin = 0, vmax = tmp.max(), cbar = False, \
                    xticklabels = [], ax = ax1)
        sns.heatmap(tmp2[:,np.newaxis], square = False, annot = True, cmap = 'Blues', cbar = False, \
                    xticklabels = [], yticklabels = [], ax = ax2)
        
        return