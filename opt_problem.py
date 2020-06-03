"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spp_solver import stochastic_prox_point


class problem:
    
    def __init__(self, f, phi, x0 = None, params = dict(), verbose = False):
        self.f = f
        self.phi = phi
        self.A = f.A.copy()
        self.n = self.A.shape[1]
        
        self.x0 = x0
        self.params = params
        self.verbose = verbose
        
    
    def solve(self):
        
        if self.x0 is None:
            self.x0 = np.zeros(self.n)

        self.x, self.xavg, self.info = stochastic_prox_point(self.f, self.phi, self.x0, eps = 1e-4, params = self.params, \
                         verbose = self.verbose, measure = False)
        
        return
    
    def plot_path(self):
        
        fig, ax = plt.subplots()
        sns.heatmap(self.info['iterates'], cmap = 'coolwarm', vmin = -1, vmax = 1, ax = ax)
        
    def plot_objective(self):
        fig, ax = plt.subplots()
        ax.plot(self.info['objective'])
        ax.plot(self.info['objective_mean'])
        ax.legend(['obj(x_t)', 'obj(x_star)'])
        
        return
    
    def plot_samples(self):
        tmpfun = lambda x: np.isin(np.arange(self.f.N), x)
        
        tmp = np.apply_along_axis(tmpfun, axis = 1, arr = self.info['samples'])
        tmp2 = tmp.sum(axis=0)
        
        fig = plt.figure(figsize=(6, 6))
        grid = plt.GridSpec(1, 10, wspace=0.4, hspace=0.3)
        ax1 = fig.add_subplot(grid[:, :-3])
        ax2 = fig.add_subplot(grid[:, -3:])
        
        sns.heatmap(tmp.T, square = False, annot = False, cmap = 'Blues', cbar = False, \
                    xticklabels = [], ax = ax1)
        sns.heatmap(tmp2[:,np.newaxis], square = False, annot = True, cmap = 'Blues', cbar = False, \
                    xticklabels = [], yticklabels = [], ax = ax2)
        
        return