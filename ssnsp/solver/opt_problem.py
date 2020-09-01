"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .spp_solver import stochastic_prox_point
from .saga import saga
from .warm_spp import warm_spp
from .fast_gradient import stochastic_gradient

#sns.set()
#sns.set_context("paper")

color_dict = {"default": "#002A4A", "saga": "#FFB03B", "adagrad" : "#B64926", \
              "ssnsp": "#468966", "ssnsp_noVR": "grey"}

##FFF0A5
##8E2800

class problem:
    
    def __init__(self, f, phi, x0 = None, tol = 1e-3, params = dict(), verbose = False, measure = False):
        self.f = f
        self.phi = phi
        #self.A = f.A.copy()
        self.n = f.A.shape[1]
        
        self.x0 = x0
        self.tol = tol
        self.params = params
        self.verbose = verbose
        self.measure = measure
        
    
    def solve(self, solver = 'ssnsp'):
        
        self.solver = solver
        if self.x0 is None:
            self.x0 = np.zeros(self.n)
        
        if solver == 'ssnsp':
            self.x, self.xavg, self.info = stochastic_prox_point(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                         verbose = self.verbose, measure = self.measure)
        elif solver == 'saga_pure':
            self.x, self.xavg, self.info =  saga(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                                                 verbose = self.verbose, measure = self.measure)
        elif solver == 'saga' or solver == 'adagrad':
            self.x, self.xavg, self.info =  stochastic_gradient(self.f, self.phi, self.x0, solver = self.solver, tol = self.tol, params = self.params, \
                                                 verbose = self.verbose, measure = self.measure)        
        elif solver == 'warm_ssnsp':
            self.x, self.xavg, self.info = warm_spp(self.f, self.phi, self.x0, tol = self.tol, params = self.params, \
                                                    verbose = self.verbose, measure = self.measure)
        else:
            raise ValueError("Not a known solver option")
            
        return
    
    def plot_path(self, ax = None, runtime = True, mean = False, xlabel = True, ylabel = True):
        # sns.heatmap(self.info['iterates'], cmap = 'coolwarm', vmin = -1, vmax = 1, ax = ax)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams['font.size'] = 10
        
        if ax is None:
            fig, ax = plt.subplots()
            
        coeffs = self.info['iterates'][-1,:]
        c = plt.cm.Blues(abs(coeffs)/max(abs(coeffs)))
        
        if  not mean:
            to_plot = 'iterates'
            title_suffix = ''
        else:
            to_plot = 'mean_hist'
            title_suffix = ' (mean iterate)'
            
        for j in range(len(coeffs)):
            if runtime:
                ax.plot(self.info['runtime'].cumsum(), self.info[to_plot][:,j], color = c[j])
                if xlabel:
                    ax.set_xlabel('Runtime [sec]')
            else:
                ax.plot(self.info[to_plot][:,j], color = c[j])
                if xlabel:
                    ax.set_xlabel('Iteration/ epoch number')
        
        if ylabel:
            ax.set_ylabel('Coefficient')
        ax.set_title(self.solver + title_suffix)
        
        #ax.grid(axis = 'y', ls = '-', lw = .5)
        
        return
    
    def plot_objective(self, ax = None, runtime = True, mean_obj = False, label = None, marker = 'o', ls = '-', psi_star = 0, log_scale = False):
        """
        
        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axis where to plot. The default is None.
        runtime : bool, optional
            whether to plot runtime as x-axis (or iteration number). The default is True.
        mean_obj : bool, optional
            plot additionally the objective of the mean iterate. The default is False.
        label : str, optional
            label for legend. The default is None.
        marker : str, optional
            markerstyle. The default is 'o'.
        ls : str, optional
            linestyle. The default is '-'.
        psi_star : float, optional
            offset with true optimal value. The default is 0.
        log_scale : bool, optional
            plot y-axis in log scale. The default is False.

        Returns
        -------
        None.

        """
        plt.rcParams["font.family"] = "serif"
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1
        plt.rc('text', usetex=True)
        #plt.rc('xtick', labelsize=12)
        #plt.rc('ytick', labelsize=12)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if runtime:
            x = self.info['runtime'].cumsum()
        else:
            x = np.arange(len(self.info['objective']))
        
        y = self.info['objective'] - psi_star
        
        
        if label is None:
            label = self.solver
        else:
            label = self.solver + label
        
        try:
            c = color_dict[label]
        except:
            c = color_dict["default"]
        pt = ax.plot(x,y, marker = marker, ls = ls, label = label, markersize = 5, c = c)
        
        if mean_obj:
            y1 = self.info['objective_mean'] - psi_star
            ax.plot(x,y1, marker = None, ls = 'dotted', lw = 2, label = label + '_mean',\
                    c = pt[0].get_color())
            
        ax.legend()
        if runtime:
            ax.set_xlabel("Runtime [sec]", fontsize = 12)
        else:
            ax.set_xlabel("Iteration / epoch number", fontsize = 12)
        
        if psi_star == 0:
            ax.set_ylabel(r"$\psi(x^k)$", fontsize = 12)
        else:
            ax.set_ylabel(r"$\psi(x^k) - \psi^\star$", fontsize = 12)
            
        
        ax.grid(ls = '-', lw = .5)
        
        if log_scale:
            ax.set_yscale('log')
        
        return
    
    def plot_samples(self):
        assert 'samples' in self.info.keys(), "No sample information"
        
        tmpfun = lambda x: np.isin(np.arange(self.f.N), x)
        
        tmp = np.array([tmpfun(s) for s in self.info['samples']])
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