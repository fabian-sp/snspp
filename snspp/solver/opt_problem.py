"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from .spp_solver import stochastic_prox_point
from .fast_gradient import stochastic_gradient

#sns.set()
#sns.set_context("paper")

color_dict = {"svrg": "#002A4A", "saga": "#FFB03B", "batch-saga": "#BF842C", "adagrad" : "#B64926", \
              "tick-svrg": "#002A4A",
              "snspp": "#468966", "default": "#142B40"}

marker_dict = {"svrg": "<", "saga": ">", "batch-saga": ">", "adagrad" : "D", \
               "tick-svrg": "<",
               "snspp": "o", "default": "+"}

class problem:
    """
    A class for composite optimization problems of the form 
    
    .. math::
        \min{x} f(x) + \phi(x)
        
    where 
    
    .. math::
        f(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(A_i x)
        
    For the case that each :math:`f_i` has scalar output, several classical algorithms are implemented:
        
        * SGD: mini-batch stochastic proximal gradient descent.
        * SAGA: available also with mini-batches using ``solver = batch-saga``.
        * SVRG
        * ADAGRAD
        * SNSPP: stochastic proximal point (with or without variance reduction).
        
    """
    def __init__(self, f, phi, A, x0 = None, tol = 1e-3, params = dict(), verbose = True, measure = True):
        """

        Parameters
        ----------
        f : loss function object
            This object describes the function :math:`f(x)`. 
            See ``snspp/helper/loss1.py`` for an example.
        phi : regularization function object
            This object describes the function :math:`\phi(x)`. See ``snspp/helper/regz.py`` for an example.
        x0 : array, optional
            Starting point. The default is zero.
        tol : float, optional
             Tolerance for the stopping criterion (sup-norm of relative change in the coefficients). The default is 1e-3.
        params : dict, optional
            Parameters for the solver. Common keys are
            
            * `alpha` : Step size. For variance reduced method constant step sizes are used. Otherwise, it decays as ``(alpha/k**beta)```with ``beta > 1/2``. 
            * `batch_size` : Size of the mini-batch.
            
        verbose : boolean, optional
            Verbosity. The default is True.
        measure : boolean, optional
            Whether to evaluate the objective after each itearion. The default is False.
            For the experiments, needs to be set to ``True``, for actual computation it is recommended to set this to ``False``.

        Returns
        -------
        None.

        """
        self.f = f
        self.phi = phi
        self.A = A
        self.n = A.shape[1]
        
        self.x0 = x0
        self.tol = tol
        self.params = params.copy()
        self.verbose = verbose
        self.measure = measure
        
    
    def solve(self, solver = 'snspp', eval_x0 = True, store_hist = True):
        
        self.solver = solver
        if self.x0 is None:
            self.x0 = np.zeros(self.n)
        
        if solver == 'snspp':
            self.x, self.info = stochastic_prox_point(self.f, self.phi, self.A, self.x0, tol = self.tol, params = self.params, \
                         verbose = self.verbose, measure = self.measure, store_hist = store_hist)

        elif solver in ['saga', 'batch-saga', 'svrg', 'adagrad', 'sgd', 'tick-svrg']:
            self.x, self.info =  stochastic_gradient(self.f, self.phi, self.A, self.x0, solver = self.solver, tol = self.tol, params = self.params, \
                                                     verbose = self.verbose, measure = self.measure)        
        else:
            raise ValueError("Not a known solver option")
        
        # evaluate at starting point
        if eval_x0 and self.measure:
            self.info['evaluations'] = np.insert(self.info['evaluations'], 0, 0)
            self.info['runtime'] = np.insert(self.info['runtime'], 0, 0)
            
            psi0 = self.f.eval(self.A@self.x0) + self.phi.eval(self.x0)
            self.info['objective'] = np.insert(self.info['objective'], 0, psi0)
            if store_hist:
                self.info['iterates'] = np.vstack((self.x0, self.info['iterates']))
        
        if self.measure:
            assert len(self.info['runtime']) == len(self.info['objective']), "Runtime + objective measurements must be of same length for plotting."   
            if store_hist:
                assert len(self.info['iterates']) == len(self.info['runtime']), "Runtime + objective measurements and iterate history must be of same length for plotting." 
        return
    
    def plot_path(self, ax = None, runtime = True, mean = False, xlabel = True, ylabel = True):
        """
        Plots the coefficients of the iterates.
        """
        plt.rcParams["font.family"] = "serif"
        plt.rcParams['font.size'] = 10
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if self.info['iterates'].shape[1] >= 1e4:
            lazy = True
        else:
            lazy = False
        
        coeffs = self.info['iterates'][-1,:]
        c = plt.cm.Blues(abs(coeffs)/max(abs(coeffs)))
        
        if not mean:
            to_plot = 'iterates'
            title_suffix = ''
        else:
            to_plot = 'mean_hist'
            title_suffix = ' (mean iterate)'
            
        for j in range(len(coeffs)):
            # for large problems only draw important coefficients
            if lazy and abs(coeffs[j]) <= 5e-2:
                continue
            
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
    
    def plot_objective(self, ax = None, runtime = True, label = None, markersize = 3, markevery = 1, lw = 0.4, ls = '-', psi_star = 0, log_scale = False):
        """
        
        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axis where to plot. The default is None.
        runtime : bool, optional
            whether to plot runtime as x-axis (else: num evaluations). The default is True.
        label : str, optional
            label for legend. The default is None.
        markersize : float, optional
            markersize. The default is 3.
        lw : float, optional
            linewidth. The default is 0.4.
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
            x = self.info['evaluations'].cumsum()
        
        y = self.info['objective'] - psi_star
        
        
        if label is None:
            label = self.solver
        else:
            label = self.solver + label
        
        try:
            c = color_dict[label]
            marker = marker_dict[label]
        except:
            c = color_dict["default"]
            marker = marker_dict["default"]
            
        ax.plot(x,y, marker = marker, ls = ls, label = label, markersize = markersize, markevery = markevery, c = c)
        
        ax.legend()
        if runtime:
            ax.set_xlabel("Runtime [sec]", fontsize = 12)
        else:
            ax.set_xlabel("Epoch", fontsize = 12)
        
        if psi_star == 0:
            ax.set_ylabel(r"$\psi(x^k)$", fontsize = 12)
        else:
            ax.set_ylabel(r"$\psi(x^k) - \psi^\star$", fontsize = 12)
                
        ax.grid(ls = '-', lw = .5)
        
        if log_scale:
            ax.set_yscale('log')
        
        return
    
    #%% newton convergence

    def plot_subproblem(self, stepsize = True, M = 20, start = 0):
        """
        For SNSPP, this plots the convergence of the semismooth Newton method for solving the subproblems.
        """
        assert self.solver == "snspp"
        
        plt.rcParams["font.family"] = "serif"
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1
        plt.rc('text', usetex=True)
        
        info = self.info['ssn_info']
        
        nrow = 4
        ncol = 5
        fig, axs = plt.subplots(nrow, ncol, figsize = (12,9))
        
        
        col_dict = {'objective': "#91BED4", 'residual': "#304269", 'step_size': "#F26101"}
        
        for j in np.arange(nrow):
            for l in np.arange(ncol):
                ax = axs[j,l]
                ix = start + j*ncol + l
                
                ax.plot(info[ix]['residual'], c = col_dict["residual"], marker = "o", ls = ':')
                ax2 = ax.twinx()
                ax2.plot(info[ix]['objective'], c = col_dict["objective"] , marker = "o", ls = "--")
                
                if stepsize:
                    ax.plot(info[ix]['step_size'], c = col_dict["step_size"], marker = "x", ls = ':')
                
                ax.set_title(f"outer iteration {ix}", fontsize = 8)
                ax.set_yscale('log')
                
                ax.set_ylim(1e-6,1e1)
                ax.grid(ls = '-', lw = .5)
                
                if l%ncol !=0:
                    ax.set_yticklabels([])
                ax2.set_yticklabels([])
                
                ax.tick_params(axis='both', labelsize=8)
                
                if l%ncol == 0:
                    ax.set_ylabel(r"$\mathcal{V}(\xi^j)$")
    
                if l%ncol == ncol-1:
                    ax2.set_ylabel(r"$\mathcal{U}(\xi^j)$")            
                
                
                if j == nrow-1:
                    ax.set_xlabel("Iteration")            
      
        fig.suptitle('Convergence of the subproblem')
        
        legend_elements = [Line2D([0], [0], marker = 'o', ls = '--', color=col_dict["objective"], label='objective'),
                           Line2D([0], [0], marker='o', ls = ':', color=col_dict["residual"], label='residual')]
     
        if stepsize:
            legend_elements.append(Line2D([0], [0], marker = 'x', ls = ':', color=col_dict["step_size"], label='step size'))    
            
        fig.legend(handles=legend_elements, loc='upper right')
        fig.subplots_adjust(hspace = 0.4)
        
        return fig
    
