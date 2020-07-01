"""
author: Fabian Schaipp
"""

import numpy as np
from .spp_solver import stochastic_prox_point
from .saga import saga



def warm_spp(f, phi, x0, tol = 1e-4, params = dict(), verbose = False, measure = False):
    
    ###### PHASE 1: SAGA ########
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 2
    
    x_t_saga, x_mean_saga, info_saga = saga(f, phi, x0, tol, params, verbose, measure)
    
    ###### PHASE 2: SPP ########
    
    #xi_0 = dict(zip(np.arange(f.N), [ -.5 * np.ones(m[i]) for i in np.arange(f.N)]))
    xi_0 = None
    
    x_t, x_mean, info_spp = stochastic_prox_point(f, phi, x_t_saga, xi_0, tol, params, verbose, measure)
    
    ###### Combine diagnostics ########
    info = dict()
    
    info['objective'] = np.hstack((info_saga['objective'], info_spp['objective']))
    info['objective_mean'] = np.hstack((info_saga['objective_mean'], info_spp['objective_mean']))
    info['iterates'] = np.vstack((info_saga['iterates'], info_spp['iterates']))
    info['step_sizes'] = np.hstack((info_saga['step_sizes'], info_spp['step_sizes']))
    
    if measure:
        info['runtime'] = np.hstack((info_saga['runtime'], info_spp['runtime']))

    info['xi_hist'] = info_spp['xi_hist'].copy()
    info['n_iter_saga'] = len(info_saga['objective'])
    
    return x_t, x_mean, info
