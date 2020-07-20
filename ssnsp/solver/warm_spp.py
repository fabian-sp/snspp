"""
author: Fabian Schaipp
"""

import numpy as np
from .spp_solver import stochastic_prox_point
from .saga_fast import saga_fast
from ..helper.utils import compute_full_xi



def warm_spp(f, phi, x0, tol = 1e-4, params = dict(), verbose = False, measure = False):
    
    info = dict()
    ###### PHASE 1: SAGA ########
    if 'n_epochs' not in params.keys():    
        params['n_epochs'] = 2
    
    x_t_saga, x_mean_saga, info_saga = saga_fast(f, phi, x0, tol, params, verbose, measure)
    
    ###### PHASE 2: SPP ########
    
    xi_0 = compute_full_xi(f, x_t_saga)
    info['xi_0'] = xi_0.copy()
    #xi_0 = None
    
    x_t, x_mean, info_spp = stochastic_prox_point(f, phi, x_t_saga, xi_0, tol, params, verbose, measure)
    
    ###### Combine diagnostics ########
    
    info['objective'] = np.hstack((info_saga['objective'], info_spp['objective']))
    info['objective_mean'] = np.hstack((info_saga['objective_mean'], info_spp['objective_mean']))
    info['iterates'] = np.vstack((info_saga['iterates'], info_spp['iterates']))
    info['step_sizes'] = np.hstack((info_saga['step_sizes'], info_spp['step_sizes']))
    
    
    info['runtime'] = np.hstack((info_saga['runtime'], info_spp['runtime']))
    assert len(info_saga['runtime']) == len(info_saga['objective'])
    
    info['xi_hist'] = info_spp['xi_hist'].copy()
    info['n_iter_saga'] = len(info_saga['objective'])# params['n_epochs'] * f.N
    
    info['ssn_info'] = info_spp['ssn_info']
    
    return x_t, x_mean, info
