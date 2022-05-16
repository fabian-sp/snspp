"""
@author: Fabian Schaipp

Runs experiments for step size selection stability. A setup file is used to specify all the relevant parameters, see `../data/setups` for examples.

"""

import numpy as np
import matplotlib.pyplot as plt

from snspp.experiments.stability_utils import load_setup, create_instance, compute_psi_star, compute_x0, create_alpha_range,\
                                                do_grid_run, plot_result, load_stability_results

#%%

setup_id = 'tstudent2'
results = dict()

setup = load_setup(setup_id)
f, phi, A, b = create_instance(setup)
psi_star, xsol = compute_psi_star(setup, f, phi, A, b)
x0 = compute_x0(setup, f, phi)

#%%

methods = list(setup["methods"].keys())

for mt in methods:
    
    params = setup["methods"][mt]["params"]
    batch_size_range = setup["methods"][mt]["batch"]
    step_size_range = create_alpha_range(setup, mt)
    
    this_res = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                           psi_tol = setup["psi_tol"], n_rep = setup["n_rep"], solver = setup["methods"][mt]["solver"],\
                           solver_params = params, x0 = x0)
    
    
    results.update({mt: this_res})
        

#%% store (or load results)

np.save('../data/output/exp_stability_'+setup_id+'.npy', results)    

#results = load_stability_results(setup_id)

#%% plot runtime (until convergence) vs step size
save = False

SIGMA = 1. # plot 2SIGMA band around the mean

fig, ax = plt.subplots(figsize = (7,5))

for mt in methods:
    plot_result(results[mt], ax = ax, replace_inf = setup["y_max"], sigma = SIGMA, psi_tol = setup["psi_tol"])
    

annot_y = setup["y_max"] * 0.9 # y value for annotation
ax.hlines(annot_y , ax.get_xlim()[0], ax.get_xlim()[1], 'grey', ls = '-')
ax.annotate("no convergence", (ax.get_xlim()[0]*1.5, annot_y*1.02), color = "grey", fontsize = 14)

ax.set_ylim(0,)

if save:
    fig.savefig('../data/plots/exp_other/stability_'+setup_id+'.pdf', dpi = 300)


