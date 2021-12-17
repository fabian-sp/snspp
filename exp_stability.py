import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LogisticRegression


from snspp.helper.data_generation import lasso_test, logreg_test, get_gisette, get_mnist
from snspp.experiments.experiment_utils import load_stability_results
from snspp.experiments.stability_utils import load_setup, create_instance, compute_psi_star, create_alpha_range,\
                                                do_grid_run, plot_result

#%%

setup_id = 'gisette1'
results = dict()

setup = load_setup(setup_id)
f, phi, A, b = create_instance(setup)
psi_star = compute_psi_star(setup, f, phi, A, b)


#%%

solvers = list(setup["solvers"].keys())

for s in solvers:
    
    params = setup["solvers"][s]["params"]
    step_size_range = np.logspace(-2,2,20)
    batch_size_range = setup["solvers"][s]["batch"]
    step_size_range = create_alpha_range(setup, s)
    
    this_res = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                           psi_tol = setup["psi_tol"], n_rep = setup["n_rep"], solver = s, solver_params = params)
    
    
    results.update({s: this_res})
        

#%% store (or load results)

np.save('data/output/exp_stability_'+setup_id+'.npy', results)    

#res_spp, res_saga, res_svrg = load_stability_results(problem_type, l1)

#%% plot runtime (until convergence) vs step size
save = False

SIGMA = 1. # plot 2SIGMA band around the mean

fig, ax = plt.subplots(figsize = (7,5))

for s in solvers:
    plot_result(results[s], ax = ax, replace_inf = setup["y_max"], sigma = SIGMA, psi_tol = setup["psi_tol"])
    

annot_y = setup["y_max"] * 0.9 # y value for annotation

ax.hlines(annot_y , ax.get_xlim()[0], ax.get_xlim()[1], 'grey', ls = '-')
ax.annotate("no convergence", (ax.get_xlim()[0]*1.1, annot_y+0.3), color = "grey", fontsize = 14)

ax.set_ylim(-1e-3,)

if save:
    fig.savefig('data/plots/exp_other/stability_'+setup_id+'.pdf', dpi = 300)


