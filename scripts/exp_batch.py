"""
@author: Fabian Schaipp

Runs experiments for step size selection stability. A setup file is used to specify all the relevant parameters, see `../data/setups` for examples.

"""

import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from snspp.solver.opt_problem import problem, color_dict
from snspp.helper.data_generation import get_mnist

from snspp.experiments.experiment_utils import initialize_solvers


f, phi, X_train, y_train, X_test, y_test = get_mnist()
initialize_solvers(f, phi)

#%%

batch_sizes = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
K = len(batch_sizes)

params_snspp = {'max_iter' : 100, 'sample_style': 'constant', 'alpha' : 0.5, 'reduce_variance': True}

res = dict()

for b in batch_sizes:
    
       
    params_snspp['batch_size'] = int(f.N*b)
    print(params_snspp)
    
    P = problem(f, phi, tol = 1e-9, params = params_snspp, verbose = False, measure = True)
    P.solve(solver = 'snspp')

    res[b] = P.info
    
    
#%%
#psi_star =  0.552
mean_rt = dict()

fig, axs = plt.subplots(1,2,figsize = (7, 3), gridspec_kw=dict(width_ratios=[5,2]))

##############################
## first ax

ax = axs[0]
#ax2 = ax.twinx()

colors = sns.light_palette(color_dict['snspp'], K+1, reverse=False)
colors = sns.cubehelix_palette(K, start=.5, rot=-.75, as_cmap=False)

for j,b in enumerate(batch_sizes):
    
    x = np.arange(len(res[b]['sub_runtime']))
    y = res[b]['sub_runtime']
    #y2 = res[b]['objective'][1:] - psi_star
    mean_rt[b] = np.mean(y)
    
    ax.plot(x,y, c=colors[j], lw=1, marker='o', markersize=4, markevery=(0,20), label=rf"$b/N={b}$ ")
    #ax2.plot(x,y2, c=colors[j], ls='--', lw=2, marker='X', markersize=5, markevery=(5,10))

ax.set_yscale('log')
#ax2.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Subproblem runtime [sec]', fontsize=10)
ax.legend(fontsize=8)

_ylim = ax.get_ylim()

##############################
## second ax

ax = axs[1]
ax.plot(mean_rt.keys(), mean_rt.values(), c='darkgray', lw = 3, marker='p', markersize=8, markeredgecolor='k')
ax.grid(ls = '-', lw = .5) 
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$b/N$')
ax.set_ylabel('subproblem runtime/iter ', fontsize=10)
ax.set_ylim(_ylim)

fig.tight_layout()


if False:
    fig.savefig('../data/plots/exp_mnist/batch_size.pdf')

