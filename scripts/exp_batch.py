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

params_snspp = {'max_iter' : 100, 'sample_style': 'constant', 'alpha' : .7, 'reduce_variance': True}

res = dict()

for b in batch_sizes:
    
       
    params_snspp['batch_size'] = int(f.N*b)
    print(params_snspp)
    
    P = problem(f, phi, tol = 1e-9, params = params_snspp, verbose = False, measure = True)
    P.solve(solver = 'snspp')

    res[b] = P.info
    
    
#%%
res2 = dict()

fig, axs = plt.subplots(1,2,figsize = (8, 3), gridspec_kw=dict(width_ratios=[4,2]))

##############################
## first ax

ax = axs[0]

colors = sns.light_palette(color_dict['snspp'], K+1, reverse=False)
colors = sns.cubehelix_palette(K, start=.5, rot=-.75, as_cmap=False)

for j,b in enumerate(batch_sizes):
    
    x = np.arange(len(res[b]['sub_runtime']))
    y = res[b]['sub_runtime']
    #y2 = res[b]['objective'][1:]
    obj = res[b]['objective']
    #np.diff(obj).std()
    res2[b] = dict(mean_rt=np.mean(y), obj_diff_std= (obj[1:]/obj[:-1]).std())
    
    ax.plot(x,y, c=colors[j], lw=1, marker='o', markersize=4, markevery=(0,20), label=rf"$b/N={b}$ ")
    #ax2.plot(x,y2, c=colors[j], ls='--', lw=2, marker='X', markersize=5, markevery=(5,10))

ax.set_yscale('log')
ax.set_xlabel('Iteration')
ax.set_ylabel('Subproblem runtime [sec]', fontsize=10)
ax.legend(fontsize=8)

_ylim = ax.get_ylim()

##############################
## second ax

ax = axs[1]
#ax2 = ax.twinx()

y1 = [r['mean_rt'] for r in res2.values()]
y2 = [r['obj_diff_std'] for r in res2.values()]
ax.plot(res2.keys(), y1, c='darkgray', lw = 3, marker='p', markersize=8, markeredgecolor='k', label = 'subproblem runtime/iter')
ax.plot(res2.keys(), y2, c='steelblue', lw = 3, marker='s', markersize=6, markeredgecolor='k', label = r'st. dev. $\psi(x^{k+1})/\psi(x^k)$')

ax.grid(ls = '-', lw = .5) 
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$b/N$')

ax.set_ylim(_ylim)
ax.legend(fontsize=8)

fig.tight_layout()


if False:
    fig.savefig('../data/plots/exp_mnist/batch_size.pdf')

