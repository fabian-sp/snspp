"""
@author: Fabian Schaipp

Runs experiments for step size selection stability. A setup file is used to specify all the relevant parameters, see `../data/setups` for examples.

"""

import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from snspp.solver.opt_problem import problem, color_dict
from snspp.helper.data_generation import get_mnist

from snspp.experiments.experiment_utils import initialize_solvers

from sklearn.linear_model import LogisticRegression

#%%

f, phi, A, X_train, y_train, X_test, y_test = get_mnist()
initialize_solvers(f, phi, A)

sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-9, \
                        solver = 'saga', max_iter = 300, verbose = 1)

sk.fit(X_train, y_train)
x_sk = sk.coef_.copy().squeeze()
psi_star = f.eval(A@x_sk) + phi.eval(x_sk)

#%%

step_sizes = [1e-1, 0.5, 1.]
batch_sizes = [1e-3, 5e-3, 1e-2, 5e-2]

K = len(batch_sizes)

params_snspp = {'max_iter' : 100, 'sample_style': 'constant', 'reduce_variance': True}

res = dict()

for b in batch_sizes:
    for a in step_sizes:
        _key = (a,b) 
           
        params_snspp['batch_size'] = int(f.N*b)
        params_snspp['alpha'] = a
        
        print(params_snspp)
        
        P = problem(f, phi, A, tol = 1e-9, params = params_snspp, verbose = False, measure = True)
        P.solve(solver = 'snspp')
    
        res[_key] = P.info
        
    
#%%
from matplotlib.lines import Line2D

res2 = list()

fig, axs = plt.subplots(1,2,figsize = (8, 3.5), gridspec_kw=dict(width_ratios=[4,2]))

##############################
## first ax

ax = axs[0]

#colors = sns.light_palette(color_dict['snspp'], K+1, reverse=False)
#colors = ["#abc9c8", "#72aeb6", "#4692b0",  "#134b73"]

colors = np.array(["#c969a1", "#ce4441", "#ee8577", "#eb7926", "#ffbb44", "#859b6c", "#62929a", "#004f63", "#122451"])
colors = colors[[1,3,5,7]]
# one color per batch, brightness+ls for step size
cpals = dict()

for j,b in enumerate(batch_sizes):
    cpals[b] = sns.light_palette(colors[j], 4, reverse=True)[:len(step_sizes)][::-1]    

lss = ['-', '--', ':']
lw = 2.
markers = ['.', 's', 'P', '*']

batch_handles = [Line2D([0], [0], color=colors[j], lw=lw, marker=markers[j]) for j in range(len(batch_sizes))] 
step_handles = [Line2D([0], [0], color='darkgray', ls=_ls, lw=lw) for _ls in lss] 

labels = [rf"$b/N={b}$" for b in batch_sizes] + [rf"$\alpha={a}$" for a in step_sizes]

### plot
plot_runtime_x = False
for _k,_v in res.items():
    
    a,b = _k
    
    x = res[_k]['runtime'].cumsum()
    y = res[_k]['sub_runtime']
    y2 = res[_k]['objective'] - psi_star
    
    obj = res[_k]['objective']
    res2.append(dict(a=a, b=b, mean_rt=np.mean(y), obj_diff_std= (obj[1:]/obj[:-1]).std()))
    
    j = batch_sizes.index(b)
    l = step_sizes.index(a)
    
    col = cpals[b][step_sizes.index(a)]
    
    if plot_runtime_x:
        ax.plot(x, y2, c=col, ls = lss[l], lw=lw, marker=markers[j], markersize=6, markevery=(4*j,30), alpha=0.99)
    else:
        ax.plot(y2, c=col, ls = lss[l], lw=lw, marker=markers[j], markersize=6, markevery=(4*j,30), alpha=0.99)

ax.set_yscale('log')
if plot_runtime_x:
    ax.set_xscale('log')
    ax.set_xlabel('Runtime [sec]')
else:
    ax.set_xlabel('Iteration')
    
ax.set_ylabel(r'$\psi(x^k)-\psi^\star$', fontsize=10)
ax.legend(batch_handles+step_handles, labels, fontsize=8, ncol=2)

#_ylim = ax.get_ylim()

##############################
## second ax
df=pd.DataFrame(res2)

ax = axs[1]
#ax2 = ax.twinx()

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax.plot(df.groupby('b')['mean_rt'].mean(), c='darkgray', lw = 3, marker='p', markersize=8, markeredgecolor='k', label = 'subproblem runtime/iter')
#ax.plot(res2.keys(), y2, c='steelblue', lw = 3, marker='s', markersize=6, markeredgecolor='k', label = r'st. dev. $\psi(x^{k+1})/\psi(x^k)$')

ax.grid(ls = '-', lw = .5) 
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$b/N$')
ax.set_ylabel('Subproblem runtime [sec]')

#ax.set_ylim(_ylim)
#ax.legend(fontsize=8)

fig.tight_layout()


if False:
    fig.savefig('../data/plots/exp_mnist/batch_size.pdf')

