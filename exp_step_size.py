import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import Lasso, LogisticRegression


from snspp.helper.data_generation import lasso_test, logreg_test, get_gisette, get_mnist
from snspp.solver.opt_problem import problem, color_dict, marker_dict
from snspp.experiments.experiment_utils import initialize_solvers, load_stability_results
from snspp.experiments.stability_utils import do_grid_run, plot_result

problem_type = "gisette"

# parameter setup
if problem_type == "gisette":
    l1 = 0.05
    EPOCHS = 55 # epochs for SAGA/SVRG
    MAX_ITER = 230 # max iter for SNSPP
    PSI_TOL = 1e-4 # relative accuracy for objective to be considered as converged
    
    Y_MAX = 8. # y-value of not-converged stepsizes


elif problem_type == "mnist":
    l1 = 0.02
    EPOCHS = 30 # epochs for SAGA/SVRG
    MAX_ITER = 180 # max iter for SNSPP
    PSI_TOL = 1e-4 # relative accuracy for objective to be considered as converged
    
    Y_MAX = 25. # y-value of not-converged stepsizes

elif problem_type in ["logreg", "lasso"]:
    N = 100; n = 10; k = 5
    l1 = 0.01
    EPOCHS = 50 # epochs for SAGA/SVRG
    MAX_ITER = 150 # max iter for SNSPP
    PSI_TOL = 1e-3 # relative accuracy for objective to be considered as converged


N_REP = 5 # number of repetitions for each setting

#%% 

if problem_type == "lasso":
    xsol, A, b, f, phi, _, _ = lasso_test(N, n, k, l1, block = False, noise = 0.1, kappa = 15., dist = 'ortho')

elif problem_type == "logreg":
    xsol, A, b, f, phi, _, _ = logreg_test(N, n, k, lambda1 = l1, noise = 0.1, kappa = 10., dist = 'ortho')

elif problem_type == "gisette":
    f, phi, A, b, _, _ = get_gisette(lambda1 = l1)

elif problem_type == "mnist":
    f, phi, A, b, _, _ = get_mnist(lambda1 = l1)

initialize_solvers(f, phi)

#%% solve with scikt or large max_iter to get psi_star

if problem_type == "lasso":
    sk = Lasso(alpha = l1/2, fit_intercept = False, tol = 1e-8, selection = 'cyclic', max_iter = 1e6)
    sk.fit(f.A,b)

    xsol = sk.coef_.copy()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)

elif problem_type in ["logreg", "gisette", "mnist"]:
    sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-20, \
                            solver = 'saga', max_iter = 200, verbose = 1)
    sk.fit(A, b)
    
    xsol = sk.coef_.copy().squeeze()
    psi_star = f.eval(xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)
    
    
# elif problem_type == "tstudent":
#     ref_params = {'n_epochs': 100, 'alpha': 2}
#     ref_P = problem(f, phi, tol = 1e-6, params = ref_params, verbose = True, measure = True)
#     ref_P.solve(solver = 'saga')
#     ref_P.plot_objective()
#     psi_star = ref_P.info["objective"][-1]
#     print("Optimal value: ", psi_star)
    

#%%



#%% SNSPP

solver_params = {'max_iter': MAX_ITER, 'sample_style': 'constant', 'reduce_variance': True}

step_size_range = np.logspace(-2,2,20)
batch_size_range = np.array([0.005,0.01,0.05])

res_spp = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                      psi_tol = PSI_TOL, n_rep = N_REP, solver = "snspp", solver_params = solver_params)


#%% SAGA

solver_params = {'n_epochs': EPOCHS}

step_size_range = np.logspace(-1,3,20)
batch_size_range = []

res_saga = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                       psi_tol = PSI_TOL, n_rep = N_REP, solver = "saga", solver_params = solver_params)


#%% SVRG

solver_params = {'n_epochs': EPOCHS}

step_size_range = np.logspace(0,6,25)
batch_size_range = np.array([0.005,0.01,0.05])

res_svrg = do_grid_run(f, phi, step_size_range, batch_size_range = batch_size_range, psi_star = psi_star, \
                       psi_tol = PSI_TOL, n_rep = N_REP, solver = "svrg", solver_params = solver_params)



#%% store (or load results)

strl1 = str(l1).replace('.','')
filename = f'stability_{problem_type}_l1_{strl1}_psistar_' + str(float(PSI_TOL)).split('.')[1]

res_to_save = dict()
res_to_save.update({'snspp': res_spp})
res_to_save.update({'saga': res_saga})
res_to_save.update({'svrg': res_svrg})

np.save('data/output/exp_'+filename+'.npy', res_to_save)    

#res_spp, res_saga, res_svrg = load_stability_results(problem_type, l1)

#%% plot runtime (until convergence) vs step size
save = False

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%

SIGMA = 1. # plot 2SIGMA band around the mean

fig, ax = plt.subplots(figsize = (7,5))

plot_result(res_spp, ax = ax, replace_inf = Y_MAX, sigma = SIGMA, psi_tol = PSI_TOL)
plot_result(res_saga, ax = ax, replace_inf = Y_MAX, sigma = SIGMA, psi_tol = PSI_TOL)
plot_result(res_svrg, ax = ax, replace_inf = Y_MAX, sigma = SIGMA, psi_tol = PSI_TOL)

annot_y = Y_MAX * 0.9 # y value for annotation

ax.hlines(annot_y , ax.get_xlim()[0], ax.get_xlim()[1], 'grey', ls = '-')
ax.annotate("no convergence", (ax.get_xlim()[0]*1.1, annot_y+0.3), color = "grey", fontsize = 14)

ax.set_ylim(-1e-3,)

if save:
    fig.savefig('data/plots/exp_other/'+filename+'.pdf', dpi = 300)

#%% plot objective of last iterate vs step size

# fig, ax = plt.subplots()

# for k in np.arange(K):
#     c_arr = np.array(colors[k]).reshape(1,-1)
#     #ax.scatter(A[k,:], OBJ_ERR[k,:], c = c_arr, edgecolors = 'k', label = rf"$b =  N \cdot$ {batch_size[k]} ")
#     ax.plot(ALPHA, OBJ_ERR[k,:], c = colors[k], linestyle = '--', marker = 'o', label = rf"$b =  N \cdot$ {batch_size[k]} ")

# ax.set_xlabel(r"Step size $\alpha$")    
# ax.set_ylabel(r"$(\psi(x^k)-\psi^\star)/\psi^\star$")   

# ax.set_xscale('log')
# ax.set_yscale('log')

# ax.set_ylim(1e-12,1)
# ax.legend()


