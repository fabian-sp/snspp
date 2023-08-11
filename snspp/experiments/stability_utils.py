"""
@author: Fabian Schaipp
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso, LogisticRegression

from snspp.helper.data_generation import tstudent_test, logreg_test, get_gisette, get_mnist, get_sido, get_libsvm, get_higgs, get_poly, get_e2006
from snspp.helper.data_generation import get_sido_reg, get_mnist_reg
from snspp.solver.opt_problem import problem, color_dict, marker_dict
from snspp.experiments.experiment_utils import initialize_solvers, logreg_accuracy

def load_setup(setup_id = ''):
    
    file = open('../data/setups/' + setup_id + '.json',)
    setup = json.load(file)

    return setup

def create_instance(setup):
    
    if setup['instance']['dataset'] == "tstudent":
        _kappa = setup['instance'].get('kappa', 15.)
        _nu = setup['instance'].get('nu', 1)
        
        f, phi, A, X_train, y_train, _, _, _ = tstudent_test(setup['instance']['N'], setup['instance']['n'], setup['instance']['k'], setup['instance']['l1'], \
                                              v = _nu, noise = 0.1, kappa = _kappa, dist = 'ortho')

    elif setup['instance']['dataset'] == "logreg":
        f, phi, A, X_train, y_train, _, _, _ = logreg_test(setup['instance']['N'], setup['instance']['n'], setup['instance']['k'], lambda1 = setup['instance']['l1'],\
                                               noise = 0.1, kappa = 15., dist = 'ortho')
    
    elif setup['instance']['dataset'] == "gisette":
        f, phi, A, X_train, y_train, _, _ = get_gisette(lambda1 = setup['instance']['l1'])
    
    elif setup['instance']['dataset'] == "mnist":
        f, phi, A, X_train, y_train, _, _ = get_mnist(lambda1 = setup['instance']['l1'])
        
    elif setup['instance']['dataset'] == "sido":
        f, phi, A, X_train, y_train, _, _ = get_sido(lambda1 = setup['instance']['l1'])
        
    elif setup['instance']['dataset'] == "higgs":
        f, phi, A, X_train, y_train, _, _ = get_higgs(lambda1 = setup['instance']['l1'], train_size=None)
    
    elif setup['instance']['dataset'] in ["madelon"]:
        f, phi, A, X_train, y_train, _, _ = get_poly(name = setup['instance']['dataset'], lambda1 = setup['instance']['l1'], train_size=None,\
                                                     scale=True, poly=setup['instance']['poly'])
    
    elif setup['instance']['dataset'] in ["rcv1", "covtype"]:
        f, phi, A, X_train, y_train, _, _ = get_libsvm(name = setup['instance']['dataset'], lambda1 = setup['instance']['l1'], train_size=0.8)
        
    elif setup['instance']['dataset'] == 'e2006':
        f, phi, A, X_train, y_train, _, _ = get_e2006(lambda1 = setup['instance']['l1'], train_size = 0.8)
    
    elif setup['instance']['dataset'] == 'sido_reg':
        f, phi, A, X_train, y_train, _, _ = get_sido_reg(lambda1 = setup['instance']['l1'], train_size = 0.8,
                                                         v = setup['instance']['nu'], k = setup['instance']['k'])
    
    elif setup['instance']['dataset'] == 'mnist_reg':
        f, phi, A, X_train, y_train, _, _ = get_mnist_reg(lambda1 = setup['instance']['l1'], train_size = 0.8,
                                                         v = setup['instance']['nu'], k = setup['instance']['k'])
    

    # IMPORTANT: Initialize numba
    initialize_solvers(f, phi, A)

    return f, phi, A, X_train, y_train

def compute_psi_star(setup, f, phi, A, X_train, y_train):
    
    _max_iter = setup.get("max_iter_star", 200)
    
    if setup['instance']['loss'] == "logistic":
        sk = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-20, \
                            solver = 'saga', max_iter = _max_iter, verbose = 1)
        sk.fit(X_train, y_train)
        xsol = sk.coef_.copy().squeeze()

        print("Train accuracy: ", logreg_accuracy(xsol, X_train, y_train))

    elif setup['instance']['loss'] == "squared":
        sk = Lasso(alpha = phi.lambda1/2, fit_intercept = False, tol = 1e-20, selection = 'cyclic', max_iter = _max_iter)
        sk.fit(X_train, y_train)
        xsol = sk.coef_.copy().squeeze()
        
    elif setup['instance']['loss'] == "tstudent":
        orP = problem(f, phi, A, tol = 1e-20, params = {'n_epochs': _max_iter}, verbose = True, measure = False)
        orP.solve(solver = 'saga')
        xsol = orP.x.copy()
        
    psi_star = f.eval(A@xsol) + phi.eval(xsol)
    print("Optimal value: ", psi_star)
    print("Nonzeros: ", np.count_nonzero(xsol))
 
    return psi_star, xsol

def compute_x0(setup, f, phi, A, X_train, y_train):
    assert setup["start"] >= 0
    
    if setup["start"] == 0:
        x0 = None
    # compute setup['start'] many epochs for starting point
    else:
        if setup['instance']['loss'] == "logistic":
            sk0 = LogisticRegression(penalty = 'l1', C = 1/(f.N * phi.lambda1), fit_intercept= False, tol = 1e-8, \
                                     solver = 'saga', max_iter = 1, verbose = 0).fit(X_train, y_train)
            x0 = sk0.coef_.squeeze()
        
        # default step size should be identical to scikit (1/3L)
        else:
            Q = problem(f, phi, A, tol = 1e-20, params = {'n_epochs': setup["start"]}, verbose = False, measure = False)
            Q.solve(solver = 'saga')
            x0 = Q.x.copy()
        
        psi0 = f.eval(A@x0) + phi.eval(x0)
        print("psi(x0) = ", psi0)
        print("x0 max", x0.max())
            
    return x0

def create_alpha_range(setup, method):
    
    amin = setup["methods"][method]["alpha_min"]
    amax = setup["methods"][method]["alpha_max"]
    n_ = setup["methods"][method]["n_alpha"]
    
    return np.logspace(amin, amax, n_)

    
#%% MAIN FUNCTION

def do_grid_run(f, phi, A, step_size_range, batch_size_range = [], psi_star = 0, psi_tol = 1e-3, n_rep = 5, \
                solver = "snspp", x0 = None, solver_params = dict()):
    
    ALPHA = step_size_range.copy()
    
    if len(batch_size_range) == 0:
        _batch_size_one = True
        batch_size_range = [1/f.N]
    else:
        _batch_size_one = False

    BATCH = batch_size_range.copy()
    
    GRID_A, GRID_B = np.meshgrid(ALPHA, BATCH)
    
    # K ~ len(batch_size_range), L ~ len(step_size_range)
    K,L = GRID_A.shape
        
    RTIME = np.ones_like(GRID_A) * np.inf
    RT_STD = np.ones_like(GRID_A) * np.inf
    OBJ = np.ones_like(GRID_A) * np.inf
    CONVERGED = np.zeros_like(GRID_A)
    NITER = np.ones_like(GRID_A) * np.inf
    
    for l in np.arange(L):
        
        for k in np.arange(K):
            this_params = solver_params.copy()
            
            print("######################################")
            
            if not _batch_size_one:
                this_params['batch_size'] = max(1, int(GRID_B[k,l] * f.N))
                print(f"BATCH = {this_params['batch_size']}")
            else:
                assert this_params.get('batch_size') is None

            this_params['alpha'] = GRID_A[k,l]
            print(f"ALPHA = {this_params['alpha']}")
                       
            # repeat n_rep times
            this_obj = list(); this_time = list(); this_stop_iter = list()
            for j_rep in np.arange(n_rep):
                try:
                    # SOLVE
                    P = problem(f, phi, A, x0 = x0, tol = 1e-20, params = this_params, verbose = False, measure = True)
                    P.solve(solver = solver)
                          
                    obj_arr = P.info['objective'].copy()
                    
                    print(f"OBJECTIVE = {obj_arr[-1]}")
                    
                    if np.any(obj_arr <= psi_star *(1+psi_tol)):
                        stop = np.where(obj_arr <= psi_star *(1+psi_tol))[0][0]
                        
                        # account for possibility of reaching accuracy inside the epoch --> take lower bound for runtime
                        # first entry is starting point
                        if solver != 'snspp':
                            if stop <= 1:
                                print("Convergence during first EPOCH!")
                            stop -= 1
                            
                        this_stop_iter.append(stop)
                        _rt = P.info['runtime'].cumsum()[stop]
                                
                        this_time.append(_rt)
                        this_obj.append(obj_arr[-1])
                        
                        print(f"RUNTIME = {P.info['runtime'].cumsum()[stop]}")
                        
                    else:
                        this_stop_iter.append(np.inf)
                        this_time.append(np.inf)
                        this_obj.append(obj_arr[-1])
                        
                        print("NO CONVERGENCE!")
                        break
                except:
                    print("SOLVER FAILED!")
                    this_stop_iter.append(np.inf)
                    this_time.append(np.inf)
                    this_obj.append(np.inf)
                    break
                
            # set as CONVERGED if all repetitions converged
            CONVERGED[k,l] = np.all(~np.isinf(this_stop_iter))
            OBJ[k,l] = np.mean(this_obj)
            RTIME[k,l] = np.mean(this_time)
            RT_STD[k,l] = np.std(this_time)
            NITER[k,l] = np.mean(this_stop_iter)

        ALPHA[l] = this_params['alpha']
    
    CONVERGED = CONVERGED.astype(bool)     
    
    assert np.all(~np.isinf(RTIME) == CONVERGED), "Runtime and convergence arrays are incosistent!"
    assert np.all(~np.isnan(ALPHA)), "Step sizes contains nans!"
    
    results = {'step_size': ALPHA, 'batch_size': BATCH, 'objective': OBJ, 'runtime': RTIME, 'runtime_std': RT_STD,\
               'n_iter': NITER, 'converged': CONVERGED, 'solver': solver}
    
    return results


#%% PLOTTING

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)


saga_best_batch_size = {'mnist1': [0.001],
                        'gisette1': [0.005, 0.01],
                        'sido1': [0.001],
                        'covtype1': [0.0001, 0.0005],
                        'higgs2': [1e-5],
                        'madelon1': [0.005]
                        }

def get_ymax(results, methods):
    ymax = 0.
    for m in methods:
        r = results[m].copy()
        if not np.any(r['converged']):
            pass
        else:
            this_max = r['runtime'][r['converged']].max()
            ymax = max(ymax, this_max)
    
    return 1.2*ymax

def plot_result(setup_id, method, res, ax = None, replace_inf = 10., sigma = 0., psi_tol = 1e-3, label = None):
    
    K,L = res['runtime'].shape
    rt = res['runtime'].copy()
    rt_std = res['runtime_std'].copy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize = (7,5))
    
    if label is None:
        label =  res['solver']
        
    try:
        color = color_dict[res["solver"]]
        marker = marker_dict[res["solver"]]
    except:
        color = color_dict["default"]
        marker = marker_dict["default"]

    if method == 'snspp-skip':
        color = '#234334' #'#1C3528'
        label = 'snspp-skip'

    colors = sns.light_palette(color, K+2, reverse = True)
    
    # filter to best batch size (for SAGA only; if best not recorded plot all)
    _filter_saga = True

    if method == 'batch-saga':
        if _filter_saga and (saga_best_batch_size.get(setup_id) is not None):
            idx = [res['batch_size'].index(_b) for _b in saga_best_batch_size.get(setup_id)]
        else:
            _filter_saga = False
        
        if _filter_saga: # color shades depend on filter or not    
            colors = sns.light_palette(color, 5, reverse = True)[1:]
            #col = sns.light_palette(color, 5, reverse = True)[1]
            #colors = K*[col] # full color is used for b=1, here use first shade always
        else:
            colors = sns.light_palette(color, K+2, reverse = True)[1:] 

    color_ix = 0
    for k in np.arange(K):    
        
        if _filter_saga and method == 'batch-saga':
            if k not in idx:
                continue
            #print(colors[color_ix])

        rt[k,:][~res['converged'][k,:]] = replace_inf
        rt_std[k,:][~res['converged'][k,:]] = 0 
                
        if K > 1:
            legend_label = label + ", " + rf"$b =  N \cdot${res['batch_size'][k]} "
        else:
            legend_label = label + ", " + r"$b = 1$"
            
        ax.plot(res['step_size'], rt[k,:], c = colors[color_ix], linestyle = '-', marker = marker, markersize = 4,\
                label = legend_label)
        
        # add standard dev of runtime
        if sigma > 0.:
            ax.fill_between(res['step_size'], rt[k,:] - sigma*rt_std[k,:], rt[k,:] + sigma*rt_std[k,:],\
                            color = colors[k], alpha = 0.5)
            
        color_ix += 1

    ax.set_xlabel(r"Step size $\alpha$")    
    ax.set_ylabel(r"Runtime until convergence [sec]")    
    
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend(loc = 'upper right', fontsize = 8)
    ax.set_title(rf'Convergence = objective less than {1+psi_tol}$\psi^\star$')

    return ax

#%%
##########################################################################
## Store and read
##########################################################################
    
def load_stability_results(setup_id):
    
    tmp = np.load(f'../data/output/exp_stability_'+setup_id+'.npy', allow_pickle = True)
    res = tmp[()]

    return res