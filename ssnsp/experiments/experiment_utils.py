"""
@author: Fabian Schaipp
"""

import numpy as np
import matplotlib.pyplot as plt
from ..solver.opt_problem import color_dict

def plot_multiple(allP, ax = None, label = "ssnsp", name = None):
    
    if name is None:
        name = label
    
    if ax is None:
            fig, ax = plt.subplots()
            
    K = len(allP)
    
    all_obj = np.vstack([allP[k]["objective"] for k in range(K)])
    all_mean = all_obj.mean(axis=0)
    all_std = all_obj.std(axis=0)

    all_rt = np.vstack([allP[k]["runtime"] for k in range(K)]).mean(axis=0).cumsum()
    
    
    ax.plot(all_rt, all_mean, marker = 'o', color = color_dict[label], label = name)
    ax.fill_between(all_rt, all_mean-all_std, all_mean+all_std, \
                    color = color_dict[label], alpha = .5)
        
    return
