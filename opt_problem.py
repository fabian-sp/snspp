"""
@author: Fabian Schaipp
"""

import numpy as np
from ssnal_solver import stochastic_ssnal
import matplotlib.pyplot as plt
import seaborn as sns

class problem:
    
    def __init__(self, f, phi, A, x0 = None, verbose = False):
        self.f = f
        self.phi = phi
        self.A = A.copy()
        self.n = self.A.shape[1]
        self.x0 = x0
        self.verbose = verbose
        
    
    def solve(self):
        
        if self.x0 is None:
            self.x0 = np.random.rand(self.n)

        self.x, self.info = stochastic_ssnal(self.f, self.phi, self.x0, self.A, eps = 1e-4, params = None, \
                         verbose = self.verbose, measure = False)
            
        return
    
    def plot_path(self):
        
        fig, ax = plt.subplots()
        sns.heatmap(self.info['iterates'], cmap = 'coolwarm', ax = ax)