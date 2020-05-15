import numpy as np


N = 8
n = 8
#m = np.random.randint(low = 3, high = 10, size = N)
m = np.ones(N, dtype = 'int')

A = []
for i in np.arange(N):
    A.append(np.random.rand(m[i], n))


A = np.vstack(A)
b = np.random.randn(m.sum())

x = np.random.rand(n)

# for testing only
sample_size = 5
alpha = .1



#%%
phi = Norm1(.1)    
phi.prox(np.ones(3), alpha = 1)


f = lsq(A, b)
