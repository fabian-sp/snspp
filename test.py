import numpy as np

m=100
n=10

A = np.random.randn(m,n)
b = np.random.randn(m)
x = np.ones(n)

f, fstar, g, gstar, Hstar = lsq_functions(b)

y = g(gstar(A@x))


#abs(y - A@x)




#%%
N = 8
n = 10
m = np.random.randint(low = 3, high = 10, size = N)

A = []
for i in np.arange(N):
    A.append(np.random.rand(m[i], n))


A = np.vstack(A)

    


