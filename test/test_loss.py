import numpy as np
import os 

os.chdir(os.path.pardir)


from snspp.helper.loss import huber_loss


n = 100
N = 200

A = np.random.rand(N,n)
b = np.zeros(N)

mu = np.ones(N)
f = huber_loss(A,b,mu)


x0 = np.zeros(n)
x1 = np.random.randn(n)

r = A@x1 -b

S = np.arange(10, dtype = int)


f.eval(x0)


for i in S:
    print(f.g(r[i], i))


xi = np.random.randn(len(S))

f.fstar_vec(xi, S)
f.gstar_vec(xi, S)
f.Hstar_vec(xi, S)
