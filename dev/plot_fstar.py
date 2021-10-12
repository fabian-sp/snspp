import numpy as np
import matplotlib.pyplot as plt

def huber(x, mu, b, eps):
    if np.abs(x-b) <= mu:
        y = (x-b)**2/(2*mu) + eps/2*(x**2)
    else:
        y = np.abs(x-b) - mu/2  + eps/2*(x**2)
    return y


vec_huber = np.vectorize(huber)


mu = 1
b = 0.
eps = 0.001

z=2.

X = np.linspace(-100,1000,10000)
Y = z*X - vec_huber(X, mu, b, eps)


plt.figure()
plt.plot(X,Y)
