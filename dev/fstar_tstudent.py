"""
This file serves to derive the convex conjugate of the (regularized) likelihood of the t-student loss
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

sp.init_printing(use_unicode=True)


x,b,v, gamma = sp.symbols('x,b,v, gamma')
z = sp.symbols('z', real = True)

f = sp.log(1+((z-b)**2/v)) + gamma/2*z**2
obj = x*z - f

#%% compute stationary point
gamma = 0.251
b = 1.
v = 1.
x = 1.

poly = -gamma* z**3 + (x+2*gamma*b)*z**2 +(-2*x*b-2-gamma*v-gamma*b**2)*z - (x*v+x*b**2+2*b)
poly_fun = lambda z: -gamma* z**3 + (x+2*gamma*b)*z**2 +(-2*x*b-2-gamma*v-gamma*b**2)*z - (x*v+x*b**2+2*b)

# first solution is the real one!
sol = sp.solvers.solve(poly, z)[0]
zstar = sol.simplify()

print(sp.latex(zstar))

#%% draw cubic polynomial

Z = np.linspace(-5,5,100)

plt.figure()
plt.plot(Z, poly_fun(Z))
plt.hlines(0, -5, 5)

#%% compute weak convexity constant

x,b,v = sp.symbols('x,b,v')
z = sp.symbols('z', real = True)

f = sp.log(1+((x-b)**2/v))

g = f.diff(x).simplify()
H = g.diff(x).simplify()

# get stat. point sof Hessian
sols = sp.solvers.solve(H.diff(x), x)

for s in sols:    
    print(H.subs(x, s))


#%%
   
# A= np.random.randn(50,100)
# b = np.random.randn(50)
# b[0] = 0.
# x = np.random.randn(100)

# f = tstudent_loss(A, b, v=1.)

# z = np.array([1.], dtype = np.float64)

# f._zstar(1,1)

# f.fstar(z,1)
# f.gstar(z,1)
# f.Hstar(z,1)


# f.g(np.array([4]),4)

# for i in range(1000):
    
#     x = np.random.randn(1)
#     y = np.random.randn(1)
    
#     print(f.f(x,3) + f.gamma/2 * x**2 + f.fstar(y,3) - x*y)
#     assert (f.f(x,3) + f.gamma/2*x**2 + f.fstar(y,3) - x*y) >= 0



# all_x = np.linspace(-100,100, 2000)
# all_f = np.zeros_like(all_x)
# all_g = np.zeros_like(all_x)
# all_h = np.zeros_like(all_x)

# for j in range(len(all_x)):
    
#     xx = all_x[j]
#     all_f[j] = f.fstar(np.array([xx]), 166)
#     all_g[j] = f.gstar(np.array([xx]), 166)
#     all_h[j] = f.Hstar(np.array([xx]), 166)

    
# plt.plot(all_x, all_f)
# plt.plot(all_x, all_g)
# plt.plot(all_x, all_h)


