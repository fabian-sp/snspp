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
poly = -gamma* z**3 + (x+2*gamma*b)*z**2 +(-2*x*b-2-gamma*v-gamma*b**2)*z - (x*v+x*b**2+2*b)

# first solution is the real one!
sol = sp.solvers.solve(poly, z)[0]
#sol = sp.solveset(poly, z, domain=sp.S.Reals)

#g = f.diff(z).simplify()
#sol = sp.solvers.solve(x-g, z)

zstar = sol.simplify()

print(sp.latex(zstar))


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
