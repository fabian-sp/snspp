import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


sp.init_printing(use_unicode=True)


x,b,v = sp.symbols('x,b,v,z')
z = sp.symbols('z', real = True)

f = sp.log(1+((z-b)**2/v)) + 1/(8*v)*z**2

obj = x*z - f

#%% compute stationary point
poly = z**3 + (-4*v*x-2*b)*z**2 +(b**2+8*x*v*b+9*v)*z -4*v*(x*v+x*b**2+2*b)

# first solution is the real one!
sol = sp.solvers.solve(poly, z)[0]
#sol = sp.solveset(poly, z, domain=sp.S.Reals)

zstar = sol.simplify()

print(sp.latex(zstar))


#%%

fstar = obj.subs(z,zstar).simplify()
