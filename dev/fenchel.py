"""
This file serves to derive the convex conjugate of a loss
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sp.init_printing(use_unicode=True)

# standard symbols
z,u = sp.symbols('z,u')

### PSEUDOHUBER ####
b,mu = sp.symbols('b,mu') # specific symbols 
f = mu*sp.sqrt(1+((z-b)/mu)**2)
uhat = b + (mu*z)/sp.sqrt(1-z**2)

#%%

def compute_conjugate(f, uhat = None, danskin = True):
     
    f1 = f.diff(z).simplify() # f prime
    H = u*z - f.subs(z,u) # conjugate objective
    
    if uhat is None:
        eq1 = z - f1.subs(z,u) # stat. point equation

        sols = [s.simplify() for s in sp.solve(eq1, u)]
        print(f"Number of stationary points found by Sympy: {len(sols)}")
        uhat = sols[0]
        print(f"Chosen uhat: {uhat}")
        
    fast = H.subs(u, uhat).simplify() # plug in
    
    #sp.simplify(fast1-uhat)
    if danskin:
        fast1 = uhat
        fast2 = uhat.diff(z).simplify()
    else:
        fast1 = fast.diff(z).simplify()
        fast2 = fast1.diff(z).simplify()
        
    return f1, H, fast, fast1, fast2

f1, H, fast, fast1, fast2 = compute_conjugate(f, uhat = uhat, danskin = True)

print(sp.latex(fast1))

#%%

def plot_fun(g, arg = z, sub0 = dict(), xrange = np.linspace(-5,5,100), ax = None, label = '', c = 'k'):
    """
    """
    if ax is None:
        fig, ax = plt.subplots()
   
    g0 = g.subs(sub0)
    gnum = sp.lambdify(arg, g0)
    y = gnum(xrange)
     
    ax.plot(xrange, y, label = label, c = c)
    
    return (xrange, y)

#%% plot loss and conjugates

sub0 = {mu:0.5, b:0}
X = np.linspace(-1,1,1000)

colors = ['#2C3E50','#E74C3C','#3498DB','#2980B9']

fig, ax = plt.subplots()        
_ = plot_fun(f, arg = z, sub0 = sub0, xrange = X, ax = ax, label = 'f', c= colors[0])
_ = plot_fun(fast, arg = z, sub0 = sub0, xrange = X, ax = ax, label = r'$f^\ast$', c=colors[1])
_ = plot_fun(fast1, arg = z, sub0 = sub0, xrange = X, ax = ax, label = r"$(f^\ast)$", c=colors[2])
_ = plot_fun(fast2, arg = z, sub0 = sub0, xrange = X, ax = ax, label = r"$(f^\ast)''$", c=colors[3])

ax.set_ylim(-2,2)
ax.legend()


#%% plot objective of conjugate problem

sub0 = {mu:0.5, b:0}
Z = np.linspace(-1,1,20) # range of fixed z values
U = np.linspace(-5,5,100) # xaxis
pal = sns.color_palette("rocket", len(Z))

fig, ax = plt.subplots()
for j in range(len(Z)):
    sub0[z] = Z[j]
    plot_fun(H, arg = u, sub0 = sub0, xrange = U, ax = ax, label = '', c = pal[j])
