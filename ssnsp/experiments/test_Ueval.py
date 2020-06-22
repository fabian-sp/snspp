
alpha = .5

S = [4,5]
sample_size = 2

dims = np.repeat(np.arange(f.N),f.m)
subA = np.vstack([f.A[dims == i,:] for i in S])

sub_dims = np.repeat(range(sample_size), f.m[S])

def Ueval(xi_stack, f, phi, x, alpha, S, sub_dims, subA):
    
    sample_size = len(S)
    
    z = x - (alpha/sample_size) * (subA.T @ xi_stack)
    tmp = .5 * np.linalg.norm(z)**2 - phi.moreau(z, alpha)
    
    res = sum([f.fstar(xi_stack[sub_dims == l], S[l]) for l in range(sample_size)]) + (sample_size/alpha) * tmp
    
    return res.squeeze()


x = np.random.randn(500)

all_xi = np.linspace(-1,1,200)
Y1, Y2 = np.meshgrid(all_xi, all_xi)

res = np.zeros(Y1.shape)

for i in np.arange(Y1.shape[0]):
    for l in np.arange(Y1.shape[1]):
        
        y = np.array([Y1[i,l], Y2[i,l]])
        
        res[i,l] = Ueval(y, f, phi, x, alpha, S, sub_dims, subA)
        
        
#%%
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(Y1,Y2,res)