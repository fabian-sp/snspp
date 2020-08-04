"""
Miscallaneous snippets for SPP solver
"""

##################
## SOlve CG faster after Sun et al.
##################

# start = time.time()
#         if m.max() == 1 and phi.name == '1norm':
#             L = np.sqrt(tmp_d)
#             L_inv = 1./L 
#             tilde_rhs = diag_mult(L_inv, rhs)
#             tilde_A =  diag_mult(L_inv, subA)[:, bool_d].astype('float32')
            
#             r_dim = bool_d.sum()
#             B = (sample_size/alpha) * np.eye(r_dim) + tilde_A.T @ tilde_A
#             B_inv = np.linalg.inv(B)
            
#             tilde_d = (np.eye(sample_size) - tilde_A @ B_inv @ tilde_A.T) @ tilde_rhs
#             d2 = diag_mult(L_inv,  tilde_d)
            
#         end = time.time(); print(end-start) 



# def diag_mult(d, A):
#     """
#     if d is a vector and A a matrix, computes diag(d)@A fast
#     """
#     if len(A.shape) == 2:
#         res = d[:,np.newaxis] * A
#     else:
#         res = d*A
#     return res


##################
## updazte xi when (1,) arrays
##################

# for l in range(sample_size):
#    xi[S[l]] = xi_stack[[l]].copy()



##################
## Prints
##################
#print(f"U_old: {U_old} with residual {np.linalg.norm(rhs)}")
#print(f"U_new: {U_new} vs . { U_old + newton_params['mu'] * beta * (d @ -rhs)} with beta being {beta}")

#counter = 0
# reset if getting stuck
            #counter +=1
            # if counter >= 15:
            #     print("Semismooth Newton: reset step size and ignore Armijo")
            #     beta = .8
            #     break



###################
# for debugging/testing
###################

#S = np.arange(100)
#z = np.random.rand(784)
#alpha = .1
#xi = dict(zip(np.arange(f.N), [ -.5 * np.ones(f.m[i]) for i in np.arange(f.N)]))


###################
# variance reduction
####################

# if gradient_table is not None:
#     #xi_stack_old = np.hstack([xi_old[i] for i in S])
#     #xi_full_old = np.hstack([xi_old[i] for i in range(f.N)])
#     #correct =  (alpha/sample_size) * (subA.T @ xi_stack_old) - (alpha/f.N) * (f.A.T @ xi_full_old)
    
#     #tmp_g = np.vstack([gradient_table[i,:] for i in S])
#     #correct = (alpha/sample_size) * tmp_g.sum(axis = 0) - (alpha/f.N) * gradient_table.sum(axis = 0)
#     #print(np.linalg.norm(correct))
#     correct = 0.      
# else:
#     correct = 0.



#new_z = new_x - (1/sample_size) * (subA.T @ xi_stack)
#eta = np.linalg.norm(new_x - phi.prox(new_z, alpha = 1))