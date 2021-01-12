# stochastic-ssnal

## Introduction
A Python package for solving problems of the form

<img src="https://latex.codecogs.com/gif.latex?\min_xf(x)+\varphi(x)" title="problem formulation"/>

where the first part of the objective has the special form

<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{N}\sum_{i=1}^{N}f_i(A_ix)" title="f structure"/>

This problem structure is common in statistical learning problems: each summand of `f` is the loss at one data sample and `phi` is a regularizer.

## Functions as objects
For all solvers of this packages, `f` and `phi` have to be instances of a class. As the solvers need certain information on the involved functions `f` and `phi`, these classes need to have several methods and attributes implemented.

We list the methods and attributes that these objects need to have for the algorithms. This is relevant only if you want to solve problems which functions that are not yet implemented. For a list of implemented functions, see below.

### Loss functions `f`
Methods:
* `eval(self, x)`: evaluates the function at `x`.
* `f(self, x, i)`: evaluates `f_i` at `x`.
* `g(self, x, i)`: evaluates the gradient of `f_i` at `x`.
* `fstar(self, x, i)`, `gstar(self, x, i)` and `Hstar(self, x, i)`: evaluates the Fenchel conjugate (its gradient/ its Hessian) of `f_i` at `x`.

Note that `fstar` (and `gstar`, `Hstar`) evaluate each sample `i` individually. In many applications, `f_i` is identical for every `i` (up to data input). In this case, the performance is improved if vectorized methods are implemented, i.e. `fstar_vec(self, x, S)` which computes the conjugate at x for a batch of indices `S`. See `ssnsp/helper/lasso` for an example. The algorithm detects automatically if such a method is implemented, hence the function called for solving is the same in both cases.


Attributes:
* `name`: name of `f`. This is needed e.g. to decide on starting points or make some computations more efficient for the l1-norm.
* `convex`: boolean that indicates whether `f` is convex.

### Regularizer `phi`

Methods:
* `eval(self, x)`: evaluates the function at `x`.
* `prox(self, x, alpha)`: evaluates the proximal operator of `alpha*phi` at `x`.
* `jacobian_prox(self, x, alpha)`: computes an element of the subdifferential of the proximal operator of `alpha*phi` at `x`.
* `moreau(self, x, alpha)`: evaluates the moreau envelope of `alpha*phi` at `x`.

Note that many common regularizer, e.g the l1/l2-norm or combinations of it, the proximal operator as well as its subdifferential can be computed in closed form.

The package already contains the classes `logistic_loss`, the loss for logistic regression, the squared loss `lsq` and the Student-t loss `tstudent_loss`. As a regularizer, it contains `Norm1` the l1-norm. The definitions for these classes is in `ssnsp/helper/lasso` and `ssnsp/helper/tstudent`.
Note that for optimal performance `f` and `phi` should be Numba jitted classes (see an example how this is done in the files mentioned above).


## First-order methods
The package also contains fast implementations of SAGA [3], SVRG [2] and AdaGrad [1]. These algorithms do not need all of the methods listed above. In general, only `eval` for evaluation and `g` for computing gradients is needed for `f`. For `phi` we only need the `prox` method (and for AdaGrad a `adagrad_prox` method which computes the proximal operator wrt a custom norm).
The implementation of these algorithms is optimized for Numba-jitted function classes.




## References 

* [1] Duchi, J., Hazan, E., and Singer, Y. (2011).  [Adaptive subgradient methods for online learning and stochastic optimization.](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)J. Mach. Learn.Res., 12(null):2121–2159.

* [2] J.  Reddi,  S.,  Sra,  S.,  Poczos,  B.,  and  Smola,  A.  J.  (2016).[Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization.](https://papers.nips.cc/paper/2016/hash/291597a100aadd814d197af4f4bab3a7-Abstract.html)In Lee, D. D., Sugiyama, M., Luxburg, U. V., Guyon, I., and Garnett, R., editors,Advances in Neural Information Processing Systems 29, pages 1145–1153. CurranAssociates, Inc.

* [3] Defazio, A., Bach, F., and Lacoste-Julien, S. (2014).  [Saga:  A fast incremental gradient method with support for non-strongly convex composite objectives.](https://papers.nips.cc/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf) In  Ghahramani,  Z.,  Welling,  M.,  Cortes,  C.,  Lawrence,  N.,  andWeinberger, K. Q., editors,Advances in Neural Information Processing Systems,volume 27, pages 1646–1654. Curran Associates, Inc.
