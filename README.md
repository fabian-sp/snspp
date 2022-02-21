# SNSPP

Code associated to A. Milzarek, F. Schaipp, M. Ulbrich, *A semismooth Newton Stochastic Proximal Point method with variance reduction*.
The `SNSPP` method is implemented in `snspp/solver/spp_solver` and references therein. The graphs in the paper can be reproduced with the files in `experiments/`.

## Introduction

This is a Python package for solving problems of the form

<img src="https://latex.codecogs.com/gif.latex?\min_xf(x)+\varphi(x)" title="problem formulation"/>

where the first part of the objective has the special form

<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{N}\sum_{i=1}^{N}f_i(A_ix)" title="f structure"/>

This problem structure is common in statistical learning problems: each summand of `f` is the loss at one data sample and `phi` is a (convex), possibly nonsmooth regularizer.

## Getting started

Install via 

    python setup.py

or in order to install in developer mode via

    python setup.py clean --all develop clean --all


## Functionality

For all solvers of this packages, `f` and `phi` have to be instances of a class. As the solvers need certain information on the involved functions `f` and `phi`, these classes need to have several methods and attributes implemented.

We list the methods and attributes that these objects need to have for the algorithms. This is relevant only if you want to solve problems which functions that are not yet implemented. For a list of implemented functions, see below.

### Loss functions `f`

Methods:
* `eval(x)`: evaluates the function at `x`.
* `f(x, i)`: evaluates `f_i` at `x`. Note that here `x` is typically a scalar.
* `g(x, i)`: evaluates the derivative/gradient of `f_i` at `x`.
* `fstar(x, i)`, `gstar(x, i)` and `Hstar(x, i)`: evaluates the Fenchel conjugate (its gradient/ its Hessian) of `f_i` at `x`.

Note that `fstar` (and `gstar`, `Hstar`) evaluate each sample `i` individually. In many applications, `f_i` is identical for every `i` (up to data input). In this case, the performance is improved if vectorized methods are implemented, i.e. `fstar_vec(x, S)` which computes the conjugate at x for a batch of indices `S`. See `snspp/helper/lasso` for an example. 
The algorithm detects automatically if vectorized methods are implemented, hence the function called for solving is the same in both cases.


Attributes:
* `name`: name of `f`. This is needed e.g. to decide on starting points.
* `convex`: boolean that indicates whether `f` is convex.

### Regularizer `phi`

Methods:
* `eval(x)`: evaluates the function at `x`.
* `prox(x, alpha)`: evaluates the proximal operator of `alpha*phi` at `x`.
* `jacobian_prox(x, alpha)`: computes an element of the subdifferential of the proximal operator of `alpha*phi` at `x`.
* `moreau(x, alpha)`: evaluates the moreau envelope of `alpha*phi` at `x`.

Attributes:
* `name`: name of `phi`. For the l1-norm, some computations are simplified and thus this can be useful.

A detailled documentation of how the above methods are intended is given for `L1Norm` in `snspp/helper/regz`. Note that many common regularizers, e.g the l1/l2-norm or combinations of it, the proximal operator as well as its subdifferential can be computed in closed form.

### Examples

The package already contains the following losses

* `logistic_loss`: the loss for logistic regression.
* `lsq`: the squared loss.
* `tstudent_loss`: loss for regression with Student-t residuals.
* `huber_loss`: the Huber loss function.
* `squared_hinge_loss`: the squared hinge loss.

and regularizers
* `L1Norm`: the l1-norm.
* `Ridge`: the squared l2-norm (known from ridge regression). 
* `Zero`: the constant zero function for unregularized problems.

For examples of loss functions, see `snspp/helper/loss1`, `snspp/helper/loss2` and `snspp/helper/tstudent`. 
For examples of regularizers, see `snspp/helper/regz`.

Note that for optimal performance `f` and `phi` should be [Numba jitted classes](https://numba.pydata.org/numba-doc/dev/user/jitclass.html).


## First-order methods

The package also contains fast implementations of AdaGrad [1], SVRG [2] and SAGA [3]. These algorithms do not need all of the methods listed above. In general, only `eval` for evaluation (which is not actually used for the algorithm) and `g` for computing gradients is needed for `f`. For `phi` we only need the `prox` method (and for AdaGrad a `adagrad_prox` method which computes the proximal operator wrt a custom norm).

**Note:** The implementation of these algorithms is only available for Numba-jitted function classes.


## References 

* [1] J. Duchi, E. Hazan, and  Y. Singer, [Adaptive subgradient methods for online learning and stochastic optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), J. Mach. Learn. Res., 12 (2011), pp. 2121–2159.

* [2] S. J. Reddi, S. Sra, B. Poczos, and A. J. Smola, [Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization](https://papers.nips.cc/paper/2016/hash/291597a100aadd814d197af4f4bab3a7-Abstract.html), in Advances in Neural Information Processing Systems 29, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, eds., Curran Associates, Inc., 2016, pp. 1145–1153.

* [3] A. Defazio, F. Bach, and  S. Lacoste-Julien, [Saga: A fast incremental gradient method with support for non-strongly convex composite objectives](https://papers.nips.cc/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf), in Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q. Weinberger, eds., vol. 27, CurranAssociates, Inc., 2014, pp. 1646–1654.
