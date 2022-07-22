# SNSPP

Code associated with *A semismooth Newton Stochastic Proximal Point algorithm with Variance Reduction*. 

The `SNSPP` method is implemented in [`snspp/solver/spp_solver`](/snspp/solver/spp_solver.py) and references therein.


## Introduction

This is a Python package for solving problems of the form

<img src="https://latex.codecogs.com/gif.latex?\min_xf(x)+\varphi(x)" title="problem formulation"/>

where the first part of the objective has the special form

<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{N}\sum_{i=1}^{N}f_i(A_ix)" title="f structure"/>

This problem structure is common in statistical learning problems: each summand of `f` is the loss at one data sample and `phi` is a (convex), possibly nonsmooth regularizer. Note that for optimal performance `f` and `phi` should be [Numba jitted classes](https://numba.pydata.org/numba-doc/dev/user/jitclass.html).

## Getting started

Install via 

    python setup.py

or in order to install in developer mode via

    python setup.py clean --all develop clean --all



## First-order methods

The package also contains implementations of AdaGrad [1], SVRG [2] and SAGA [3]. 

## References 

* [1] J. Duchi, E. Hazan, and  Y. Singer, [Adaptive subgradient methods for online learning and stochastic optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), J. Mach. Learn. Res., 12 (2011), pp. 2121–2159.

* [2] S. J. Reddi, S. Sra, B. Poczos, and A. J. Smola, [Proximal stochastic methods for nonsmooth nonconvex finite-sum optimization](https://papers.nips.cc/paper/2016/hash/291597a100aadd814d197af4f4bab3a7-Abstract.html), in Advances in Neural Information Processing Systems 29, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, eds., Curran Associates, Inc., 2016, pp. 1145–1153.

* [3] A. Defazio, F. Bach, and  S. Lacoste-Julien, [Saga: A fast incremental gradient method with support for non-strongly convex composite objectives](https://papers.nips.cc/paper/2014/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Paper.pdf), in Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q. Weinberger, eds., vol. 27, CurranAssociates, Inc., 2014, pp. 1646–1654.
