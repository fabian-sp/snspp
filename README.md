# stochastic-ssnal

A Python package for solving problems of the form

<img src="https://latex.codecogs.com/gif.latex?\min_xf(x)+\varphi(x)" title="problem formulation"/>

where the first part of the objective has the special form

<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{N}\sum_{i=1}^{N}f_i(A_ix)" title="f structure"/>

This problem structure is common in machine learning, where each summand of `f` is one data sample and `phi` is a regularizer in order to avoid overfitting.

The main contribution of this package is the implementation of the stochastic SSNAL algorithm. As a side product, we provide fast implementaions of SAGA and AdaGrad, two widely known first-order methods.
