# SNSPP

Code associated with *A semismooth Newton Stochastic Proximal Point algorithm with Variance Reduction* [(see paper)](https://arxiv.org/abs/2204.00406). 

The `SNSPP` method is implemented in [`snspp/solver/spp_solver`](/snspp/solver/spp_solver.py) and references therein.


## Introduction

We aim for solving problems of the form

<img src="https://latex.codecogs.com/gif.latex?\min_xf(x)+\varphi(x)" title="problem formulation"/>

where the first part of the objective has the special form

<img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{N}\sum_{i=1}^{N}f_i(A_ix)" title="f structure"/>

This problem structure is common in statistical learning problems: each summand of `f` is the loss at one data sample and `phi` is a (convex), possibly nonsmooth regularizer. Note that for optimal performance `f` and `phi` should be [Numba jitted classes](https://numba.pydata.org/numba-doc/dev/user/jitclass.html).

## Getting started

Install via 

    python setup.py

or in order to install in developer mode via

    python setup.py clean --all develop clean --all

