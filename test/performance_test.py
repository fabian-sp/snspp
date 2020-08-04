#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:35:19 2020

@author: fabian
"""


import numpy as np

n = 5000

np.random.seed(123)

A = np.random.randn(n,n)
d = np.random.randn(n)

def plus(A, d):
    
    return A + np.diag(d)

def plus_diag(A, d):
    
    d2 = d + np.diag(A)
    np.fill_diagonal(A, d2)
    return A

%timeit plus(A,d)

%timeit plus_diag(A,d)
