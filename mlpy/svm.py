#!/bin/python
from __future__ import division
import numpy as np
import utility as util

# support vector machine
#
# cost = max(0, 1 - y*x*w)
# dc/dw = -x'*y
def svm(x,y,eta=0.01,brk=0.00001,max_iter=10000):
    x = util.sanitize(x)
    y = util.sanitize(y)
    w = np.random.rand(x.shape[1], y.shape[1])
    for i in range(max_iter):
        h = np.dot(x,w)*y
        g = eta * np.dot(x.T, y*(h<1)) / x.shape[0]
        w = w + g
        if np.sum(g**2) < brk:
            break
    return w

# support vector machine predict
# given weights calculate predicted results on input
def svm_pred(x,w):
    return 2*(np.dot(x,w)>0)-1
