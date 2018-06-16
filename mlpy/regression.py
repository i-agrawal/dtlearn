#!/bin/python
from __future__ import division
import numpy as np
import utility as util

# ordinary least squares regression
#     cost is determined by ordinary least squares: (y - h)^2
#     find derivative and find w that make it equal to 0
#     result is w = inv(x'x)x'y
def olsr(x, y):
    x = util.sanitize(x)
    y = util.sanitize(y)
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

# logistic regression with batch gradient descent
#     activation function is sigmoid
#     cost is determined by log likelihood: -y*log(h) - (1-y)*log(1-h)
#     gradient descent -
#         set w to random weights
#         get derivative of cost function: (y-h)*x
#         the gradient is learning rate * derivative
#         if gradient is small stop, otherwise add gradient to w
def logr(x, y, eta=0.01, brk=0.00001, max_iter=10000):
    x = util.sanitize(x)
    y = util.sanitize(y)
    w = np.random.rand(x.shape[1], y.shape[1])
    for i in range(max_iter):
        h = 1 / (1 + np.exp(-np.dot(x,w)))
        g = eta * np.dot(x.T, y-h) / x.shape[0]
        w = w + g
        if np.sum(g**2) < brk:
            return w
    return w
#-------------------  regression end  -------------------
