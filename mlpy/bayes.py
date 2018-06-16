#!/bin/python
from __future__ import division
import numpy as np
import utility as util

# gaussian naive bayes
# given features predict class based on probability
# find mean and variance of each attribute given class
# assume normal and use to get probability
def gnb(x,y):
    x = util.sanitize(x)
    labels = np.unique(y)
    u = np.array([np.mean(x[np.where(y==labels[i])],axis=0) for i in range(labels.shape[0])])
    v = np.array([np.var(x[np.where(y==labels[i])],axis=0) for i in range(labels.shape[0])])
    p = np.array([np.sum(y==labels[i])/y.shape[0] for i in range(labels.shape[0])])
    return [labels, u[:,None], v[:,None], p]

# gaussian naive bayes prediction
# calculate the highest possible class
#      use normal prob for p(x|c)
#      multiple together for each sample
def gnb_predict(x,prior):
    labels,u,v,p = prior
    px = np.exp(-(x-u)**2 / (2*v)) / np.sqrt(2*np.pi*v)
    return labels[np.argmax(np.prod(px,axis=2).T * p, axis=1)]
