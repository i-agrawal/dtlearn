#!/bin/python
from __future__ import division
import numpy as np

# make sure data is in 2d form
# check if it is 1, add second axis with length of 1
def sanitize(data):
    m = data.shape[0]
    if data.ndim == 1:
        data = np.reshape(data, (m,1))
    return data

# add bias to some data
# add a left most column of 1's
def add_bias(data):
    data = sanitize(data)
    m,n = data.shape
    ones = np.ones((m, n+1))
    ones[:,1:] = data
    return ones
