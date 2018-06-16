#!/bin/python
from __future__ import division
import numpy as np
import utility as util

# k nearest neighbors
# for each point
#     find k closest points
#     assign the most frequent class of the k points
def knn(x, y, k):
    x = util.sanitize(x)
    dist = np.sum((x - x[:,None])**2, axis=2)
    part = np.argpartition(dist, k+1)[:,:k+1]
    votes = y[np.array([np.delete(part[i], np.where(part[i]==i)) for i in range(x.shape[0])])]
    return np.array([np.bincount(votes[i]).argmax() for i in range(x.shape[0])])
