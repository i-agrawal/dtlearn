#!/bin/python
from __future__ import division
import numpy as np
import utility as util

# kmeans++
# choose random point as 1st centroid
# for next k-1 points
#     give each point prob = min_dist / total_dist
#     use random number to choose point
def kmeanspp(x,k):
    centroids = x[np.random.randint(x.shape[0])];
    for i in range(1,k):
        dist = np.min(np.sum((centroids - x[:,None])**2, axis=2), axis=1)
        index = (np.cumsum(dist / np.sum(dist)) > np.random.uniform()).searchsorted(True)
        centroids = np.vstack([centroids, x[index]])
    return centroids

# kmeans prediction
# given k centroids tell the closest of x
def kmeans_predict(x,centroids):
    return np.argmin(np.sum((centroids - x[:,None])**2, axis=2), axis=1)

# kmeans
# given k centroids
# assign each point the closest centroid
# move centroid to mean of all points assigned to it
# keep going until centroids dont change
def kmeans(x,k):
    x = util.sanitize(x)
    centroids = kmeanspp(x,k)
    while 1:
        elect = np.argmin(np.sum((centroids - x[:,None])**2, axis=2), axis=1)
        change = np.array([x[elect==i].mean(axis=0) for i in range(k)])
        if (centroids==change).all():
            break;
        centroids = change
    return centroids
