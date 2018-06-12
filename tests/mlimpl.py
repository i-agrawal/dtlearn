#!/bin/python
from __future__ import division
import numpy as np

#------------------- regression begin -------------------
# least squares regression
# cost is determined by ordinary least squares regression
# find derivative and find w that make it equal to 0
# result is w = inv(x'x)x'y
def lsreg(x, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)

# logistic regression with gradient descent
# set w to random weights
# change g by derivative of the cost function with respect to w
# if weight change is small, break
def logreg(x, y, alpha=0.01, brk=0.00001, max_iter=10000):
    w = np.random.rand(x.shape[1], y.shape[1])
    for i in range(max_iter):
        h = 1 / (1 + np.exp(-np.dot(x,w)))
        g = alpha * np.dot(x.T, y-h) / x.shape[0]
        w = w + g
        if np.sum(g**2) < brk:
            break
    return w
#-------------------  regression end  -------------------


#------------------- instance begin -------------------
# implementation of knn
# for each point
#     find k closest points
#     assign the most frequent class of the k points
def knn(x, y, k):
    dist = np.sum((x - x[:,None])**2, axis=2)
    part = np.argpartition(dist, k+1)[:,:k+1]
    votes = y[np.array([np.delete(part[i], np.where(part[i]==i)) for i in range(x.shape[0])])]
    return np.array([np.bincount(votes[i]).argmax() for i in range(x.shape[0])])
#-------------------  instance end  -------------------


#------------------- bayesian begin -------------------
# implementation of gaussian naive bayes
# given features predict class based on probability
# find mean and variance of each attribute given class
# assume normal and use to get probability
def gnb(x,y):
    labels = np.unique(y)
    u = np.array([np.mean(x[np.where(y==labels[i])],axis=0) for i in range(labels.shape[0])])
    v = np.array([np.var(x[np.where(y==labels[i])],axis=0) for i in range(labels.shape[0])])
    p = np.array([np.sum(y==labels[i])/y.shape[0] for i in range(labels.shape[0])])
    return [labels, u[:,None], v[:,None], p]

# implementation of gaussin naive bayes prediction
# calculate the highest possible class
#      use normal prob for p(x|c)
#      multiple together for each sample
def gnb_predict(x,prior):
    labels,u,v,p = prior
    px = np.exp(-(x-u)**2 / (2*v)) / np.sqrt(2*np.pi*v)
    return labels[np.argmax(np.prod(px,axis=2).T * p, axis=1)]


#-------------------  bayesian end  -------------------


#------------------- clustering begin -------------------
# implementation of kmeanspp
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

# implementation of kmeans prediction
# given k centroids tell the closest of x
def kmeans_predict(x,centroids):
    return np.argmin(np.sum((centroids - x[:,None])**2, axis=2), axis=1)

# implementation of kmeans
# given k centroids
# assign each point the closest centroid
# move centroid to mean of all points assigned to it
# keep going until centroids dont change
def kmeans(x,k):
    centroids = kmeanspp(x,k)
    while 1:
        elect = np.argmin(np.sum((centroids - x[:,None])**2, axis=2), axis=1)
        change = np.array([x[elect==i].mean(axis=0) for i in range(k)])
        if (centroids==change).all():
            break;
        centroids = change
    return centroids
#-------------------  clustering end  -------------------
