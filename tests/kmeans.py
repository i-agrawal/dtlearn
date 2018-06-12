#!/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import mlimpl as ml

if __name__ == "__main__":
    # load data
    x = np.loadtxt(open("data/iris.data", "rb"), delimiter=",")[:,:4]

    # calculate the centroids
    centroids = ml.kmeans(x,3)

    # plot assignment
    pred = ml.kmeans_predict(x,centroids)
    ind = np.where(pred==0)
    plt.scatter(x[ind, 0], x[ind, 1])
    ind = np.where(pred==1)
    plt.scatter(x[ind, 0], x[ind, 1])
    ind = np.where(pred==2)
    plt.scatter(x[ind, 0], x[ind, 1])
    plt.show()
