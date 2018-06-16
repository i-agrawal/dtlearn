#!/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import mlpy

if __name__ == "__main__":
    # load data
    data = np.loadtxt(open("data/iris.data", "rb"), delimiter=",")
    x = data[:,:-1]
    y = data[:,-1]

    # gaussian naive bayes to check iris plant class
    prior = mlpy.gnb(x,y)
    pred = mlpy.gnb_predict(x,prior)
    print("accuracy = %f" % (np.sum(pred==y)/y.shape[0]))
