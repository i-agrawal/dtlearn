#!/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import mlimpl as ml

if __name__ == "__main__":
    # load data
    data = np.loadtxt(open("data/iris.data", "rb"), delimiter=",")
    x = data[:,:-1]
    y = data[:,-1]

    # gaussian naive bayes to check if someone is a social drinker
    prior = ml.gnb(x,y)
    pred = ml.gnb_predict(x,prior)
    print("accuracy = %f" % (np.sum(pred==y)/y.shape[0]))
