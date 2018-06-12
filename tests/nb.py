#!/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import mlimpl as ml

if __name__ == "__main__":
    # load data
    data = np.loadtxt(open("data/absenteeism.data", "rb"), delimiter=",")
    x = np.delete(data, 15, axis=1)
    y = data[:,15]

    # gaussian naive bayes to check if someone is a social drinker
    prior = ml.gnb(x,y)
    pred = ml.gnb_predict(x,prior)
    print("accuracy = %f" % (np.sum(pred==y)/y.shape[0]))
