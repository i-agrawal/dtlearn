#!/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import mlimpl as ml

if __name__ == "__main__":
    # load data
    data = np.loadtxt(open("data/iris.data", "rb"), delimiter=",")
    x = np.ones(data.shape)
    x[:,1:] = data[:,:-1]
    y = 2*(data[:,-1]>0)-1

    w = ml.svm(x,y)
    print(np.sum((ml.svm_pred(x,w)==y)) / y.shape[0])
