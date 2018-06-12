#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math
import mlimpl as ml

# generate data
def gen_data(tc, fc, m, s=2):
    v = [[s,0],[0,s]]
    # generate true sample around tc
    xt = np.random.multivariate_normal(tc,v,m)

    # generate false sample around tc
    xf = np.random.multivariate_normal(fc,v,m)

    # add the bias
    x = np.vstack((xt,xf))
    x = np.hstack((np.ones((2*m,1)),x))

    # generate trues and falses
    y = np.vstack((np.ones((m,1)), np.zeros((m,1)))).astype(np.float64)
    return x,y

# get point from x value
def pt(w, x):
    return (0.5 - w[1,0]*x - w[0,0]) / w[2,0]

# main function
if __name__ == "__main__":
    # generate data
    x,y = gen_data([4,0],[0,4],1000)
    xtest,ytest = gen_data([4,0],[0,4],100)

    # solve logistic regression
    w = ml.logreg(x,y)

    # display plot
    ts = plt.scatter(xtest[0:100,1], xtest[0:100,2])
    fs = plt.scatter(xtest[100:200,1], xtest[100:200,2])
    ax = plt.axis()
    plt.plot([ax[0], ax[1]], [pt(w,ax[0]), pt(w,ax[1])], color='red')
    plt.legend([ts, fs], ['trues','falses'])
    plt.show()
