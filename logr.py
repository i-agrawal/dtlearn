#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
import mlpy

# get point from x value
def pt(w, x):
    return (0.5 - w[1,0]*x - w[0,0]) / w[2,0]

# main function
if __name__ == "__main__":
    # generate data
    # m = number of samples in training
    # tc = center for true values
    # fc = center for false values
    # s = amount of deviation
    m,tc,fc,s = [400,[4,0],[0,4],1]

    # t = data points around true center
    # f = data points around false center
    # x = all data points stacked together combines
    t = np.random.normal(tc,s,(m,2))
    f = np.random.normal(fc,s,(m,2))
    x = np.vstack((t,f))
    x = mlpy.add_bias(x)

    # y = 1's and 0's (true and false) for each data point
    y = np.ones(2*m)
    y[m:] = np.zeros(m)

    # solve with logistic regression
    w = mlpy.logr(x,y)

    # display plot
    ts = plt.scatter(x[:m,1], x[:m,2])
    fs = plt.scatter(x[m:2*m,1], x[m:2*m,2])
    ax = plt.axis()
    plt.plot([ax[0], ax[1]], [pt(w,ax[0]), pt(w,ax[1])], color='red')
    plt.legend([ts, fs], ['trues','falses'])
    plt.show()
