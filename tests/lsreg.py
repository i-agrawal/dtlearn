#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
import mlimpl as ml

def gen_data(m, sigma):
    # create mats, bias already 1
    x = np.ones((m, 2))
    y = np.ones((m, 1))
    for i in xrange(0,m):
        # choose random point from 0 to 100
        x[i, 1] = np.random.uniform(0, 100)
        # choose y to be norm(x,std)
        y[i, 0] = np.random.normal(x[i, 1], sigma)
    return x,y


if __name__ == "__main__":
    # data generation
    # create data around y = x
    sigma = input("standard deviation size: ") # deviation in values from y = x
    x,y = gen_data(700,sigma)
    xtest,ytest = gen_data(200,sigma)

    # solve linear regression
    # inv(x'x)x'y
    w = ml.lsreg(x, y)

    # display plot
    plt.scatter(xtest[:, 1], ytest)
    plt.plot([0, 100], [w[0], w[0]+100*w[1]], color="red")
    plt.show()
