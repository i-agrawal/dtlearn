#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
import mlpy

if __name__ == "__main__":
    # generate data
    # a - b = range of values on x-axis
    # m     = number of samples in training
    # s     = amount of deviation
    a,b,m,s = [0,100,300,12]

    # x = value from a to b
    # y = x with some deviation
    x = np.random.uniform(a,b,m)
    y = x + np.random.normal(0,s,m)
    x = mlpy.add_bias(x)

    # solve linear regression on training data
    w = mlpy.olsr(x,y)

    # display calculated weights on testing
    plt.scatter(x[:,1],y)
    plt.plot([a,b], [w[0]+a*w[1], w[0]+b*w[1]], color="red")
    plt.show()
