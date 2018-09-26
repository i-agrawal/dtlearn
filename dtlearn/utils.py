import numpy as np


def add_bias(x):
    """
    add a column of ones to the right of a matrix
    """
    m = x.shape[0]
    return np.append(x, np.ones((m, 1)), axis=1)


def normal_pdf(x, u, v):
    """
    find the probability of a value (x) given the mean (u) and variance (v)

    to see the function look up 'normal distribution' in wikipedia
    """
    return np.exp(-(x - u)**2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def sigmoid(x):
    """
    find the sigmoid of x

    to see the function look up 'sigmoid function' in wikipedia
    """
    return 1.0 / (1.0 + np.exp(-x))


def fraction_correct(y, h):
    """
    calculates the number of correct predictions over total predictions
    """
    return np.sum(y == h) / len(y)


def coef_determination(y, h):
    """
    calculates the coefficient of determination

    to see the function look up 'coefficient of determination' in wikipedia
    """
    dist = np.sum((y - h)**2)
    variance = np.sum((y - np.mean(y))**2)
    return 1.0 - dist / variance
