import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def sqsum(x, axis=None):
    return np.sum(np.multiply(x, x), axis=axis)


def correlation(y, h):
    return 1 - sqsum(y - h) / sqsum(y - np.mean(y))


def purity(y, h):
    cont = contingency_matrix(y, h)
    return np.sum(np.max(cont, axis=0)) / np.sum(cont)


def biased(x):
    m, n = x.shape
    xb = np.ones((m, n+1))
    xb[:, :n] = x
    return xb
