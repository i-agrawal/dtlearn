import numpy as np


def manhattan(a, b):
    """
    the manhattan distances between a and b
    the resulting matrix is [m x q]

    :type a: np.ndarray
    :desc a: matrix of samples [m x n]

    :type b: np.ndarray
    :desc b: matrix of samples [q x n]
    """
    dists = a[:, np.newaxis] - b[np.newaxis]
    return np.sum(np.abs(dists), axis=2)


def euclidean(a, b):
    """
    the euclidean distances between a and b
    the resulting matrix is [m x q]

    :type a: np.ndarray
    :desc a: matrix of samples [m x n]

    :type b: np.ndarray
    :desc b: matrix of samples [q x n]
    """
    dists = a[:, np.newaxis] - b[np.newaxis]
    return np.sqrt(np.sum(dists**2, axis=2))


def sqeuclidean(a, b):
    """
    the squared euclidean distances between a and b
    the resulting matrix is [m x q]

    :type a: np.ndarray
    :desc a: matrix of samples [m x n]

    :type b: np.ndarray
    :desc b: matrix of samples [q x n]
    """
    dists = a[:, np.newaxis] - b[np.newaxis]
    return np.sum(dists**2, axis=2)


def minkowski(a, b, p):
    """
    the minkowski distances between a and b
    the resulting matrix is [m x q]

    :type a: np.ndarray
    :desc a: matrix of samples [m x n]

    :type b: np.ndarray
    :desc b: matrix of samples [q x n]

    :type p: float
    :desc p: the power value
    """
    dists = np.abs(a[:, np.newaxis] - b[np.newaxis])
    dists = np.sum(dists ** p, axis=2)
    return dists ** (1/p)


def cosine(a, b):
    """
    the cosine distances between a and b
    the resulting matrix is [m x q]

    :type a: np.ndarray
    :desc a: matrix of samples [m x n]

    :type b: np.ndarray
    :desc b: matrix of samples [q x n]
    """
    return 1.0 - cosine_sim(a, b)


def cosine_sim(a, b):
    """
    the cosine similarities between a and b
    the resulting matrix is [m x q]

    :type a: np.ndarray
    :desc a: matrix of samples [m x n]

    :type b: np.ndarray
    :desc b: matrix of samples [q x n]
    """
    dists = np.dot(a, b.T)
    anorm = np.sqrt(np.sum(a**2, axis=1))
    bnorm = np.sqrt(np.sum(b**2, axis=1))
    return dists / np.outer(anorm, bnorm)
