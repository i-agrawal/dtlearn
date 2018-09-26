import numpy as np
from scipy.optimize import linear_sum_assignment

from . import Model


class KMeans(Model):
    def __init__(self, k):
        """
        k means clustering model

        :type k: int
        :desc k: number of clusters

        :type centroids: np.ndarray
        :desc centroids: the center for each cluster [k x n]
        """
        self.k = k

    def kmeanspp(self, X):
        """
        find k initial centers from the given sample points
        each point has a higher probability of being chosen the
        further away it is from the already chosen centers
        """
        centroids = X[np.random.randint(X.shape[0])]
        for i in range(1, self.k):
            dists = X[:, np.newaxis] - centroids[np.newaxis]
            dists = np.min(np.sum(dists**2, axis=2), axis=1)
            prob = np.cumsum(dists / np.sum(dists))
            chosen = np.searchsorted(prob > np.random.uniform(), True)
            centroids = np.vstack((centroids, X[chosen]))
        return centroids

    def train(self, X):
        """
        find the best centers for k clusters where each point
        is assigned to its closest cluster center
        """
        change = self.kmeanspp(X)
        centroids = change - 1
        while np.any(centroids != change):
            centroids = change
            dists = X[:, np.newaxis] - centroids[np.newaxis]
            elect = np.argmin(np.sum(dists**2, axis=2), axis=1)
            change = np.array([np.mean(X[elect == i], axis=0) for i in range(self.k)])
        self.centroids = centroids

    def predict(self, X):
        """
        assign each sample in X to it's closest centroid
        """
        dists = X[:, np.newaxis] - self.centroids[np.newaxis]
        elect = np.argmin(np.sum(dists**2, axis=2), axis=1)
        return elect

    def score(self, y, h):
        """
        assign each centroid to a label and find the percentage correct
        """
        labels, counts = np.unique(y, return_counts=True)
        guesses = [y[h == i] for i in range(self.k)]

        matches = np.array([[np.sum(guess == label) for label in labels] for guess in guesses])
        master = matches / counts[:, np.newaxis]
        rows, cols = linear_sum_assignment(-master)
        return np.sum([master[r, c] * len(guesses[r]) for r, c in zip(rows, cols)]) / len(y)
