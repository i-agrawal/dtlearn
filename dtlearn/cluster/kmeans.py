import numpy as np

from .. import Distanced
from ..utils import sqsum


class KMeans(Distanced):

    def kmeanspp(self, x, k):
        assert k > 0, 'k must be greater than 0'

        centroids = x[np.random.randint(x.shape[0])]
        for _ in range(1, k):
            dists = np.min(self.dist(x, centroids), axis=1)
            prob = np.cumsum(dists / np.sum(dists))
            chosen = np.searchsorted(prob > np.random.uniform(), True)
            centroids = np.vstack((centroids, x[chosen]))
        return centroids

    def train(self, x, k):
        next_centers = self.kmeanspp(x, k)
        centroids = next_centers - 1
        while sqsum(centroids - next_centers) > 1e-3:
            centroids = next_centers
            closest = np.argmin(self.dist(x, centroids), axis=1)
            next_centers = np.array([np.mean(x[closest == i], axis=0) for i in range(k)])
        self.centroids = centroids

    def predict(self, x):
        return np.argmin(self.dist(x, self.centroids), axis=1)
