import numpy as np

from .kmeans import KMeans
from ..utils import sqsum


class FuzzyCMeans(KMeans):

    def __weights(self, x, centers, m):
        dist = self.dist(x, centers)
        dist = np.maximum(dist, 1e-6)
        bottom = dist / np.sum(dist, axis=1)[:, np.newaxis]
        return 1. / np.power(bottom, 1 / (m-1))

    def train(self, x, c, m=2):
        next_centers = self.kmeanspp(x, c)
        centers = next_centers - 1
        while sqsum(centers - next_centers) > 1e-3:
            centers = next_centers
            weights = np.power(self.__weights(x, centers, m), m)
            next_centers = weights.T @ x / np.sum(weights, axis=0)[:, np.newaxis]
        self.centers = centers

    def predict(self, x, m=2):
        return np.argmax(self.__weights(x, self.centers, m), axis=1)
