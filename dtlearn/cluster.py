from collections import deque

import numpy as np

from . import Model
from .utils import cluster_intersect
from .distance import sqeuclidean


class KMeans(Model):
    def __init__(self, k, dist=sqeuclidean):
        """
        k means clustering model

        :type k: int
        :desc k: number of clusters

        :type centroids: np.ndarray
        :desc centroids: the center for each cluster [k x n]

        :type dist: fct(np.ndarray, np.ndarray)
        :desc dist: the distance metric used
        """
        self.k = k
        self.dist = dist

    def kmeanspp(self, X):
        """
        find k initial centers from the given sample points
        each point has a higher probability of being chosen the
        further away it is from the already chosen centers
        """
        centroids = X[np.random.randint(X.shape[0])]
        for i in range(1, self.k):
            dists = np.min(self.dist(X, centroids), axis=1)
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
            elect = np.argmin(self.dist(X, centroids), axis=1)
            change = np.array([np.mean(X[elect == i], axis=0) for i in range(self.k)])
        self.centroids = centroids

    def predict(self, X):
        """
        assign each sample in X to it's closest centroid
        """
        return np.argmin(self.dist(X, self.centroids), axis=1)

    def score(self, y, h):
        """
        assign each cluster to most intersecting label and find the percentage correct
        """
        return cluster_intersect(y, h)


class DBSCAN(Model):
    UNDEFINED = -1

    def __init__(self):
        """
        DBSCAN algorithm
        """

    def train(self):
        raise Exception('DBSCAN does not need training')

    def predict(self, X, min_pts, eps, dist=sqeuclidean):
        """
        a core point is a point with at least min_pts within eps distance of it
        for each core point not in a cluster, start a cluster with the point
        for each point in the cluster add its neighbors if it is a core point
            keep doing that until no more new points can be added

        :type min_pts: int
        :desc min_pts: minimum number of neighbors to be a core point

        :type eps: float
        :desc eps: distance for two points to be within the same clusters

        :type dist: fct(np.ndarray, np.ndarray)
        :desc dist: the distance metric used
        """
        m, n = X.shape

        neighbors = [set() for _ in range(m)]
        rows, cols = np.where(dist(X, X) <= eps)
        for i, j in zip(rows, cols):
            neighbors[i].add(j)

        cluster = 0
        labels = [DBSCAN.UNDEFINED] * m
        for i in range(m):
            if labels[i] != DBSCAN.UNDEFINED or len(neighbors[i]) < min_pts:
                continue

            searched = set([i])
            searching = deque([i])
            while searching:
                pt = searching.popleft()
                if labels[pt] == DBSCAN.UNDEFINED:
                    labels[pt] = cluster
                    if len(neighbors[pt]) >= min_pts:
                        for x in neighbors[pt]:
                            if x not in searched:
                                searching.append(x)
                                searched.add(x)
            cluster += 1
        return np.array(labels)

    def score(self, y, h):
        """
        assign each cluster to most intersecting label and find the percentage correct
        """
        return cluster_intersect(y, h)
