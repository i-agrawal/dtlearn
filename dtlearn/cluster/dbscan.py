from collections import deque

import numpy as np

from .. import Distanced


class DBSCAN(Distanced):

    def train(self, *args, **kwargs):
        raise Exception('dbscan does not have training')

    def predict(self, x, points, eps):
        m, n = x.shape

        neighbors = [set() for _ in range(m)]
        rows, cols = np.where(self.dist(x, x) <= eps)
        for i, j in zip(rows, cols):
            neighbors[i].add(j)

        cluster = 0
        labels = [-1] * m
        for i in range(m):
            if labels[i] != -1 or len(neighbors[i]) < points:
                continue

            searching = deque([i])
            while searching:
                pt = searching.popleft()
                if labels[pt] == -1:
                    labels[pt] = cluster
                    if len(neighbors[pt]) >= points:
                        searching.extend(neighbors[pt])
            cluster += 1
        return np.array(labels)
