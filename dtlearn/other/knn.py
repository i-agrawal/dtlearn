import numpy as np

from .. import Distanced


class KNN(Distanced):

    def __mode(self, x):
        return np.argmax(np.bincount(x))

    def train(self, x, y):
        self.x = np.copy(x)
        self.y = np.copy(y)

    def predict(self, x, k, mode='classify'):
        dist = self.dist(x, self.x)
        nearest = np.argpartition(dist, k)[:, :k]
        neighbors = self.y[nearest]

        if mode == 'classify':
            return np.array([self.__mode(row) for row in neighbors])
        elif mode == 'regression':
            return np.mean(neighbors, axis=1)
        raise Exception('knn: unknown mode {}'.format(mode))
