import numpy as np

from .. import Model
from ..utils import sqsum


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class Logistic(Model):

    def __gradient(self, x, y):
        m = x.shape[0]
        diff = sigmoid(x @ self.w + self.b) - y

        dw = x.T @ diff
        db = np.sum(diff, axis=0)
        return dw / m, db / m

    def train(self, x, y, eta=0.01, iterations=10000):
        m, n = x.shape
        _, k = y.shape

        std = 1. / np.sqrt(n)
        self.w = np.random.uniform(-std, std, (n, k))
        self.b = np.random.uniform(-std, std, (1, k))

        for i in range(iterations):
            dw, db = self.__gradient(x, y)
            self.w -= eta * dw
            self.b -= eta * db
            if sqsum(dw) < 1e-3:
                break

    def predict(self, x):
        h = sigmoid(x @ self.w + self.b)
        return np.argmax(h, axis=1)
