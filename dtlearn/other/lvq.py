import numpy as np

from .. import Distanced


class LVQ(Distanced):

    def train(self, x, y, eta=0.1, epochs=10):
        m = x.shape[0]
        labels = np.unique(y)

        w = np.array([x[y == label][0] for label in labels])
        for i in range(epochs):
            eta_ = eta * (1. - i / epochs)
            for pt, actual in zip(x, y):
                chosen = np.argmin(self.dist(w, pt))
                sign = 2 * (labels[chosen] == actual) - 1
                w[chosen] += sign * eta_ * (pt - w[chosen])
        self.w = w
        self.labels = labels

    def predict(self, x):
        chosen = np.argmin(self.dist(x, self.w), axis=1)
        return self.labels[chosen]
