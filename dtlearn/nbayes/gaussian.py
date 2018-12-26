import numpy as np

from .. import Model


class Gaussian(Model):

    def train(self, x, y):
        self.labels = np.unique(y)
        groups = [x[y == label] for label in self.labels]
        self.means = np.array([np.mean(group, axis=0) for group in groups])
        self.varis = np.array([np.var(group, axis=0) for group in groups])
        self.fracs = np.array([len(group) for group in groups]) / len(y)

    def predict(self, x):
        diff = x[:, np.newaxis] - self.means[np.newaxis]
        prob = np.exp(-np.multiply(diff, diff) / (2 * self.varis))
        chosen = np.argmax(np.prod(prob, axis=2) * self.fracs, axis=1)
        return self.labels[chosen]
