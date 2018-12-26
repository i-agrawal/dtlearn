import numpy as np

from .. import Model


class Bernoulli(Model):

    def train(self, x, y):
        self.labels = np.unique(y)
        groups = [x[y == label] for label in self.labels]
        self.priors = np.array([np.mean(group, axis=0) for group in groups])
        self.fracs = np.array([len(group) for group in groups]) / len(y)

    def predict(self, x):
        priors = self.priors[np.newaxis]
        prob = x[:, np.newaxis] * (2 * priors - 1) + (1 - priors)
        chosen = np.argmax(np.prod(prob, axis=2) * self.fracs, axis=1)
        return self.labels[chosen]
