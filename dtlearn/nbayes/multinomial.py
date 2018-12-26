import numpy as np

from .. import Model


class Multinomial(Model):

    def train(self, x, y):
        self.categories, encode = np.unique(x, return_inverse=True)
        encode = encode.reshape(x.shape)
        x = np.eye(len(self.categories))[encode]

        self.labels = np.unique(y)
        groups = [x[y == label] for label in self.labels]
        self.priors = np.array([np.mean(group, axis=0) for group in groups])
        self.fracs = np.array([len(group) for group in groups]) / len(y)

    def predict(self, x):
        mapping = {c: i for i, c in enumerate(self.categories)}
        encode = np.vectorize(mapping.get)(x)
        x = np.eye(len(self.categories))[encode]

        prob = np.sum(x[:, np.newaxis] * self.priors[np.newaxis], axis=3)
        chosen = np.argmax(np.prod(prob, axis=2) * self.fracs, axis=1)
        return self.labels[chosen]
