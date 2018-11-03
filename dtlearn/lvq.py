import numpy as np

from . import Model
from .utils import fraction_correct


class Classifier(Model):
    def __init__(self):
        """
        learning vector quantization classifier model

        :type theta: np.ndarray
        :desc theta: learned weight matrix [n x k]
        """

    def train(self, X, y, eta, epochs=10):
        """
        for each sample in x find the closest weight and update it

        :type eta: float
        :desc eta: learning rate
        """
        m, n = X.shape
        self.labels, encode = np.unique(y, return_inverse=True)

        theta = np.array([X[y == label][0] for label in self.labels])
        for i in range(epochs):
            eta_ = eta * (1.0 - i / epochs)
            for x, label in zip(X, encode):
                dist = np.sum((theta - x)**2, axis=1)
                ind = np.argmin(dist)
                change = eta_ * (x - theta[ind])
                if ind == label:
                    theta[ind] += change
                else:
                    theta[ind] -= change
        self.theta = theta

    def predict(self, X):
        """
        use trained weights to predict the class of each sample
        """
        dists = X[:, np.newaxis] - self.theta[np.newaxis]
        dists = np.sum(dists**2, axis=2)
        return self.labels[np.argmin(dists, axis=1)]

    def score(self, y, h):
        """
        calculates the number of correct predictions over total predictions
        """
        return fraction_correct(y, h)
