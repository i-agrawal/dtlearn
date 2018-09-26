import numpy as np

from .. import Model
from ..utils import normal_pdf, fraction_correct


class Gaussian(Model):
    def __init__(self):
        """
        gaussian naive bayes model

        :type labels: np.ndarray
        :desc labels: possible classes [k x 1]

        :type u: np.ndarray
        :desc u: mean of features per class [k x n]

        :type v: np.ndarray
        :desc v: variance of features per class [k x n]

        :type p: np.ndarray
        :desc p: appearance rate of each class [k x 1]
        """

    def train(self, X, y):
        """
        find mean, variance, and appearance of each feature given the class
        """
        self.labels = np.unique(y)
        groups = [X[y == label] for label in self.labels]
        self.u = np.array([np.mean(group, axis=0) for group in groups])
        self.v = np.array([np.var(group, axis=0) for group in groups])
        self.p = np.array([len(group) / len(y) for group in groups])

    def predict(self, X):
        """
        given training mean, variance, and appearance, predict the
        most probable class of each sample in X

        we assume every class is normally distributed so the
        probability of x for each class is the normal pdf

        to see the function look up 'normal distribution' in wikipedia
        """
        X = X[:, np.newaxis]
        u = self.u[np.newaxis]
        v = self.v[np.newaxis]

        prob = normal_pdf(X, u, v)
        prob = np.prod(prob, axis=2) * self.p
        return self.labels[np.argmax(prob, axis=1)]

    def score(self, y, h):
        """
        calculates the number of correct predictions over total predictions
        """
        return fraction_correct(y, h)
