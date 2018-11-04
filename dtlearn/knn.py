import numpy as np
from scipy import stats

from . import Model
from .utils import fraction_correct, coef_determination


class Classifier(Model):
    def __init__(self):
        """
        k nearest neighbors model for classification

        :type k: int
        :desc k: number of neighbors to check
        """

    def train(self, X, y):
        """
        use the points given to judge the class of
        each test point in predict
        """
        self.X = np.copy(X)
        self.y = np.copy(y)

    def predict(self, X, k):
        """
        assign each sample in X to the most occurring
        class from the sample's k nearest neighbors
        """
        dists = X[:, np.newaxis] - self.X[np.newaxis]
        dists = np.sum(dists**2, axis=2)
        nearest = np.argpartition(dists, k)[:, :k]
        neighbors = self.y[nearest]
        best = stats.mode(neighbors, axis=1)[0]
        return np.ravel(best)

    def score(self, y, h):
        """
        returns the fraction of guesses in h that are correct
        """
        return fraction_correct(y, h)


class Regressor(Model):
    def __init__(self):
        """
        k nearest neighbors model for regression

        :type k: int
        :desc k: number of neighbors to check
        """

    def train(self, X, y):
        """
        use the points given to judge the value of
        each test point in predict
        """
        self.X = np.copy(X)
        self.y = np.copy(y)

    def predict(self, X, k):
        """
        assign each sample in X to the average of
        values from the sample's k nearest neighbors
        """
        dists = X[:, np.newaxis] - self.X[np.newaxis]
        dists = np.sum(dists**2, axis=2)
        nearest = np.argpartition(dists, k)[:, :k]
        neighbors = self.y[nearest]
        return np.mean(neighbors, axis=1)

    def score(self, y, h):
        """
        returns the coefficient of determination
        """
        return coef_determination(y, h)
