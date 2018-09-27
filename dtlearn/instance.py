import numpy as np
from scipy import stats

from . import Model
from .utils import fraction_correct, coef_determination


class KNN(Model):
    def __init__(self, mode):
        """
        k nearest neighbors model for classification

        :type k: int
        :desc k: number of neighbors to check

        :type mode: str
        :desc mode: either 'classifier' or 'regressor'
        """
        if mode == 'classifier':
            self.handle = self.classify
            self.scorer = fraction_correct
        elif mode == 'regressor':
            self.handle = self.regress
            self.scorer = coef_determination

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
        return self.handle(self.y[nearest])

    @staticmethod
    def classify(neighbors):
        """
        find the most frequent class (i.e. mode) from neighbors
        """
        best = stats.mode(neighbors, axis=1)[0]
        return np.ravel(best)

    @staticmethod
    def regress(neighbors):
        """
        find the average of the neighbors' values
        """
        return np.mean(neighbors, axis=1)

    def score(self, y, h):
        """
        depends on the type of knn
        """
        return self.scorer(y, h)
