import numpy as np

from . import Model
from .utils import add_bias, two_dims
from .utils import fraction_correct


class Classifier(Model):
    def __init__(self):
        """
        support vector machine classifier model

        :type theta: np.ndarray
        :desc theta: learned weight matrix [n+1 x k]
        """

    def train(self, X, y, epochs=1000, eta=1, exit=0.00001, seed=34):
        """
        calculate weights that best solve a1 * x1 + ... + an * xn > 0 == k
        """
        m, n = X.shape
        X = add_bias(X)
        y = 2 * two_dims(y) - 1

        # NOTE: using gradient descent for this step
        np.random.seed(seed)
        theta = np.random.rand(n+1, 1)
        for _ in range(epochs):
            grad = eta * (-np.dot(X.T, y) + theta) / m
            theta -= grad
            if np.sum(grad**2) < exit:
                break
        self.theta = theta

    def predict(self, X):
        """
        use trained weights to predict the class of each sample
        """
        X = add_bias(X)
        return np.dot(X, self.theta) > 0

    def score(self, y, h):
        """
        calculates the number of correct predictions over total predictions
        """
        return fraction_correct(y, h)
