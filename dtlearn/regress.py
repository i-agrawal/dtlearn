import numpy as np

from . import Model
from .utils import add_bias, sigmoid
from .utils import fraction_correct, coef_determination


class Linear(Model):
    def __init__(self):
        """
        linear regression model

        :type theta: np.ndarray
        :desc theta: learned weight matrix [n+1 x k]
        """

    def train(self, X, y):
        """
        calculate weights that best solve y = a1 * x1 + ... + an * xn + b
        """
        X = add_bias(X)
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        """
        use trained weights to predict the value of each sample
        """
        X = add_bias(X)
        return np.dot(X, self.theta)

    def score(self, y, h):
        """
        calculates the coefficient of determination
        """
        return coef_determination(y, h)


class Logistic(Model):
    def __init__(self):
        """
        logistic regression model

        :type theta: np.ndarray
        :desc theta: learned weight matrix [n+1 x k]
        """

    def train(self, X, y, epochs=10000, eta=0.1, exit=0.00001, seed=34):
        """
        calculate weights that best solve sigmoid(a1 * x1 + ... + an * xn + b) == k
        """
        m, n = X.shape
        X = add_bias(X)

        self.labels, encode = np.unique(y, return_inverse=True)
        k = len(self.labels)
        y = np.eye(k)[encode]

        # NOTE: using gradient descent for this step
        np.random.seed(seed)
        theta = np.random.rand(n+1, k)
        for _ in range(epochs):
            h = sigmoid(X.dot(theta))
            grad = eta * X.T.dot(y-h) / m
            theta += grad
            if np.sum(grad**2) < exit:
                break
        self.theta = theta

    def predict(self, X):
        """
        use trained weights to predict the class of each sample
        """
        X = add_bias(X)
        prob = sigmoid(X.dot(self.theta))
        return self.labels[np.argmax(prob, axis=1)]

    def score(self, y, h):
        """
        calculates the number of correct predictions over total predictions
        """
        return fraction_correct(y, h)
