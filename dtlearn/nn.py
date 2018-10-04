import abc

import numpy as np

from . import Model
from .utils import sigmoid, two_dims
from .utils import fraction_correct, coef_determination


class NeuralNetwork(Model, metaclass=abc.ABCMeta):
    def __init__(self, hidden):
        """
        neural network model

        :type hidden: List[int]
        :desc hidden: number of neurons in each hidden layer [h x 1]

        :type thetas: List[np.ndarray]
        :desc thetas: learned weight matrix between each layer [h+1 x 1]

        :type biases: List[np.ndarray]
        :desc biases: learned bias between each layer [h+1 x 1]
        """
        self.hidden = hidden

    def forward(self, a, thetas, biases):
        """
        feed forward process of the neural network calculating the input
        and output of each layer given the input to the first layer
        """
        acts, hyps = [], []
        for theta, b in zip(thetas, biases):
            acts.append(a)
            h = np.dot(a, theta) + b
            hyps.append(h)
            a = self.activate(h)
        return a, acts, hyps

    def train(self, X, y, epochs, eta, seed):
        """
        calculate the weights to best make the feed forward of X close to y
        """
        m, n = X.shape
        k = y.shape[1]
        np.random.seed(seed)

        hidden = [n] + self.hidden + [k]
        thetas = [np.random.randn(a, b) for a, b in zip(hidden[:-1], hidden[1:])]
        biases = [np.random.randn(1, b) for b in hidden[1:]]

        # NOTE: using gradient descent for this step
        for _ in range(epochs):
            a, acts, hyps = self.forward(X, thetas, biases)

            sigma = a - y
            before = np.array([])
            for i in range(len(acts)-1, -1, -1):
                a, h = acts[i], hyps[i]
                if before.size > 0:
                    sigma = np.dot(sigma, before.T)
                before = thetas[i].copy()
                sigma *= self.deactivate(h)

                biases[i] -= eta * np.mean(sigma, axis=0)
                thetas[i] -= eta * np.dot(a.T, sigma) / m

        self.thetas = thetas
        self.biases = biases

    @abc.abstractmethod
    def activate(self, x):
        """
        activation function after each calculation in a layer
        """

    @abc.abstractmethod
    def deactivate(self, x):
        """
        the derivative of the activation function
        """


class Classifier(NeuralNetwork):
    """
    logistic neural network model

    :type theta: np.ndarray
    :desc theta: learned weight matrix [n+1 x k]
    """

    def train(self, X, y, epochs=1000, eta=1, seed=34):
        """
        train the classifier network after one hot
        encoding the target values
        """
        self.labels, encode = np.unique(y, return_inverse=True)
        k = len(self.labels)
        y = np.eye(k)[encode]

        super().train(X, y, epochs, eta, seed)

    def activate(self, x):
        """
        the classifier network's activation is sigmoid
        """
        return sigmoid(x)

    def deactivate(self, x):
        """
        the derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        """
        x = sigmoid(x)
        return x * (1 - x)

    def predict(self, X):
        """
        use trained weights to predict the class of each sample
        """
        prob, _, _ = self.forward(X, self.thetas, self.biases)
        return self.labels[np.argmax(prob, axis=1)]

    def score(self, y, h):
        """
        calculates the number of correct predictions over total predictions
        """
        return fraction_correct(y, h)


class Regressor(NeuralNetwork):
    """
    regressor neural network model

    :type theta: np.ndarray
    :desc theta: learned weight matrix [n+1 x k]
    """

    def train(self, X, y, epochs=1000, eta=0.00001, seed=34):
        """
        train the classifier network after one hot
        encoding the target values
        """
        y = two_dims(y)

        super().train(X, y, epochs, eta, seed)

    def activate(self, x):
        """
        the regressor network's activation is linear
        """
        return x

    def deactivate(self, x):
        """
        the derivative of x is 1
        """
        return 1

    def predict(self, X):
        """
        use trained weights to predict the class of each sample
        """
        h, _, _ = self.forward(X, self.thetas, self.biases)
        return h

    def score(self, y, h):
        """
        calculates the number of correct predictions over total predictions
        """
        return coef_determination(y, h)
