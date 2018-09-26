import abc


class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        """
        model initialization
        """

    @abc.abstractmethod
    def train(self, X, y):
        """
        uses training data to calculate something
        meaningful in order to predict

        :type X: np.ndarray
        :desc X: training data features [m x n]

        :type y: np.ndarray
        :desc y: training data targets [n x k]
        """

    @abc.abstractmethod
    def predict(self, X):
        """
        using steps from train predict the outcome
        of the given features

        :type X: np.ndarray
        :desc X: testing data [m x n]
        """

    @abc.abstractmethod
    def score(self, y, h):
        """
        given the actual and predicted calculate a
        meaningful score of how well the predicted did

        :type y: np.ndarray
        :desc y: actual results [n x k]

        :type h: np.ndarray
        :desc h: predicted results [n x k]
        """
