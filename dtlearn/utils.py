import numpy as np
from scipy.optimize import linear_sum_assignment


def add_bias(x):
    """
    add a column of ones to the right of a matrix
    """
    m = x.shape[0]
    return np.append(x, np.ones((m, 1)), axis=1)


def two_dims(y):
    """
    make sure y has two dimensions
    """
    if y.ndim == 1:
        y = y[:, np.newaxis]
    return y


def normal_pdf(x, u, v):
    """
    find the probability of a value (x) given the mean (u) and variance (v)

    to see the function look up 'normal distribution' in wikipedia
    """
    return np.exp(-(x - u)**2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def sigmoid(x):
    """
    find the sigmoid of x

    to see the function look up 'sigmoid function' in wikipedia
    """
    return 1.0 / (1.0 + np.exp(-x))


def desigmoid(x):
    """
    find the derivative of sigmoid of x

    to see the function look up 'sigmoid function' in wikipedia
    """
    x = sigmoid(x)
    return x * (1.0 - x)


def fraction_correct(y, h):
    """
    calculates the fraction of correct predictions
    """
    return np.mean(y == h)


def coef_determination(y, h):
    """
    calculates the coefficient of determination

    to see the function look up 'coefficient of determination' in wikipedia
    """
    dist = np.sum((y - h)**2)
    variance = np.sum((y - np.mean(y))**2)
    return 1.0 - dist / variance


def setted(x):
    """
    find the unique values in x with a set of indices for each unique value
    """
    values = {}
    for i, v in enumerate(x):
        if v not in values:
            values[v] = set()
        values[v].add(i)
    return list(values.values())


def cluster_intersect(y, h):
    """
    assign each cluster to most intersecting label and find the percentage correct
    """
    labels = setted(y)
    guesses = setted(h)

    m, n = len(labels), len(guesses)
    intersect = np.empty((m, n))
    union = np.empty((m, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(guesses):
            intersect[i, j] = len(a & b)
            union[i, j] = len(a | b)

    rows, cols = linear_sum_assignment(-(intersect / union))
    return np.sum([intersect[r, c] for r, c in zip(rows, cols)]) / len(y)


class Table:
    def __init__(self, *columns, precision=5):
        """
        helpful for printing nice looking table

        :type columns: List[str]
        :desc columns: list of column names

        :type rows: List[Tuple]
        :desc rows: list of table rows from add

        :type width: List[int]
        :desc width: the width of each column
        """
        self.columns = columns
        self.__rows = []
        self.__width = [len(x) for x in self.columns]
        self.precision = precision / 10.0

    def add(self, *args):
        assert len(args) == len(self.columns)
        data = ['{1:{0}f}'.format(self.precision, x) if isinstance(x,float) else str(x) for x in args]
        self.__width = np.maximum(self.__width, [len(x) for x in data])
        self.__rows.append(data)

    def print(self):
        pattern = '  '.join(['{{:<{0}}}'.format(w) for w in self.__width])
        header = pattern.format(*self.columns)
        print(header)
        print('-'*len(header))
        for row in self.__rows:
            print(pattern.format(*row))
