import numpy as np

from abc import ABCMeta, abstractmethod

from .utils import sqsum


class Model(metaclass=ABCMeta):

    @abstractmethod
    def train(self, x, y):
        """trains the model"""

    @abstractmethod
    def predict(self, x):
        """model's prediction"""


class Distanced(Model):
    def __init__(self, dist=None):
        self.dist = dist or self.__distance

    @staticmethod
    def __distance(a, b):
        diff = a[:, np.newaxis] - b[np.newaxis]
        axis = tuple(range(diff.ndim))[2:]
        return sqsum(diff, axis=axis)
