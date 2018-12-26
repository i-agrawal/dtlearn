import numpy as np

from .. import Model
from ..utils import biased


class Linear(Model):

    def train(self, x, y):
        xb = biased(x)
        self.w = np.linalg.inv(xb.T @ xb) @ (xb.T @ y)

    def predict(self, x):
        xb = biased(x)
        return xb @ self.w
