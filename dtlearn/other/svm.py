import numpy as np

from .. import Model
from ..utils import biased


class SVM(Model):

    def train(self, x, y):
        xb = biased(x)
        self.w = np.linalg.pinv(xb) @ y

    def predict(self, x):
        h = biased(x) @ self.w
        return np.sign(h)
