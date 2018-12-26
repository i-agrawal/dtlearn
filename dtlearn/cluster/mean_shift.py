import numpy as np

from .. import Distanced
from ..utils import sqsum


class MeanShift(Distanced):

    def train(self, x, bandwidth):
        while True:
            dist = self.dist(x, x)
            nextx = np.array([np.mean(x[d < bandwidth], axis=0) for d in dist])
            if sqsum(x - nextx) < bandwidth:
                break
            x = nextx

        x = nextx[:, np.newaxis]
        centers = x[0]
        for pt in x:
            if np.min(self.dist(centers, pt)) >= bandwidth:
                centers = np.vstack((centers, pt))
        self.centers = centers

    def predict(self, x):
        return np.argmin(self.dist(x, self.centers), axis=1)
