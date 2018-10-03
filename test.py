import numpy as np
from sklearn import datasets

from dtlearn import regress, cluster, nn
from dtlearn.bayes import naive
from dtlearn.instance import knn
from dtlearn.utils import Table


table = Table('model', 'accuracy')


def test_gaussian():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = naive.Gaussian()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('gaussian', accuracy)


def test_bernoulli():
    chess = np.load('data/chess.npz')
    X = chess['data']
    y = chess['target']

    model = naive.Bernoulli()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('bernoulli', accuracy)


def test_multinomial():
    balance = np.load('data/balance.npz')
    X = balance['data']
    y = balance['target']

    model = naive.Multinomial()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('multinomial', accuracy)


def test_linear():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    model = regress.Linear()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('linear', accuracy)


def test_logistic():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = regress.Logistic()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('logistic', accuracy)


def test_kmeans():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = cluster.KMeans(3)
    model.train(X)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('kmeans', accuracy)


def test_knn_classifier():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = knn.Classifier()
    model.train(X, y)
    h = model.predict(X, 3)

    accuracy = model.score(y, h)
    table.add('knn classifier', accuracy)


def test_knn_regressor():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    model = knn.Regressor()
    model.train(X, y)
    h = model.predict(X, 3)

    accuracy = model.score(y, h)
    table.add('knn regressor', accuracy)


def test_nn_classifier():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = nn.Classifier([5])
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    table.add('nn classifier', accuracy)


if __name__ == '__main__':
    test_gaussian()
    test_bernoulli()
    test_multinomial()
    test_linear()
    test_logistic()
    test_kmeans()
    test_knn_classifier()
    test_knn_regressor()
    test_nn_classifier()
    table.print()
