import numpy as np
from sklearn import datasets

from dtlearn import regress, cluster, instance
from dtlearn.bayes import naive


def test_gaussian():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = naive.Gaussian()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    print('gaussian:', accuracy)


def test_bernoulli():
    chess = np.load('data/chess.npy')
    X = chess[:, :-1]
    y = chess[:, -1]

    model = naive.Bernoulli()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    print('bernoulli:', accuracy)


def test_linear():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    model = regress.Linear()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    print('linear:', accuracy)


def test_logistic():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = regress.Logistic()
    model.train(X, y)
    h = model.predict(X)

    accuracy = model.score(y, h)
    print('logistic:', accuracy)


def test_kmeans():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = cluster.KMeans(3)
    model.train(X)
    h = model.predict(X)

    accuracy = model.score(y, h)
    print('kmeans:', accuracy)


def test_knn_classifier():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    model = instance.KNN('classifier')
    model.train(X, y)
    h = model.predict(X, 3)

    accuracy = model.score(y, h)
    print('knn classifier:', accuracy)


def test_knn_regressor():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target

    model = instance.KNN('regressor')
    model.train(X, y)
    h = model.predict(X, 3)

    accuracy = model.score(y, h)
    print('knn regressor:', accuracy)


if __name__ == '__main__':
    test_gaussian()
    test_bernoulli()
    test_linear()
    test_logistic()
    test_kmeans()
    test_knn_classifier()
    test_knn_regressor()
