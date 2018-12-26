import numpy as np
from sklearn import datasets

from dtlearn import regress, cluster, other, nbayes, utils


iris = datasets.load_iris()
boston = datasets.load_boston()
linnerud = datasets.load_linnerud()


def test_linear():
    X = linnerud.data
    y = linnerud.target

    model = regress.Linear()
    model.train(X, y)
    h = model.predict(X)

    accuracy = utils.correlation(y, h)
    print(model.__class__.__name__, accuracy)


def test_logistic():
    X = iris.data
    y = iris.target
    one_hot = np.eye(3)[y]

    model = regress.Logistic()
    model.train(X, one_hot)
    h = model.predict(X)

    accuracy = np.sum(y == h) / y.shape[0]
    print(model.__class__.__name__, accuracy)


def test_kmeans():
    X = iris.data
    y = iris.target

    model = cluster.KMeans()
    model.train(X, 3)
    h = model.predict(X)

    accuracy = utils.purity(y, h)
    print(model.__class__.__name__, accuracy)


def test_dbscan():
    X = iris.data
    y = iris.target > 0

    model = cluster.DBSCAN()
    h = model.predict(X, 10, 0.09)

    accuracy = utils.purity(y, h)
    print(model.__class__.__name__, accuracy)


def test_knn_classifier():
    X = iris.data
    y = iris.target

    model = other.KNN()
    model.train(X, y)
    h = model.predict(X, 10)

    accuracy = utils.purity(y, h)
    print(model.__class__.__name__, accuracy)


def test_knn_regressor():
    X = linnerud.data
    y = linnerud.target

    model = other.KNN()
    model.train(X, y)
    h = model.predict(X, 3, mode='regression')

    accuracy = utils.correlation(y, h)
    print(model.__class__.__name__, accuracy)


def test_lvq():
    X = iris.data
    y = iris.target

    model = other.LVQ()
    model.train(X, y)
    h = model.predict(X)

    accuracy = np.sum(y == h) / y.shape[0]
    print(model.__class__.__name__, accuracy)


def test_gaussian():
    X = iris.data
    y = iris.target

    model = nbayes.Gaussian()
    model.train(X, y)
    h = model.predict(X)

    accuracy = np.sum(y == h) / y.shape[0]
    print(model.__class__.__name__, accuracy)


def test_bernoulli():
    X = iris.data
    y = iris.target
    X = X > np.mean(X, axis=0)

    model = nbayes.Bernoulli()
    model.train(X, y)
    h = model.predict(X)

    accuracy = np.sum(y == h) / y.shape[0]
    print(model.__class__.__name__, accuracy)


def test_multinomial():
    X = iris.data
    X = np.round((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    y = iris.target

    model = nbayes.Multinomial()
    model.train(X, y)
    h = model.predict(X)

    accuracy = np.sum(y == h) / y.shape[0]
    print(model.__class__.__name__, accuracy)


def test_svm():
    X = iris.data
    y = iris.target
    y = 2 * (y > 0) - 1

    model = other.SVM()
    model.train(X, y)
    h = model.predict(X)

    accuracy = np.sum(y == h) / y.shape[0]
    print(model.__class__.__name__, accuracy)


if __name__ == '__main__':
    test_linear()
    test_logistic()
    test_kmeans()
    test_dbscan()
    test_knn_classifier()
    test_knn_regressor()
    test_lvq()
    test_gaussian()
    test_bernoulli()
    test_multinomial()
    test_svm()
