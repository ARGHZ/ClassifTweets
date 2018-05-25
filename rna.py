print(__doc__)
# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn.learning_curve import learning_curve, validation_curve
import mlpy
from sklearn.neighbors import NearestCentroid
from sklearn import neighbors
from scipy.linalg import pinv2


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def plot_validation_curves(classif, data_x, data_y):
    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        classif, data_x, data_y, param_name="gamma", param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()


class ArtificialNN():

    def __init__(self, entradas, salidas, prueba, referencia):
        self.x_train = entradas
        self.y_train = salidas
        self.x_test = prueba
        self.y_test = referencia

        self.idx = np.arange(self.x_train.shape[0])

    def autorelevancedetermination(self):
        # Fit the ARD Regression
        clf = ARDRegression(compute_score=True)
        clf.fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def linearrgression(self):
        # Fit the ARD Regression
        clf = LinearRegression()
        clf.fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def stochasticgradientdescend(self, alfa=0.001, n_iter=500):
        # Realizamos una serie de combinaciones sobre el conjunto de datos
        np.random.seed(13)
        np.random.shuffle(self.idx)
        x = self.x_train[self.idx]
        y = self.y_train[self.idx]

        # Cierto grado de normalización de parámetros
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x = (x - mean) / std

        clf = SGDClassifier(alpha=alfa, n_iter=n_iter, loss='log').fit(x, y)  # Entrenamos la RNA

        # Obtenemos los datos para probar la RNA
        x = self.x_test

        # Realizamos una serie de combinaciones sobre el conjunto de datos
        idx = np.arange(x.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        x = x[idx]

        # Cierto grado de normalización de parámetros
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x = (x - mean) / std

        # Para cada punto seleccionamos un color, y establecemos el área de desición.
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def logisregression(self):
        logreg = linear_model.LogisticRegression(C=1e5)

        # we create an instance of Neighbours Classifier and fit the data.
        logreg.fit(self.x_train, self.y_train)
        z = logreg.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def decisiontreereg(self, opcion=2):

        # Fit regression model
        clf = DecisionTreeRegressor(max_depth=opcion)
        clf.fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def nearestcentrclassif(self, shrinkage=0.1):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = NearestCentroid(shrink_threshold=shrinkage)
        clf.fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def nearestneighclassif(self, n_neighbors=15, weight='uniform'):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
        clf.fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def multilaypercep(self):
        clf = mlpy.Ridge()  # Principal Component Analysis
        clf.learn(self.x_train, self.y_train)
        z = clf.pred(self.x_test)
        print(np.mean(self.y_test == z))

        return z


class SupportVectorM(ArtificialNN):

    def __init__(self, entradas, salidas, prueba, referencia, c=1.0, gamma=0.7, degree=3):
        super().__init__(entradas, salidas, prueba, referencia)
        self.c = c
        self.gamma = gamma
        self.degree = degree

    def linear(self):
        clf = svm.SVC(kernel='linear', C=self.c).fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        return z

    def linear2(self):
        clf = svm.LinearSVC(C=self.c).fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        cv = cross_validation.ShuffleSplit(self.x_train.shape[0], n_iter=10,
                                           test_size=0.2, random_state=0)
        '''plot_learning_curve(clf, "Learning Curves (SVM, Linear kernel)",
                            self.x_train, self.y_train, (0.5, 1.01), cv=cv, n_jobs=4)
        '''
        return z

    def radialbasisf(self):
        clf = svm.SVC(kernel='rbf', gamma=self.gamma, C=self.c).fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))

        # Plot also the training points
        colours = 'ryg'
        for i in range(self.x_test.shape[0]):
            c_index = int(self.y_test[i])
            plt.scatter(self.x_test[i, 0], self.x_test[i, 1], c=colours[c_index])

        plt.xlabel('Total de Palabras')
        plt.ylabel('Malas Palabras')
        plt.title('RBF kernel SVM')
        plt.show()

        # SVC is more expensive so we do a lower number of CV iterations:
        cv = cross_validation.ShuffleSplit(self.x_train.shape[0], n_iter=10,
                                           test_size=0.2, random_state=0)
        '''plot_learning_curve(clf, "Learning Curves (SVM, RBF kernel)",
                            self.x_train, self.y_train, (0.5, 1.01), cv=cv, n_jobs=4)

        plot_validation_curves(clf, self.x_train, self.y_train)
        '''
        return z

    def polynomial(self):
        clf = svm.SVC(kernel='poly', degree=self.degree, C=self.c).fit(self.x_train, self.y_train)
        z = clf.predict(self.x_test)
        print(np.mean(self.y_test == z))
        return z