# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'


from sklearn import svm, cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

from utiles import contenido_csv, guardar_csv


def searchingclassifier(features_space):
    np.random.shuffle(features_space)
    min_max_scaler = MinMaxScaler()

    c, gamma, cache_size = 1000, 0.001, 300
    X = min_max_scaler.fit_transform(features_space[:, :4])
    y = features_space[:, 4:5].ravel()

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

    tuned_parameters = [{'kernel': ['poly'], 'degree': [2], 'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
                         'cache_size': [cache_size]}]
    scores = ['precision', 'recall']
    for score in scores:
        print '# Tuning hyper-parameters for %s\n' % score
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=10, scoring='%s_weighted' % score)
        clf.fit(x_train, y_train)

        print 'Best parameters set found on development set: \n'
        print '{}\n'.format(clf.best_params_)
        print 'Grid scores on development set:\n'
        for params, mean_score, scores in clf.grid_scores_:
            print '%0.3f (+/-%0.03f) for %r\n' % (mean_score, scores.std() * 2, params)

        print 'Detailed classifications report:\n'
        print 'The model is trained on the full development set.'
        print 'The scores are computed on the hull evaluation set.\n'
        y_true, y_pred = y_test, clf.predict(x_test)
        print classification_report(y_true, y_pred)


if __name__ == '__main__':
    data = contenido_csv('recursos/conjuntos.csv')
    searchingclassifier(np.array(data, dtype='f'))