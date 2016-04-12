# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm, cross_validation
from sklearn.metrics import classification_report
import numpy as np

from utiles import contenido_csv, guardar_csv, binarizearray

__author__ = 'Juan David Carrillo López'


def searchinghparameters(features_space):
    np.random.shuffle(features_space)
    min_max_scaler = MinMaxScaler()

    X = min_max_scaler.fit_transform(features_space[:, :4])

    type_classifier = {'multi': None, 'binary': None}
    for type_clf in type_classifier.keys():
        if type_clf != 'binary':
            y = features_space[:, 4:5].ravel()
        else:
            y = binarizearray(features_space[:, 4:5].ravel())
        print '\t--------->>  {} - class type  <<---------'.format(type_clf)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

        tuned_parameters = [{'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
                             'cache_size': [300, 500]}]
        #  scores = ['precision', 'recall']
        scores = ['f1', ]
        for score in scores:
            print '# Tuning hyper-parameters for %s\n' % score
            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=10, scoring='%s_weighted' % score)
            clf.fit(x_train, y_train)

            print 'Best parameters set found on development set: \n'
            type_classifier[type_clf] = clf.best_params_
            print '{}\n'.format(clf.best_params_)
            print 'Grid scores on development set:\n'
            for params, mean_score, scores in clf.grid_scores_:
                print '%0.3f (+/-%0.03f) for %r\n' % (mean_score, scores.std() * 2, params)

            print 'Detailed classifications report:\n'
            print 'The model is trained on the full development set.'
            print 'The scores are computed on the hull evaluation set.\n'
            y_true, y_pred = y_test, clf.predict(x_test)
            print classification_report(y_true, y_pred)
    return type_classifier


def learningtoclassify(type_dataset, n_iter=1, data_set=[]):
    features_space = data_set
    np.random.shuffle(features_space)
    min_max_scaler = MinMaxScaler()

    new_params = searchinghparameters(data_set)
    for i_iter in range(n_iter):
        print '\titeration: {}'.format(i_iter + 1)
        #  training_set = features_space[:int(number_rows * .8)]
        #  valid_set = features_space[int(number_rows*.5)+1:int(number_rows*.8)]
        #  test_set = features_space[int(number_rows * .8) + 1:]

        type_classifier = {'multi': None, 'binary': None}

        x = min_max_scaler.fit_transform(features_space[:, :4])

        kf_total = cross_validation.KFold(len(x), n_folds=10)
        for type_clf in type_classifier.keys():
            classifiers = {'Poly-2 Kernel': svm.SVC(**new_params[type_clf]), }
            general_metrics = {'Poly-2 Kernel': [[], []], }
            if type_clf == 'binary':
                y = np.array(binarizearray(features_space[:, 4:5].ravel()))
            else:
                y = features_space[:, 4:5].ravel()

            for train_ind, test_ind in kf_total:
                scaled_test_set = x[test_ind]
                for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                    inst_clf = clf.fit(x[train_ind], y[train_ind])
                    y_pred = clf.predict(scaled_test_set)
                    y_true = y[test_ind]
                    ind_score = inst_clf.score(x[test_ind], y[test_ind])
                    general_metrics[clf_name][0].append(ind_score)
                    general_metrics[clf_name][1].append(np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
            '''
            for clf_name in classifiers.keys():
                results = np.concatenate((np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1),
                                          np.array(general_metrics[clf_name][1])), axis=1)
                guardar_csv(results, 'recursos/resultados/{}_{}_hparamt_{}_{}.csv'.
                            format(type_dataset, type_clf, clf_name, i_iter + 1))
                            '''


def machinelearning(type_set):
    print '\n--------------------------------------->>>>   SEARCHING HYPERPARAMETERS   ' \
          '<<<<-------------------------------------------'
    data = contenido_csv('recursos/{}.csv'.format(type_set))
    features_space = np.array(data, dtype='f')
    learningtoclassify(type_set, 1, features_space)
