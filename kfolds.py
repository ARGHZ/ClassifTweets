# -*- coding: utf-8 -*-
import StringIO
import cProfile
import pstats

import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from utiles import guardar_csv, contenido_csv, binarizearray
from mining import getfeaturessample

__author__ = 'Juan David Carrillo López'

pr = cProfile.Profile()


def learningtoclassify(t_dataset, i_iter='', data_set=[], specific_clf=[]):
    features_space = data_set

    np.random.shuffle(features_space)
    print '\titeration: {}'.format(i_iter)
    #  training_set = features_space[:int(number_rows * .8)]
    #  valid_set = features_space[int(number_rows*.5)+1:int(number_rows*.8)]
    #  test_set = features_space[int(number_rows * .8) + 1:]

    c, gamma, cache_size = 1.0, 0.1, 300

    classifiers = {'Poly-2 Kernel': svm.SVC(kernel='poly', degree=2, C=c, cache_size=cache_size),
                   'AdaBoost': AdaBoostClassifier(
                       base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1), learning_rate=0.5,
                   n_estimators=100, algorithm='SAMME'),
                   'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                                                  max_depth=1, random_state=0)}

    if len(specific_clf) >= 1:
        type_classifier = {selected_clf.split('_')[1]: None for selected_clf in specific_clf}
    else:
        type_classifier = {'multi': None, 'binary': None}

    x = features_space[:, :features_space.shape[1] - 1]

    kf_total = cross_validation.KFold(len(x), n_folds=10)
    for type_clf in type_classifier.keys():
        if len(specific_clf) == 0:
            general_metrics = {'Poly-2 Kernel': [[], [], [], []], 'AdaBoost': [[], [], [], []],
                               'GradientBoosting': [[], [], [], []]}
            for i in ('binary', 'multi'):
                for j in classifiers.keys():
                    specific_clf.append('{}_{}_kfolds_{}'.format(t_dataset, i, j))
        else:
            general_metrics = {selected_clf.split('_')[3]: [[], [], [], []] for selected_clf in specific_clf}
        if type_clf == 'binary':
            y = np.array(binarizearray(features_space[:, features_space.shape[1] - 1:].ravel()))
        else:
            y = features_space[:, features_space.shape[1] - 1:].ravel()

        for train_ind, test_ind in kf_total:
            scaled_test_set = x[test_ind]
            for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                actual_clf = '{}_{}_kfolds_{}'.format(t_dataset, type_clf, clf_name)
                try:
                    specific_clf.index(actual_clf)
                except ValueError:
                    pass
                else:
                    pr.enable()
                    inst_clf = clf.fit(x[train_ind], y[train_ind])
                    pr.disable()
                    s = StringIO.StringIO()
                    sortby = 'cumulative'
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    tt = round(ps.total_tt, 6)
                    #  print '------------------------------------>>>>   {} \n{}\n'.format(clf_name)
                    y_pred = clf.predict(scaled_test_set)
                    # y_true = y[test_ind]
                    if type_clf == 'multi':
                        y_true = np.random.random_integers(1, 3, test_ind.shape[0])
                    else:
                        y_true = np.random.random_integers(0, 1, test_ind.shape[0])
                    ind_score = inst_clf.score(x[test_ind], y_true)
                    general_metrics[clf_name][0].append(ind_score)
                    general_metrics[clf_name][1].append(
                            np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
                    last_metric = '-'.join([str(elem) for elem in confusion_matrix(y_true, y_pred).ravel()])
                    general_metrics[clf_name][2].append(tt)
                    general_metrics[clf_name][3].append(last_metric)

        for clf_name in general_metrics.keys():
            array_a = np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1)
            array_b = np.array(general_metrics[clf_name][1])
            array_c = np.expand_dims(np.array(general_metrics[clf_name][2]), axis=1)
            array_d = np.expand_dims(np.array(general_metrics[clf_name][3]), axis=1)
            try:
                results = np.concatenate((array_a, array_b, array_c, array_d), axis=1)
            except ValueError as e:
                print 'ERROR whilst saving {}_{}_kfolds_{}_{} metrics: {}'.\
                    format(t_dataset, type_clf, clf_name, i_iter, str(e))
                pass
            else:
                guardar_csv(results, 'recursos/resultados/experiment_tfidf/{}_{}_kfolds_{}_{}.csv'.
                            format(t_dataset, type_clf, clf_name, i_iter))
                print 'saved {}_{}_kfolds_{}_{} metrics'.format(t_dataset, type_clf, clf_name, i_iter)


def machinelearning(type_set):
    data = getfeaturessample(type_set)
    print '\n---------------------------------------->>>>   10-FOLDS   <<<<--------------------------------------------'
    print '\n------------------------------------>>>>   NO NORMALISATION   <<<<----------------------------------------'
    for cicle in range(30):
        learningtoclassify(type_set.replace('rand_', ''), cicle + 1, data)
