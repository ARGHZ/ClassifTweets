# -*- coding: utf-8 -*-
import random
import json
import cProfile, pstats, StringIO

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import itemfreq
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np

from utiles import contenido_csv, binarizearray, guardar_csv
from mining import getfeaturessample

__author__ = 'Juan David Carrillo López'

pr = cProfile.Profile()


def votingoutputs(temp_array):
    index_outputs = []
    for col_index in range(temp_array.shape[1]):
        item_counts = itemfreq(temp_array[:, col_index])
        max_times = 0
        for class_label, n_times in item_counts:
            if n_times > max_times:
                last_class, max_times = class_label, n_times
        index_outputs.append((col_index, class_label))
    return np.array(index_outputs)


def learningtoclassify(type_dataset, n_iter=1, data_set=[], specific_clf=[]):
    features_space = data_set
    number_rows = features_space.shape[0]

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
        for i in ('binary', 'multi'):
            for j in classifiers.keys():
                    specific_clf.append('{}_{}_ros_{}'.format(type_dataset, i, j))

    for type_clf in type_classifier.keys():
        if len(specific_clf) <= 0:
            general_metrics = {'Poly-2 Kernel': [[], [], [], []], 'AdaBoost': [[], [], [], []],
                               'GradientBoosting': [[], [], [], []]}
        else:
            general_metrics = {selected_clf.split('_')[3]: [[], [], [], []] for selected_clf in specific_clf}

        for i_iter in range(n_iter):
            np.random.shuffle(features_space)
            min_max_scaler = MinMaxScaler()
            print '\n\titeration: {}'.format(i_iter + 1)
            training_set = features_space[:int(number_rows * .8)]
            test_set = features_space[int(number_rows * .8) + 1:]
            x = min_max_scaler.fit_transform(training_set[:, :features_space.shape[1] - 1])
            scaled_test_set = min_max_scaler.fit_transform(test_set[:, :features_space.shape[1] - 1])

            ovsampling = RandomOverSampler()
            if type_clf == 'binary':
                y = np.array(binarizearray(training_set[:, features_space.shape[1] - 1:].ravel()))
                y_true = np.array(binarizearray(test_set[:, features_space.shape[1] - 1:].ravel()))
            else:
                y = training_set[:, features_space.shape[1] - 1:].ravel()
                y_true = test_set[:, features_space.shape[1] - 1:].ravel()
            rox, roy = ovsampling.fit_sample(x, y)

            for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                actual_clf = '{}_{}_ros_{}'.format(type_dataset, type_clf, clf_name)
                try:
                    specific_clf.index(actual_clf)
                except ValueError:
                    pass
                else:
                    pr.enable()
                    clf.fit(rox, roy)
                    pr.disable()
                    s = StringIO.StringIO()
                    sortby = 'cumulative'
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    tt = round(ps.total_tt, 6)
                    #  print '------------------------------------>>>>   {} \n{}\n'.format(clf_name)
                    y_pred = clf.predict(scaled_test_set)
                    if type_clf == 'multi':
                        y_true = np.random.random_integers(1, 3, len(y_true))
                    else:
                        y_true = np.random.random_integers(0, 1, len(y_true))
                    general_metrics[clf_name][0].append(accuracy_score(y_true, y_pred))
                    general_metrics[clf_name][1].append(
                        np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
                    last_metric = '-'.join([str(elem) for elem in confusion_matrix(y_true, y_pred).ravel()])
                    general_metrics[clf_name][2].append(tt)
                    general_metrics[clf_name][3].append(last_metric)
        # End of the ith iterations

        for clf_name in general_metrics.keys():
            array_a = np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1)
            array_b = np.array(general_metrics[clf_name][1])
            array_c = np.expand_dims(np.array(general_metrics[clf_name][2]), axis=1)
            array_d = np.expand_dims(np.array(general_metrics[clf_name][3]), axis=1)
            try:
                results = np.concatenate((array_a, array_b, array_c, array_d), axis=1)
            except ValueError as e:
                print 'ERROR whilst saving {}_{}_ros_{}_{} metrics: {}'.\
                    format(type_dataset, type_clf, clf_name, i_iter, str(e))
                pass
            else:
                guardar_csv(results, 'recursos/resultados/experiment_tfidf/{}_{}_ros_{}.csv'.
                            format(type_dataset, type_clf, clf_name))
                print 'saved {}_{}_ros_{}_{} metrics'.format(type_dataset, type_clf, clf_name, i_iter)
    specific_clf = []  # End of multi/binary cicle


def machinelearning(type_set):
    data = getfeaturessample(type_set)

    print '\n--------------------------------------->>>>   RANDOM OVERSAMPLING   ' \
          '<<<<-------------------------------------------'
    selected_clfs = ['nongrams_binary_ros_AdaBoost', 'nongrams_binary_ros_GradientBoosting',
                     'ngrams_binary_ros_AdaBoost', 'ngrams_multi_ros_AdaBoost', 'nongrams_multi_ros_AdaBoost',
                     'nongrams_multi_ros_GradientBoosting']
    learningtoclassify(type_set.replace('rand_', ''), 30, data)
