# -*- coding: utf-8 -*-
import random
import json

from pyexcel_xlsx import XLSXBook
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import itemfreq
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from unbalanced_dataset.over_sampling import OverSampler
import numpy as np
import matplotlib.pyplot as plt

from utiles import contenido_csv, binarizearray, guardar_csv

__author__ = 'Juan David Carrillo López'


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
    type_classifier = {selected_clf.split('_')[1]: None for selected_clf in specific_clf}
    for type_clf in type_classifier.keys():
        if len(specific_clf) <= 0:
            general_metrics = {'Poly-2 Kernel': [[], [], []], 'AdaBoost': [[], [], []],
                               'GradientBoosting': [[], [], []]}
        else:
            general_metrics = {selected_clf.split('_')[3]: [[], [], []] for selected_clf in specific_clf}

        for i_iter in range(n_iter):
            np.random.shuffle(features_space)
            min_max_scaler = MinMaxScaler()
            print '\n\titeration: {}'.format(i_iter + 1)
            training_set = features_space[:int(number_rows * .8)]
            test_set = features_space[int(number_rows * .8) + 1:]
            x = min_max_scaler.fit_transform(training_set[:, :4])
            scaled_test_set = min_max_scaler.fit_transform(test_set[:, :4])

            ovsampling = OverSampler(verbose=True)
            if type_clf == 'binary':
                y = np.array(binarizearray(training_set[:, 4:5].ravel()))
                y_true = np.array(binarizearray(test_set[:, 4:5].ravel()))
            else:
                y = training_set[:, 4:5].ravel()
                y_true = test_set[:, 4:5].ravel()
            rox, roy = ovsampling.fit_transform(x, y)

            for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                actual_clf = '{}_{}_ros_{}'.format(type_dataset, type_clf, clf_name)
                try:
                    ith_idx = specific_clf.index(actual_clf)
                except ValueError:
                    pass
                else:
                    clf.fit(rox, roy)
                    y_pred = clf.predict(scaled_test_set)
                    general_metrics[clf_name][0].append(accuracy_score(y_true, y_pred))
                    general_metrics[clf_name][1].append(
                        np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
                    last_metric = confusion_matrix(y_true, y_pred).ravel()
                    general_metrics[clf_name][2].append(last_metric)
        for clf_name in general_metrics.keys():
            array_a = np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1)
            array_b = np.array(general_metrics[clf_name][1])
            array_c = np.array(general_metrics[clf_name][2])
            try:
                results = np.concatenate((array_a, array_b, array_c), axis=1)
            except ValueError as e:
                print '{}: {}'.format(clf_name, str(e))
            else:
                guardar_csv(results, 'recursos/resultados/elite_{}_{}_ros_{}.csv'.
                            format(type_dataset, type_clf, clf_name))


def machinelearning(type_set, cmd_line=''):
    data = contenido_csv('recursos/{}.csv'.format(type_set))
    print '\n--------------------------------------->>>>   RANDOM OVERSAMPLING   ' \
          '<<<<-------------------------------------------'
    selected_clfs = ['nongrams_binary_ros_AdaBoost', 'nongrams_binary_ros_GradientBoosting',
                     'ngrams_binary_ros_AdaBoost', 'ngrams_multi_ros_AdaBoost', 'nongrams_multi_ros_AdaBoost',
                     'nongrams_multi_ros_GradientBoosting']
    learningtoclassify(type_set, 30, np.array(data, dtype='f'), specific_clf=selected_clfs)
