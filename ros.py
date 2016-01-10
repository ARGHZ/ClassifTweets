# -*- coding: utf-8 -*-
import random
import json

from pyexcel_xlsx import XLSXBook
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import itemfreq
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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


def learningtoclassify(type_dataset, n_iter=1, data_set=[]):
    features_space = data_set
    number_rows = features_space.shape[0]

    c, gamma, cache_size = 1.0, 0.1, 300
    classifiers = {'Poly-2 Kernel': svm.SVC(kernel='poly', degree=2, C=c, cache_size=cache_size),
                   'AdaBoost': AdaBoostClassifier(
                       base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1), learning_rate=0.5,
                       n_estimators=100, algorithm='SAMME'),
                   'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                                                        max_depth=1, random_state=0)}

    type_classifier = {'multi': None, 'binary': None}
    for type_clf in type_classifier.keys():
        general_metrics = {'Poly-2 Kernel': [[], []], 'AdaBoost': [[], []], 'GradientBoosting': [[], []]}

        for i_iter in range(n_iter):
            np.random.shuffle(features_space)
            min_max_scaler = MinMaxScaler()
            print '\titeration: {}'.format(i_iter + 1)
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
                clf.fit(rox, roy)
                y_pred = clf.predict(scaled_test_set)
                general_metrics[clf_name][0].append(accuracy_score(y_true, y_pred))
                general_metrics[clf_name][1].append(np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())

        for clf_name in classifiers.keys():
            array_a = np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1)
            array_b = np.array(general_metrics[clf_name][1])
            results = np.concatenate((array_a, array_b), axis=1)
            guardar_csv(results, 'recursos/resultados/{}_{}_ros_{}.csv'.format(type_dataset, type_clf, clf_name))


def readexceldata(path_file):
    book = XLSXBook(path_file)
    content = book.sheets()
    data_set = np.array(content['filtro'])[:2326, :7]
    filtro = np.array([row for row in data_set if row[6] <= 2])
    n_filas, n_columnas = filtro.shape

    rangos, filtro2 = [0, 0, 0], []
    for row in filtro[:n_filas-4, :]:
        if row[6] == 2:
            valor_selecc = int((row[1] + row[2]) / 2)
        else:
            valor_selecc = int(random.choice(row[1:3]))
        if valor_selecc < 4:
            rangos[0] += 1
            valor_selecc = 1
        elif valor_selecc > 6:
            rangos[2] += 1
            valor_selecc = 3
        else:
            rangos[1] += 1
            valor_selecc = 2

        row[0] = row[0].encode('latin-1', errors='ignore').replace('<<=>>', '')
        filtro2.append((row[0], valor_selecc))
    return filtro2


def getnewdataset():
    with open('recursos/bullyingV3/tweet.json') as json_file:
        for line in json_file:
            json_data = (json.loads(line)['id'], str(json.loads(line)['text']))
    return json_data


def machinelearning(type_set):
    data = contenido_csv('recursos/{}.csv'.format(type_set))
    print '\n--------------------------------------->>>>   RANDOM OVERSAMPLING   ' \
          '<<<<-------------------------------------------'
    learningtoclassify(type_set, 30, np.array(data, dtype='f'))
