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


def learningtoclassify(n_iter=1, data_set=[]):
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
            guardar_csv(results, 'recursos/resultados/{}_ros_{}.csv'.format(type_clf, clf_name))


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


def plotmetric():
    data = {'multi':
                ((0.35919540229885055, 0.38218390804597702, 0.33333333333333331, 0.36206896551724138, 0.40229885057471265, 0.32758620689655171, 0.29310344827586204, 0.29597701149425287, 0.29022988505747127, 0.35919540229885055, 0.33908045977011492, 0.50287356321839083, 0.45402298850574713, 0.27873563218390807, 0.33045977011494254, 0.38505747126436779, 0.34195402298850575, 0.27873563218390807, 0.38505747126436779, 0.32758620689655171, 0.4511494252873563, 0.34195402298850575, 0.40804597701149425, 0.37356321839080459, 0.32183908045977011, 0.34770114942528735, 0.48275862068965519, 0.33045977011494254, 0.49712643678160917, 0.31034482758620691),
                 (0.60344827586206895, 0.58620689655172409, 0.62068965517241381, 0.65804597701149425, 0.6522988505747126, 0.57471264367816088, 0.42528735632183906, 0.60632183908045978, 0.62931034482758619, 0.60344827586206895, 0.58333333333333337, 0.64655172413793105, 0.62643678160919536, 0.65804597701149425, 0.61781609195402298, 0.63793103448275867, 0.56896551724137934, 0.58045977011494254, 0.64080459770114939, 0.63505747126436785, 0.60632183908045978, 0.60057471264367812, 0.59195402298850575, 0.56609195402298851, 0.62068965517241381, 0.58045977011494254, 0.60632183908045978, 0.59482758620689657, 0.63505747126436785, 0.5977011494252874),
                 (0.60919540229885061, 0.64080459770114939, 0.59482758620689657, 0.61494252873563215, 0.60344827586206895, 0.60057471264367812, 0.55747126436781613, 0.55172413793103448, 0.61781609195402298, 0.62643678160919536, 0.55172413793103448, 0.62356321839080464, 0.58620689655172409, 0.62643678160919536, 0.58620689655172409, 0.63218390804597702, 0.60919540229885061, 0.68390804597701149, 0.62931034482758619, 0.57471264367816088, 0.62356321839080464, 0.58333333333333337, 0.60344827586206895, 0.5977011494252874, 0.58045977011494254, 0.54022988505747127, 0.60632183908045978, 0.56321839080459768, 0.65517241379310343, 0.6522988505747126),
                 (0.63793103448275867, 0.62643678160919536, 0.61781609195402298, 0.5977011494252874, 0.62643678160919536, 0.62643678160919536, 0.5545977011494253, 0.61206896551724133, 0.62356321839080464, 0.61781609195402298, 0.59195402298850575, 0.64942528735632188, 0.62643678160919536, 0.66666666666666663, 0.61206896551724133, 0.62356321839080464, 0.63505747126436785, 0.68103448275862066, 0.62931034482758619, 0.61206896551724133, 0.62068965517241381, 0.62931034482758619, 0.60632183908045978, 0.63505747126436785, 0.60919540229885061, 0.58333333333333337, 0.60344827586206895, 0.62643678160919536, 0.64942528735632188, 0.65517241379310343),
                 (0.64655172413793105, 0.63505747126436785, 0.60057471264367812, 0.68103448275862066, 0.63218390804597702, 0.57471264367816088, 0.29310344827586204, 0.60344827586206895, 0.60919540229885061, 0.60919540229885061, 0.58908045977011492, 0.59195402298850575, 0.60344827586206895, 0.61494252873563215, 0.61494252873563215, 0.60057471264367812, 0.61781609195402298, 0.61494252873563215, 0.62931034482758619, 0.63793103448275867, 0.61494252873563215, 0.59482758620689657, 0.58045977011494254, 0.62931034482758619, 0.64942528735632188, 0.58045977011494254, 0.62643678160919536, 0.61494252873563215, 0.62643678160919536, 0.61494252873563215)),
            'binary':
                ((0.66379310344827591, 0.71551724137931039, 0.76436781609195403, 0.70402298850574707, 0.71264367816091956, 0.74712643678160917, 0.67241379310344829, 0.71264367816091956, 0.76436781609195403, 0.74137931034482762, 0.74425287356321834, 0.6954022988505747, 0.66091954022988508, 0.70977011494252873, 0.67528735632183912, 0.67241379310344829, 0.6954022988505747, 0.7385057471264368, 0.70114942528735635, 0.65804597701149425, 0.64942528735632188, 0.71551724137931039, 0.6954022988505747, 0.77298850574712641, 0.68103448275862066, 0.7183908045977011, 0.66091954022988508, 0.74425287356321834, 0.69252873563218387, 0.6954022988505747),
                 (0.67241379310344829, 0.74137931034482762, 0.74712643678160917, 0.71264367816091956, 0.73275862068965514, 0.72413793103448276, 0.67241379310344829, 0.68965517241379315, 0.72126436781609193, 0.72701149425287359, 0.71551724137931039, 0.7385057471264368, 0.67528735632183912, 0.68678160919540232, 0.71551724137931039, 0.7183908045977011, 0.70114942528735635, 0.74425287356321834, 0.71551724137931039, 0.7183908045977011, 0.66666666666666663, 0.7385057471264368, 0.57183908045977017, 0.74137931034482762, 0.67528735632183912, 0.74137931034482762, 0.68678160919540232, 0.72126436781609193, 0.67816091954022983, 0.67241379310344829),
                 (0.66379310344827591, 0.7816091954022989, 0.78735632183908044, 0.75287356321839083, 0.77011494252873558, 0.78735632183908044, 0.67241379310344829, 0.71264367816091956, 0.76724137931034486, 0.76724137931034486, 0.76436781609195403, 0.77011494252873558, 0.66091954022988508, 0.70977011494252873, 0.70114942528735635, 0.77586206896551724, 0.6954022988505747, 0.75287356321839083, 0.7614942528735632, 0.77011494252873558, 0.64942528735632188, 0.76436781609195403, 0.6954022988505747, 0.76724137931034486, 0.68103448275862066, 0.7614942528735632, 0.74137931034482762, 0.77586206896551724, 0.69252873563218387, 0.6954022988505747),
                 (0.72413793103448276, 0.77873563218390807, 0.77873563218390807, 0.75574712643678166, 0.7816091954022989, 0.78735632183908044, 0.74712643678160917, 0.7614942528735632, 0.77873563218390807, 0.7816091954022989, 0.76436781609195403, 0.77298850574712641, 0.73563218390804597, 0.75574712643678166, 0.74712643678160917, 0.77011494252873558, 0.77011494252873558, 0.7614942528735632, 0.75574712643678166, 0.77011494252873558, 0.74712643678160917, 0.77298850574712641, 0.69252873563218387, 0.77586206896551724, 0.75287356321839083, 0.77011494252873558, 0.73275862068965514, 0.78735632183908044, 0.7816091954022989, 0.76724137931034486),
                 (0.66379310344827591, 0.75, 0.72126436781609193, 0.7068965517241379, 0.73563218390804597, 0.72701149425287359, 0.67241379310344829, 0.70402298850574707, 0.71551724137931039, 0.75574712643678166, 0.7385057471264368, 0.75287356321839083, 0.66091954022988508, 0.70977011494252873, 0.72126436781609193, 0.75574712643678166, 0.69827586206896552, 0.74137931034482762, 0.69252873563218387, 0.70977011494252873, 0.64942528735632188, 0.74425287356321834, 0.5316091954022989, 0.74425287356321834, 0.68103448275862066, 0.75, 0.72413793103448276, 0.7385057471264368, 0.69252873563218387, 0.69827586206896552),
                 (0.54597701149425293, 0.59482758620689657, 0.5431034482758621, 0.5114942528735632, 0.60344827586206895, 0.46264367816091956, 0.5316091954022989, 0.51436781609195403, 0.51724137931034486, 0.51724137931034486, 0.52011494252873558, 0.50862068965517238, 0.54597701149425293, 0.54885057471264365, 0.55747126436781613, 0.52298850574712641, 0.55172413793103448, 0.58620689655172409, 0.50287356321839083, 0.52586206896551724, 0.48563218390804597, 0.52873563218390807, 0.57183908045977017, 0.56896551724137934, 0.57183908045977017, 0.56034482758620685, 0.55172413793103448, 0.52586206896551724, 0.57183908045977017, 0.51436781609195403))
    }
    #  stats = cbook.boxplot_stats(data)
    plt.subplot()
    plt.boxplot(np.array(data['binary']).T)
    plt.show()


def getnewdataset():
    with open('recursos/bullyingV3/tweet.json') as json_file:
        for line in json_file:
            json_data = (json.loads(line)['id'], str(json.loads(line)['text']))
    return json_data


def machinelearning():
    data = contenido_csv('recursos/nongrams.csv')
    print '\n--------------------------------------->>>>   RANDOM OVERSAMPLING   ' \
          '<<<<-------------------------------------------'
    learningtoclassify(30, np.array(data, dtype='f'))


if __name__ == '__main__':
    machinelearning()