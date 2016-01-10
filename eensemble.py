# -*- coding: utf-8 -*-
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from unbalanced_dataset.ensemble_sampling import EasyEnsemble
import numpy as np
import matplotlib.pyplot as plt

from utiles import contenido_csv, guardar_csv, votingoutputs, binarizearray

__author__ = 'Juan David Carrillo López'


def learningtoclassify(type_dataset, n_iter=1, data_set=[]):
    features_space = data_set
    number_rows = features_space.shape[0]
    print '\titeration: {}'.format(n_iter)
    c, gamma, cache_size = 1.0, 0.1, 300

    classifiers = {'Poly-2 Kernel': svm.SVC(kernel='poly', degree=2, C=c, cache_size=cache_size),
                   'AdaBoost': AdaBoostClassifier(
                       base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1), learning_rate=0.5,
                       n_estimators=100, algorithm='SAMME'),
                   'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                                                        max_depth=1, random_state=0)}

    type_classifier = {'multi': None, 'binary': None}
    for type_clf in type_classifier.keys():
        prediction = {'Poly-2 Kernel': [], 'AdaBoost': [], 'GradientBoosting': []}
        general_metrics = {'Poly-2 Kernel': [[], []], 'AdaBoost': [[], []], 'GradientBoosting': [[], []]}

        for i_iter in range(n_iter):
            np.random.shuffle(features_space)
            min_max_scaler = MinMaxScaler()
            print '\titeration: {}'.format(i_iter + 1)
            training_set = features_space[:int(number_rows * .8)]
            #  valid_set = features_space[int(number_rows*.5)+1:int(number_rows*.8)]
            test_set = features_space[int(number_rows * .8) + 1:]

            x = min_max_scaler.fit_transform(training_set[:, :4])
            if type_clf == 'multi':
                y = training_set[:, 4:5].ravel()
                y_true = test_set[:, 4:5].ravel()
            else:
                y = np.array(binarizearray(training_set[:, 4:5].ravel()))
                y_true = binarizearray(test_set[:, 4:5].ravel())
            easyens = EasyEnsemble(verbose=True)
            eex, eey = easyens.fit_transform(x, y)

            ciclo, target_names = 0, ('class 1', 'class 2', 'class 3')
            #  for train_ind, test_ind in kf_total:
            for i_ee in range(len(eex)):
                scaled_test_set = min_max_scaler.fit_transform(test_set[:, :4])
                #  print 'Subset {}'.format(ciclo)
                for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                    clf.fit(eex[i_ee], eey[i_ee])
                    y_pred = clf.predict(scaled_test_set)
                    prediction[clf_name].append(y_pred)
                ciclo += 1

            for clf_name, output in prediction.items():
                all_ypred = np.array(output, dtype=int)
                y_pred = votingoutputs(all_ypred)[:, 1].ravel()
                mean_accuracy = accuracy_score(y_true, y_pred)
                general_metrics[clf_name][0].append(np.array(mean_accuracy))
                general_metrics[clf_name][1].append(np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
        #  End i_ter cycle

        for clf_name in classifiers.keys():
            array_a = np.atleast_2d(np.array(general_metrics[clf_name][0])).reshape((30, 1))
            array_b = np.array(general_metrics[clf_name][1])
            results = np.concatenate((array_a, array_b), axis=1)
            guardar_csv(results, 'recursos/resultados/{}_{}_eensemble_{}.csv'.format(type_dataset, type_clf, clf_name))


def machinelearning(type_set):
    data = contenido_csv('recursos/{}.csv'.format(type_set))
    print '\n--------------------------------------->>>>   EASY ENSEMBLE UNDERSAMPLING   ' \
          '<<<<-------------------------------------------'
    learningtoclassify(type_set, 30, np.array(data, dtype='f'))
