# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utiles import contenido_csv, guardar_csv

__author__ = 'Juan David Carrillo López'


def saveandjoin30iter(type_dataset):
    type_class, improve, class_name = ('binary', 'multi'), ('kfolds', 'hparamt'), \
                                      ('Poly-2 Kernel', 'AdaBoost', 'GradientBoosting')

    for t_clf in type_class:
        for t_imprv in improve:
            for clf_name in class_name:
                new_arr = []
                if t_imprv == 'hparamt' and (clf_name == 'AdaBoost' or clf_name == 'GradientBoosting'):
                    pass
                else:
                    for ith_iter in range(1, 31):
                        csvfile_path = 'recursos/resultados/{}/{}_{}_{}_{}.csv'. \
                            format(type_dataset, t_clf, t_imprv, clf_name, ith_iter)
                        print 'Renderizing: {}'.format(csvfile_path)

                        results = contenido_csv(csvfile_path)
                        results = np.array(results, dtype='f')

                        statistics = tuple([results[:, col].ravel().mean() for col in range(results.shape[1])])
                        new_arr.append(statistics)
                    new_arr = np.array(new_arr)

                    guardar_csv(new_arr, 'recursos/resultados/{}/{}_{}_{}.csv'.
                                format(type_dataset, t_clf, t_imprv, clf_name))


def getalldata(data_type, charts=False, f_image='jpeg'):
    data_experiment = {'binary': {'kfolds': {'Poly-2 Kernel': None, 'AdaBoost': None, 'GradientBoosting': None},
                                   'hparamt': {'Poly-2 Kernel': None},
                                   'eensemble': {'Poly-2 Kernel': None, 'AdaBoost': None, 'GradientBoosting': None},
                                   'ros': {'Poly-2 Kernel': None, 'AdaBoost': None, 'GradientBoosting': None}},
                        'multi': {'kfolds': {'Poly-2 Kernel': None, 'AdaBoost': None, 'GradientBoosting': None},
                                  'hparamt': {'Poly-2 Kernel': None},
                                  'eensemble': {'Poly-2 Kernel': None, 'AdaBoost': None, 'GradientBoosting': None},
                                  'ros': {'Poly-2 Kernel': None, 'AdaBoost': None, 'GradientBoosting': None}}
                        }
    for t_clf in data_experiment.keys():
        if t_clf == 'binary':
            col_names = ['accuracy', 'precision class 1', 'precision class 2', 'recall class 1', 'recall class 2',
                         'f1-score class 1', 'f1-score class 2', 'support class 1', 'support class 2']
        else:
            col_names = ['accuracy', 'precision class 1', 'precision class 2', 'precision class 3', 'recall class 1',
                         'recall class 2', 'recall class 3', 'f1-score class 1', 'f1-score class 2', 'f1-score class 3',
                         'support class 1', 'support class 2', 'support class 3']
        for t_imprv in data_experiment[t_clf].keys():
            for clf_name in data_experiment[t_clf][t_imprv].keys():
                csvfile_path = 'recursos/resultados/{}/{}_{}_{}.csv'.format(data_type, t_clf, t_imprv, clf_name)
                print 'Getting: {}'.format(csvfile_path)

                results = contenido_csv(csvfile_path)
                results = np.array(results, dtype='f')
                if charts:
                    plt.subplot()
                    plt.boxplot(results[:, :7])
                    plt.title('{}_{}_{}'.format(t_clf, t_imprv, clf_name))
                    plt.xticks(np.arange(1, 8), ('accuracy', 'precision\nclass 1', 'precision\nclass 2',
                                                 'recall\nclass 1', 'recall\nclass 2', 'f1-score\nclass 1',
                                                 'f1-score\nclass 2'))
                    plt.savefig('recursos/charts/{}_{}_{}.{}'.format(t_clf, t_imprv, clf_name, f_image))
                    plt.close()
                data_experiment[t_clf][t_imprv][clf_name] = pd.DataFrame(results, columns=col_names)
    return data_experiment


def structuriseresults(type_dataset):
    organised_data = getalldata(type_dataset)

    improvements = ('kfolds', 'hparamt', 'eensemble', 'ros')
    for t_clf in organised_data.keys():
        if t_clf == 'binary':
            col_name = 'f1-score class 2'
        else:
            col_name = 'f1-score class 3'
        for ith_improve in range(len(improvements)):
            classifiers_data = organised_data[t_clf][improvements[ith_improve]]
            filtered = np.array([classifiers_data[clf_name][col_name] for clf_name in classifiers_data.keys()]).T
            filtered = pd.DataFrame(filtered, columns=[clf_name for clf_name in classifiers_data.keys()])
            #  print filtered
            print '\n{} - {} report:\n{}'.format(t_clf, improvements[ith_improve], filtered.mean(0))
    print 'fin'


if __name__ == '__main__':
    structuriseresults('ngrams')
    #  saveandjoin30iter('ngrams')
