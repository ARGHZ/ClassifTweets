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
                        csvfile_path = 'recursos/resultados/{}_{}_{}_{}_{}.csv'. \
                            format(type_dataset, t_clf, t_imprv, clf_name, ith_iter)
                        print 'Renderizing: {}'.format(csvfile_path)

                        results = contenido_csv(csvfile_path)
                        results = np.array(results, dtype='f')

                        statistics = tuple([results[:, col].ravel().mean() for col in range(results.shape[1])])
                        new_arr.append(statistics)
                    new_arr = np.array(new_arr)

                    guardar_csv(new_arr, 'recursos/resultados/{}_{}_{}_{}.csv'.
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
            col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'R-class 1', 'R-class 2',
                         'F1-class 1', 'F1-class 2', 'support class 1', 'support class 2']
            support_class = 2
        else:
            col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'P-class 3', 'R-class 1',
                         'R-class 2', 'R-class 3', 'F1-class 1', 'F1-class 2', 'F1-class 3',
                         'support class 1', 'support class 2', 'support class 3']
            support_class = 3
        for t_imprv in data_experiment[t_clf].keys():
            for clf_name in data_experiment[t_clf][t_imprv].keys():
                csvfile_path = 'recursos/resultados/{}_{}_{}_{}.csv'.format(data_type, t_clf, t_imprv, clf_name)
                print 'Getting: {}'.format(csvfile_path)

                results = contenido_csv(csvfile_path)
                results = np.array(results, dtype='f')
                if charts:
                    plt.subplot()
                    plt.boxplot(results[:, :len(col_names) - support_class])
                    plt.title('{}_{}_{}_{}'.format(t_clf, data_type, t_imprv, clf_name))
                    plt.xticks(np.arange(1, len(col_names) - support_class + 1), col_names, rotation='vertical')
                    plt.savefig('recursos/charts/{}_{}_{}_{}.{}'.format(data_type, t_clf, t_imprv, clf_name, f_image))
                    plt.close()
                data_experiment[t_clf][t_imprv][clf_name] = pd.DataFrame(results, columns=col_names)
    return data_experiment


def structuriseresults(type_dataset):
    organised_data = getalldata(type_dataset)

    new_arr = []
    for ith_tclf in organised_data.keys():
        if ith_tclf == 'binary':
            col_name = ['Accuracy', 'P-class 2', 'R-class 2', 'F1-class 2']
        else:
            col_name = ['Accuracy', 'P-class 3', 'R-class 3', 'F1-class 3']
        for ith_improve in organised_data[ith_tclf].keys():

            for ith_clf in organised_data[ith_tclf][ith_improve].keys():
                filtered = organised_data[ith_tclf][ith_improve][ith_clf]
                filtered = filtered[col_name]
                means = filtered.mean(0)

                full_name = '{}_{}_{}_{}'.format(type_dataset, ith_tclf, ith_improve, ith_clf)
                temp_arra = [full_name, ]
                new_arr.append((full_name, means[0], means[1], means[2], means[3]))
                temp_arra.extend(means)
                new_arr.append(temp_arra)
                #  print '\n{} report:\n{}'.format(full_name, means[:])
    guardar_csv(new_arr, 'recursos/resultados/clf_metricas_{}.csv'.format(type_dataset))
    print 'fin'


if __name__ == '__main__':
    structuriseresults('nongrams')
    #  saveandjoin30iter('ngrams')
