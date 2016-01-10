# -*- coding: utf-8 -*-
import random
import os

from pyexcel_xlsx import XLSXBook
from statsmodels.stats.weightstats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utiles import contenido_csv, guardar_csv

__author__ = 'Juan David Carrillo López'

SITE_ROOT = os.path.dirname(os.path.realpath(__file__))


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


def showresultstatistics():
    '''book = XLSXBook('recursos/ponderacion/conjuntos.xlsx')
    content = book.sheets()
    data_set = []

    for row in content['filtro'][1:]:
        data_set.append((int(row[1]), int(row[2])))

    guardar_csv(data_set, 'recursos/resultados/evaluated_data.csv')
    '''
    data_set = np.concatenate((np.array(contenido_csv('recursos/resultados/evaluated_data.csv'), dtype=np.int32),
                               np.array(contenido_csv('recursos/resultados/weighted_data.csv'), dtype=np.int32)),
                              axis=1)
    main_data, added_data = data_set[:2000, :], data_set[2000:, :]

    filtro = np.array([row for row in data_set if row[3] <= 2])
    rangos_3 = []
    for valor_selecc in filtro[:, 2]:
        if valor_selecc < 4:
            rangos_3.append(1)
        elif valor_selecc > 6:
            rangos_3.append(3)
        else:
            rangos_3.append(2)

    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}

    '''print 'Dataset \n\t ev_1 {} \n\t ev_2: {}'.format(np.bincount(main_data[:, 0]), np.bincount(main_data[:, 1]))
    plt.hist(main_data[:, 0], color='g', label='Evaluador 1', alpha=0.4)
    plt.hist(main_data[:, 1], color='y', label='Evaluador 2', alpha=0.4)
    plt.title(u'Titulo del gráfico')
    plt.xlabel(u'Escala de Ponderación')
    plt.ylabel('Cantidad de muestras')
    plt.xticks([i + 0.5 for i in range(10)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    #  plt.show()
    plt.savefig('recursos/charts/hist_dataset.jpeg')
    plt.close()'''
    print 'Added data \n\t ev_1 {} \n\t ev_2: {}'.format(np.bincount(added_data[:, 0]), np.bincount(added_data[:, 1]))
    plt.hist(added_data[:, 0], color='g', label='Evaluador 1', alpha=0.4)
    plt.hist(added_data[:, 1], color='y', label='Evaluador 2', alpha=0.4)
    plt.title(u'Titulo del gráfico')
    plt.xlabel(u'Escala de Ponderación')
    #  plt.xticks([i + 0.5 for i in range(10)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    plt.show()
    '''plt.savefig('recursos/charts/hist_addedset.jpeg')
    plt.close()
    print 'Filtro <= 2: \n\t ev_1 {} \n\t ev_2: {}'.format(np.bincount(filtro[:, 0]), np.bincount(filtro[:, 1]))
    plt.hist(filtro[:, 0], color='g', label='Evaluador 1', alpha=0.4)
    plt.hist(filtro[:, 1], color='y', label='Evaluador 2', alpha=0.4)
    plt.title(u'Titulo del gráfico')
    plt.xlabel(u'Escala de Ponderación')
    plt.xticks([i + 0.5 for i in range(10)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    plt.savefig('recursos/charts/hist_filtro.jpeg')
    plt.close()
    print 'Final counting \n\t {}'.format(np.bincount(filtro[:, 2]))
    plt.hist(filtro[:, 2], color='y', alpha=0.4)
    plt.title(u'Titulo del gráfico')
    plt.xlabel(u'Escala de Ponderación')
    plt.xticks([i + 0.5 for i in range(10)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    plt.savefig('recursos/charts/hist_finalcounting.jpeg')
    plt.close()
    print '3-class counting \n\t {}'.format(np.bincount(rangos_3))
    plt.hist(filtro[:, 2], color='y', alpha=0.4, bins=3)
    plt.title(u'Titulo del gráfico')
    plt.xlabel(u'Escala de Ponderación')
    plt.xticks([i + 0.5 for i in range(10)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    plt.savefig('recursos/charts/hist_3-class.jpeg')
    plt.close()
    '''
    datasets, methods, classes, classifiers = ('ngrams', 'nongrams'), ('kfolds', 'eensemble', 'ros', 'hparamt'),\
                                              ('binary', 'multi'), ('Poly-2 Kernel', 'AdaBoost', 'GradientBoosting')
    all_metrics = {}
    for type_data in datasets:
        for class_type in classes:
            for method in methods:
                for clf in classifiers:
                    try:
                        clasifier_name = '{}_{}_{}_{}'.format(type_data, class_type, method, clf)
                        all_metrics[clasifier_name] = (contenido_csv('recursos/resultados/experiment_a/{}/{}_{}_{}.csv'.
                                                       format(type_data, class_type, method, clf)))
                    except IOError as e:
                        pass
                    else:
                        print '{}/recursos/resultados/experiment_a/{}/{}_{}_{}'.\
                            format(SITE_ROOT, type_data, class_type, method, clf)
    fscore_idx = {'binary': [5, 6], 'multi': [7, 8, 9]}
    for class_type in classes:
        base_clf_name = 'nongrams_{}_kfolds_Poly-2 Kernel'.format(class_type)
        base_clf = np.array(all_metrics[base_clf_name], dtype='f')[:, fscore_idx[class_type]]
        print base_clf_name
        '''for ith_col in range(base_clf.shape[1]):
            print '\tclass {} -- {}'.format(ith_col + 1, ','.join(base_clf[:, ith_col].ravel()))
            '''
        for clasifier_name in all_metrics.keys():
            if clasifier_name != base_clf_name and class_type in clasifier_name:
                current_clf = np.array(all_metrics[clasifier_name], dtype='f')[:, fscore_idx[class_type]]
                print '\n\t{}'.format(clasifier_name)
                for ith_col in range(current_clf.shape[1]):
                    stats_result = ttest_ind(base_clf[:, ith_col].ravel(), current_clf[:, ith_col].ravel())
                    msg_result = '\tclass {} -- test statisic: {} \tpvalue of the t-test: {} ' \
                                 '\tdegrees of freedom used in the t-test: {}'.\
                        format(ith_col + 1, stats_result[0], stats_result[1], stats_result[2])
                    if stats_result[1] > 0.05:
                        msg_result += ' P-value mayor a 0.05'
                    print msg_result
                    #  print '\tclass {} -- {}\n\n'.format(ith_col + 1, ','.join(current_clf[:, ith_col].ravel()))


if __name__ == '__main__':
    #  structuriseresults('nongrams')
    #  saveandjoin30iter('ngrams')
    showresultstatistics()
