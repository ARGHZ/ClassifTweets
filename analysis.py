# -*- coding: utf-8 -*-
import random
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from statsmodels.stats.weightstats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utiles import contenido_csv, leerarchivo, guardararchivo, guardar_csv, binarizearray

__author__ = 'Juan David Carrillo López'

SITE_ROOT = os.path.dirname(os.path.realpath(__file__))


def makeshortname(actual_name):
    part = [elem for elem in actual_name.split('_')]
    middle_part = ''.join([elem[0] for elem in part[2:len(part)]])
    return '{}{}'.format(part[0][1].upper(), middle_part.upper())


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
                    try:
                        for ith_iter in range(1, 31):
                            csvfile_path = 'recursos/resultados/experiment_tfidf/{}_{}_{}_{}_{}.csv'. \
                                format(type_dataset, t_clf, t_imprv, clf_name, ith_iter)

                            results = contenido_csv(csvfile_path)
                            temp_array = np.array(results)
                            results = np.array(temp_array[:, :temp_array.shape[1] - 1], dtype='f')
                            confussion_m = np.array([str(row).split('-') for row in
                                                     np.nditer(temp_array[:, temp_array.shape[1] - 1:])], dtype='i')
                            confussion_m = np.around(list(confussion_m.mean(axis=0)), decimals=1)
                            statistics = [results[:, col].ravel().mean() for col in range(results.shape[1])]
                            statistics.extend(confussion_m)
                            print 'Renderizing: {}'.format(csvfile_path)
                            new_arr.append(statistics)
                    except IOError:
                        print '\nArchivo no encontrado: {}_{}_{}_{}_{}.csv'.\
                            format(type_dataset, t_clf, t_imprv, clf_name, ith_iter)
                    else:
                        new_arr = np.array(new_arr)
                        guardar_csv(new_arr, 'recursos/resultados/experiment_tfidf/{}_{}_{}_{}.csv'.
                                    format(type_dataset, t_clf, t_imprv, clf_name))
                        print 'recursos/resultados/experiment_tfidf/{}_{}_{}_{}'.\
                            format(type_dataset, t_clf, t_imprv, clf_name)


def getalldata(xperiment_name):
    datasets = {'ngrams': None, 'nongrams': None}
    for data_type in datasets.keys():
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
                             'F1-class 1', 'F1-class 2', 'support class 1', 'support class 2', 'time']
            else:
                col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'P-class 3', 'R-class 1',
                             'R-class 2', 'R-class 3', 'F1-class 1', 'F1-class 2', 'F1-class 3',
                             'support class 1', 'support class 2', 'support class 3', 'time']
            for t_imprv in data_experiment[t_clf].keys():
                for clf_name in data_experiment[t_clf][t_imprv].keys():
                    csvfile_path = 'recursos/resultados/{}/{}_{}_{}_{}.csv'.\
                        format(xperiment_name, data_type, t_clf, t_imprv, clf_name)
                    #  print 'Getting: {}'.format(csvfile_path)
                    results = np.array(contenido_csv(csvfile_path))
                    results = np.array(results[:, :len(col_names)], dtype='f')
                    try:
                        data_experiment[t_clf][t_imprv][clf_name] = pd.DataFrame(results, columns=col_names)
                    except ValueError as e:
                        print 'ERROR whilst saving metrics on {}_{}_{}: {}'.format(t_clf, t_imprv, clf_name, e)
        datasets[data_type] = data_experiment
    return datasets


def getelitedata(specific_combinations, results_path):
    eliteclf_data = {}
    for combination in specific_combinations:
        csvfile_path = '{}/{}.csv'.format(results_path, combination)
        try:
            type_classifier = combination.split('_')[1]
            if type_classifier == 'binary':
                col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'R-class 1', 'R-class 2',
                             'F1-class 1', 'F1-class 2', 'support class 1', 'support class 2']
            else:
                col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'P-class 3', 'R-class 1',
                             'R-class 2', 'R-class 3', 'F1-class 1', 'F1-class 2', 'F1-class 3',
                             'support class 1', 'support class 2', 'support class 3']

            results = contenido_csv(csvfile_path)
        except IOError:
            print '\nArchivo {} no encontrado'.format(csvfile_path)
        else:
            results = np.array(results)
            results = np.array(results[:, :len(col_names)], dtype='f')
            eliteclf_data[combination] = pd.DataFrame(results, columns=col_names)
    return eliteclf_data


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


def showstatsresults(prefx='', charts_path='recursos/charts/'):
    '''elite_clfs = ('nongrams_multi_kfolds_Poly-2 Kernel', 'nongrams_binary_kfolds_Poly-2 Kernel',
                  'nongrams_binary_ros_AdaBoost', 'nongrams_binary_ros_GradientBoosting',
                  'ngrams_binary_ros_AdaBoost', 'ngrams_multi_ros_AdaBoost',
                  'nongrams_multi_ros_AdaBoost', 'nongrams_multi_ros_GradientBoosting')
                  '''
    elite_clfs = ('nongrams_multi_kfolds_Poly-2 Kernel', 'nongrams_binary_kfolds_Poly-2 Kernel',
                  'ngrams_binary_eensemble_Poly-2 Kernel', 'ngrams_binary_eensemble_GradientBoosting',
                  'nongrams_binary_eensemble_Poly-2 Kernel', 'ngrams_multi_eensemble_Poly-2 Kernel',
                  'ngrams_multi_eensemble_GradientBoosting', 'nongrams_multi_eensemble_GradientBoosting')
    all_metrics = getelitedata(elite_clfs, 'recursos/resultados/experiment_tfidf')
    metrics_candidate = {}
    for clf_name in elite_clfs[2:]:
        if 'binary' in clf_name:
            col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'R-class 1', 'R-class 2',
                         'F1-class 1', 'F1-class 2']
        else:
            col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'P-class 3', 'R-class 1',
                         'R-class 2', 'R-class 3', 'F1-class 1', 'F1-class 2', 'F1-class 3']
        metrics_candidate[clf_name] = all_metrics[clf_name][col_names]

    metrics_main = {}
    for clf_name in elite_clfs[:2]:
        if 'binary' in clf_name:
            col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'R-class 1', 'R-class 2',
                         'F1-class 1', 'F1-class 2']
        else:
            col_names = ['Accuracy', 'P-class 1', 'P-class 2', 'P-class 3', 'R-class 1',
                         'R-class 2', 'R-class 3', 'F1-class 1', 'F1-class 2', 'F1-class 3']
        metrics_main[clf_name] = all_metrics[clf_name][col_names]

    for m_clf_name in metrics_main.keys():
        clf_case = m_clf_name.split('_')[1]
        if 'binary' in m_clf_name:
            selected_col = {'Accuracy': 0, 'F1-class 1': 5, 'F1-class 2': 6}
            base_metric, confusion_m = 'F1-class 2', ['CM-1', 'CM-2', 'CM-3', 'CM-4']
        else:
            selected_col = {'Accuracy': 0, 'F1-class 1': 7, 'F1-class 2': 8, 'F1-class 3': 9}
            base_metric, confusion_m = 'F1-class 3', ['CM-1', 'CM-2', 'CM-3', 'CM-4', 'CM-5',
                                                      'CM-6', 'CM-7', 'CM-8', 'CM-9']
        m_clf_nshort = makeshortname(m_clf_name)
        print '\n{}'.format(m_clf_name)
        elit_metrics = []
        '''query = metrics_main[m_clf_name][base_metric] == metrics_main[m_clf_name][base_metric].max()
        query_result = metrics_main[m_clf_name][query][confusion_m].values
        best_confussion_m = [round(val, 1) for val in np.nditer(query_result)]

        elit_metrics.append((m_clf_nshort, list(metrics_main[m_clf_name].mean(0).values), best_confussion_m))
        '''
        filtered_data = {'1_{}'.format(m_clf_nshort): metrics_main[m_clf_name][selected_col.keys()]}

        cont = 2
        for c_clf_name in metrics_candidate.keys():
            if clf_case in c_clf_name:
                print '\n\t{}'.format(c_clf_name)
                for selct_metric in selected_col.keys():
                    stats_result = ttest_ind(metrics_main[m_clf_name][selct_metric].values,
                                             metrics_candidate[c_clf_name][selct_metric].values)
                    msg_result = '\t{} -- test statisic: {} \tpvalue of the t-test: {} ' \
                                 '\tdegrees of freedom used in the t-test: {}'. \
                        format(selct_metric, stats_result[0], stats_result[1],
                               stats_result[2])
                    if stats_result[1] > 0.05:
                        msg_result += ' P-value mayor a 0.05'
                    print '\t{}'.format(msg_result)

                '''query = metrics_candidate[c_clf_name][base_metric] == metrics_candidate[c_clf_name][base_metric].max()
                query_result = metrics_candidate[c_clf_name][query][confusion_m].values
                best_confussion_m = [round(val, 1) for val in np.nditer(query_result)]
                '''
                filtered_data['{}_{}'.format(cont, makeshortname(c_clf_name))] = metrics_candidate[c_clf_name]
                '''elit_metrics.append((makeshortname(c_clf_name), list(metrics_candidate[c_clf_name].mean(0).values),
                                     best_confussion_m))
                                     '''
                cont += 1
        #  guardar_csv(elit_metrics, 'recursos/resultados/experiment_tfidf/elite_{}_metrics.csv'.format(clf_case))

        for selct_metric in selected_col.keys():
            data_labels, data_toplot = [], []
            plt.subplot()
            for clf_name in filtered_data.keys():
                data_toplot.append(filtered_data[clf_name][selct_metric])
                if clf_name[2:][0] == 'O':
                    clf_name = 'N{}'.format(clf_name[3:])
                else:
                    clf_name = 'L{}'.format(clf_name[3:])
                data_labels.append(clf_name)
            plt.boxplot(data_toplot)

            if 'Accuracy' in selct_metric:
                selct_metric = 'Exactitud'
            else:
                selct_metric = 'F-score_Clase_{}'.format(selct_metric[-1])
            plt.xticks(np.arange(0, len(data_toplot)) + 1, data_labels)
            [plt.savefig('{}/{}{}_{}_{}.{}'.format(charts_path, clf_case, selct_metric, clf_case, img_format))
             for img_format in ('eps', 'jpg')]
            plt.close()


def makettest(xperiment_dir):
    datasets, methods, classes, classifiers = ('ngrams', 'nongrams'), ('kfolds', 'eensemble', 'ros', 'hparamt'), \
                                              ('binary', 'multi'), ('Poly-2 Kernel', 'AdaBoost', 'GradientBoosting')
    all_metrics = {}
    for type_data in datasets:
        for class_type in classes:
            for method in methods:
                for clf in classifiers:
                    try:
                        clasifier_name = '{}_{}_{}_{}'.format(type_data, class_type, method, clf)
                        # To Select only: accuracy, precision, recall, f-score, support classes and time
                        if class_type == 'multi':
                            limit_cols = 14
                        else:
                            limit_cols = 10
                        metrics_vect = contenido_csv('{}/{}_{}_{}_{}.csv'.format(xperiment_dir, type_data, class_type,
                                                                                 method, clf))
                        metrics_vect = np.array(metrics_vect)[:, :limit_cols]
                        all_metrics[clasifier_name] = metrics_vect
                    except IOError as e:
                        pass
                    else:
                        print '{}/{}/{}_{}_{}_{}'. \
                            format(SITE_ROOT, xperiment_dir, type_data, class_type, method, clf)
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
                                 '\tdegrees of freedom used in the t-test: {}'. \
                        format(ith_col + 1, round(stats_result[0], 4), round(stats_result[1], 4),
                               round(stats_result[2]), 4)
                    if stats_result[1] > 0.05:
                        msg_result += ' P-value mayor a 0.05'
                    print msg_result
                    #  print '\tclass {} -- {}\n\n'.format(ith_col + 1, ','.join(current_clf[:, ith_col].ravel()))


def datasetstats():
    data_set = np.concatenate((np.array(contenido_csv('recursos/resultados/evaluated_data.csv'), dtype=np.int32),
                               np.array(contenido_csv('recursos/resultados/weighted_data.csv'), dtype=np.int32)),
                              axis=1)
    main_data, added_data = data_set[:2000, :], data_set[2000:, :]

    '''book = XLSXBook('recursos/ponderacion/conjuntos.xlsx')
    content = book.sheets()
    data_set = []

    for row in content['filtro'][1:]:
        data_set.append((int(row[1]), int(row[2])))

    guardar_csv(data_set, 'recursos/resultados/evaluated_data.csv')
    '''

    filtro = np.array([row for row in data_set if row[3] <= 2])
    rangos_3, rangos_2 = [], []
    for valor_selecc in filtro[:, 2]:
        if valor_selecc < 4:
            rangos_3.append(1)
            rangos_2.append(1)
        elif valor_selecc > 6:
            rangos_3.append(3)
            rangos_2.append(2)
        else:
            rangos_3.append(2)
            rangos_2.append(1)

    font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 16}

    print 'Dataset \n\t ev_1 {} \n\t ev_2: {}'.format(np.bincount(main_data[:, 0]), np.bincount(main_data[:, 1]))
    plt.hist(main_data[:, 0], color='g', label='Evaluador 1', alpha=0.4)
    plt.hist(main_data[:, 1], color='y', label='Evaluador 2', alpha=0.4)
    plt.title(u'Conjunto de Datos')
    plt.xlabel(u'Escala de Ponderación')
    plt.ylabel('Cantidad de muestras')
    plt.xticks([i + 0.5 for i in range(11)], [j for j in range(11)])
    plt.legend(loc='upper right')
    #  plt.show()
    plt.savefig('recursos/charts/hist_dataset.jpeg')
    plt.close()
    print 'Added data \n\t ev_1 {} \n\t ev_2: {}'.format(np.bincount(added_data[:, 0]), np.bincount(added_data[:, 1]))
    plt.hist(added_data[:, 0], color='g', label='Evaluador 1', alpha=0.4)
    plt.hist(added_data[:, 1], color='y', label='Evaluador 2', alpha=0.4)
    plt.title(u'Porción añadida al conjunto de datos')
    plt.xlabel(u'Escala de Ponderación')
    #  plt.xticks([i + 0.5 for i in range(10)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    #  plt.show()
    plt.savefig('recursos/charts/hist_addedset.jpeg')
    plt.close()
    print 'Filtro <= 2: \n\t ev_1 {} \n\t ev_2: {}'.format(np.bincount(filtro[:, 0]), np.bincount(filtro[:, 1]))
    plt.hist(filtro[:, 0], color='g', label='Evaluador 1', alpha=0.4)
    plt.hist(filtro[:, 1], color='y', label='Evaluador 2', alpha=0.4)
    plt.title(u'Conjunto de Datos Filtrado')
    plt.xlabel(u'Escala de Ponderación')
    plt.xticks([i + 0.5 for i in range(11)], [j for j in range(11)])
    plt.legend(loc='upper right')
    plt.savefig('recursos/charts/hist_filtro.jpeg')
    plt.close()
    print 'Final counting \n\t {}'.format(np.bincount(filtro[:, 2]))
    plt.hist(filtro[:, 2], color='y', alpha=0.4)
    plt.title(u'Ponderación Unificado')
    plt.xlabel(u'Escala de Ponderación')
    plt.xticks([i + 0.5 for i in range(11)], [j for j in range(11)])
    plt.legend(loc='upper right')
    plt.savefig('recursos/charts/hist_finalcounting.jpeg')
    plt.close()
    '''
    rangos = np.array((rangos_2, rangos_3))
    rangos = rangos.T
    guardar_csv(rangos, 'recursos/resultados/labelled_dataset.csv')
    print '3-class counting \n\t {}'.format(np.bincount(rangos_3))
    '''
    plt.hist(filtro[:, 2], color='y', alpha=0.4, bins=3)
    plt.title(u'Conjunto de 3 clases')
    plt.xlabel(u'Escala de Ponderación')
    plt.xticks([i + 0.5 for i in range(11)], [j + 1 for j in range(10)])
    plt.legend(loc='upper right')
    plt.savefig('recursos/charts/hist_3-class.jpeg')
    plt.close()


def metricsofrandomsample(type_set, n_samples=30):
    for type_clf in ('binary', 'multi'):
        data = np.array(contenido_csv('recursos/{}.csv'.format(type_set)), dtype='f')[:, 4]
        np.random.shuffle(data)
        number_rows = data.shape[0]
        test_set = data[int(number_rows * .8) + 1:].ravel()

        rand_labels = np.array(contenido_csv('recursos/rand_labelling.csv'), dtype='f')[0]
        rand_labels = rand_labels[:test_set.shape[0] * n_samples]

        rand_labellings = []
        for rand_val in rand_labels:
            rand_val = int(rand_val)
            if rand_val < 4:
                rand_labellings.append(1)
            elif rand_val > 6:
                rand_labellings.append(3)
            else:
                rand_labellings.append(2)

        if type_clf == 'binary':
            test_set = binarizearray(test_set)
            rand_labellings = binarizearray(rand_labellings)
        rand_labellings = np.split(np.array(rand_labellings), n_samples)

        y_labels = {'true': test_set, }
        metrics = {}
        for i in range(n_samples):
            y_labels['{}'.format(i + 1)] = []
            metrics['{}'.format(i + 1)] = []
        for ith_sample in range(n_samples):
            y_labels[str(ith_sample + 1)] = rand_labellings[ith_sample]
            general_metric = [accuracy_score(test_set, rand_labellings[ith_sample])]
            specific_metric = np.array(
                    precision_recall_fscore_support(test_set, rand_labellings[ith_sample])[:3]).ravel()
            temp_arr = np.concatenate((general_metric, specific_metric))
            metrics[str(ith_sample + 1)] = temp_arr
        y_labels, metrics = pd.DataFrame(y_labels), pd.DataFrame(metrics)
        y_labels.to_csv('recursos/resultados/{}_{}_y_labels_rand.csv'.format(type_set, type_clf))
        metrics.T.to_csv('recursos/resultados/{}_{}_y_labels_rand_metrics.csv'.format(type_set, type_clf))


def analyserandomlabellings():

    for type_dataset in ('ngrams', 'nongrams'):
        for type_clf in ('binary', 'multi'):
            file_base_path = 'recursos/resultados/labels_analysis/{}_{}_y_labels_rand_metrics.csv'.\
                format(type_dataset, type_clf)
            all_metrics = pd.read_csv(file_base_path, index_col='iters')
            print '\n{}\n{}\n{}'.format(file_base_path, all_metrics, all_metrics.mean(0))

            for column in all_metrics:
                plt.subplot()
                data_toplot = all_metrics[column].values

                plt.boxplot(data_toplot)
                plt.xticks([1], [column.title()])
                # plt.show()
                [plt.savefig('recursos/charts/random_sampling/{}_{}_{}_random.{}'.
                             format(type_dataset, type_clf, column, img_format))
                 for img_format in ('eps', 'jpg')]
                plt.close()

    print ''


if __name__ == '__main__':
    '''random_labels = np.asarray(np.random.random_integers(0, 10, 30000), dtype=np.str_)
    random_labels = (','.join(random_labels), )
    guardararchivo(random_labels, 'recursos/random_numbers.txt')

    data_features = np.array(contenido_csv('recursos/nongrams.csv'))

    random_labels = []
    for rand_val in leerarchivo('recursos/random_numbers.txt')[0].split(',')[:data_features.shape[0]]:
        rand_val = int(rand_val)
        if rand_val < 4:
            random_labels.append(1)
        elif rand_val > 6:
            random_labels.append(3)
        else:
            random_labels.append(2)
    random_labels = np.array(random_labels).reshape((len(random_labels), 1))
    new_data = np.concatenate((data_features, random_labels), axis=1)
    guardar_csv(new_data, 'recursos/rand_nongrams.csv')
    '''
    # getalldata('experiment_tfidf')
    # structuriseresults('experiment_tfidf')
    # saveandjoin30iter('nongrams')
    # makettest('recursos/resultados/experiment_tfidf')
    # showstatsresults(prefx='', charts_path='recursos/charts/tfidf_vector')
    # datasetstats()
    # metricsofrandomsample('nongrams')
    analyserandomlabellings()
