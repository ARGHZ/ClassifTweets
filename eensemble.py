# -*- coding: utf-8 -*-
import cProfile, pstats, StringIO

from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from unbalanced_dataset.ensemble_sampling import EasyEnsemble
import numpy as np

from utiles import contenido_csv, guardar_csv, votingoutputs, binarizearray

__author__ = 'Juan David Carrillo López'

pr = cProfile.Profile()


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
        prediction = {'Poly-2 Kernel': [[], []], 'AdaBoost': [[], []], 'GradientBoosting': [[], []]}
        general_metrics = {'Poly-2 Kernel': [[], [], [], []], 'AdaBoost': [[], [], [], []],
                           'GradientBoosting': [[], [], [], []]}

        for i_iter in range(n_iter):
            np.random.shuffle(features_space)
            min_max_scaler = MinMaxScaler()
            print '\titeration: {}'.format(i_iter + 1)
            training_set = features_space[:int(number_rows * .8)]
            #  valid_set = features_space[int(number_rows*.5)+1:int(number_rows*.8)]
            test_set = features_space[int(number_rows * .8) + 1:]

            x = min_max_scaler.fit_transform(training_set[:, :features_space.shape[1] - 1])
            if type_clf == 'multi':
                y = training_set[:, features_space.shape[1] - 1:].ravel()
                y_true = test_set[:, features_space.shape[1] - 1:].ravel()
            else:
                y = np.array(binarizearray(training_set[:, features_space.shape[1] - 1:].ravel()))
                y_true = binarizearray(test_set[:, features_space.shape[1] - 1:].ravel())
            easyens = EasyEnsemble(verbose=True)
            eex, eey = easyens.fit_transform(x, y)

            ciclo, target_names = 0, ('class 1', 'class 2', 'class 3')
            #  for train_ind, test_ind in kf_total:
            for i_ee in range(len(eex)):
                scaled_test_set = min_max_scaler.fit_transform(test_set[:, :features_space.shape[1] - 1])
                #  print 'Subset {}'.format(ciclo)
                for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                    pr.enable()
                    clf.fit(eex[i_ee], eey[i_ee])
                    pr.disable()
                    s = StringIO.StringIO()
                    sortby = 'cumulative'
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    tt = round(ps.total_tt, 6)
                    #  print '------------------------------------>>>>   {} \n{}\n'.format(clf_name)
                    y_pred = clf.predict(scaled_test_set)
                    prediction[clf_name][0].append(y_pred)
                    prediction[clf_name][1].append(tt)
                ciclo += 1
            # End of the ith_subsamples

            for clf_name, (output, t_times) in prediction.items():
                all_ypred = np.array(output, dtype=int)
                y_pred = votingoutputs(all_ypred)[:, 1].ravel()
                if type_clf == 'multi':
                    y_true = np.random.random_integers(1, 3, len(y_true))
                else:
                    y_true = np.random.random_integers(0, 1, len(y_true))
                general_metrics[clf_name][0].append(accuracy_score(y_true, y_pred))
                tri_metrics = np.array(precision_recall_fscore_support(y_true, y_pred)).ravel()
                general_metrics[clf_name][1].append(tri_metrics)
                last_metric = '-'.join([str(elem) for elem in confusion_matrix(y_true, y_pred).ravel()])
                general_metrics[clf_name][2].append(np.array(t_times).mean())
                general_metrics[clf_name][3].append(last_metric)
        #  End i_ter cycle

        for clf_name in general_metrics.keys():
            array_a = np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1)
            array_b = np.array(general_metrics[clf_name][1])
            array_c = np.expand_dims(np.array(general_metrics[clf_name][2]), axis=1)
            array_d = np.expand_dims(np.array(general_metrics[clf_name][3]), axis=1)
            try:
                results = np.concatenate((array_a, array_b, array_c, array_d), axis=1)
            except ValueError as e:
                print 'ERROR whilst saving {}_{}_eensemble_{}_{} metrics: {}'.\
                    format(type_dataset, type_clf, clf_name, i_iter, str(e))
                pass
            else:
                guardar_csv(results, 'recursos/resultados/experiment_tfidf/{}_{}_eensemble_{}_{}.csv'.
                            format(type_dataset, type_clf, clf_name, i_iter + 1))
                print 'saved {}_{}_eensemble_{}_{} metrics'.format(type_dataset, type_clf, clf_name, i_iter + 1)
    # End of type of classifier iteration


def machinelearning(type_set):
    if 'rand' in type_set:
        data = np.array(contenido_csv('recursos/{}.csv'.format(type_set)), dtype='f')
        data = np.delete(data, data.shape[1] - 2, 1)  # removing the examiner gradeing
    else:
        data = np.array(contenido_csv('recursos/{}.csv'.format(type_set)), dtype='f')
        # replacing tfidf vectorial sum by each tfidf vector
        data = np.delete(data, 0, 1)
        tfidf_vects = np.array(contenido_csv('recursos/tfidf_vectors.csv'.format(type_set)), dtype='f')
        data = np.concatenate((tfidf_vects, data), axis=1)

    print '\n--------------------------------------->>>>   EASY ENSEMBLE UNDERSAMPLING   ' \
          '<<<<-------------------------------------------'
    learningtoclassify(type_set.replace('rand_', ''), 30, data)
