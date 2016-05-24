# -*- coding: utf-8 -*-
import cProfile, pstats, StringIO

from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn import svm, cross_validation
from sklearn.metrics import classification_report
import numpy as np

from utiles import contenido_csv, guardar_csv, binarizearray

__author__ = 'Juan David Carrillo López'

pr = cProfile.Profile()


def searchinghparameters(features_space):
    np.random.shuffle(features_space)
    min_max_scaler = MinMaxScaler()

    X = min_max_scaler.fit_transform(features_space[:, :features_space.shape[1] - 1])

    type_classifier = {'multi': None, 'binary': None}
    for type_clf in type_classifier.keys():
        if type_clf != 'binary':
            y = features_space[:, features_space.shape[1] - 1:].ravel()
        else:
            y = binarizearray(features_space[:, features_space.shape[1] - 1:].ravel())
        print '\t--------->>  {} - class type  <<---------'.format(type_clf)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

        tuned_parameters = [{'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
                             'cache_size': [300, 500]}]
        #  scores = ['precision', 'recall']
        scores = ['f1', ]
        for score in scores:
            print '# Tuning hyper-parameters for %s\n' % score
            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=10, scoring='%s_weighted' % score)
            clf.fit(x_train, y_train)

            print 'Best parameters set found on development set: \n'
            type_classifier[type_clf] = clf.best_params_
            print '{}\n'.format(clf.best_params_)
            print 'Grid scores on development set:\n'
            for params, mean_score, scores in clf.grid_scores_:
                print '%0.3f (+/-%0.03f) for %r\n' % (mean_score, scores.std() * 2, params)

            print 'Detailed classifications report:\n'
            print 'The model is trained on the full development set.'
            print 'The scores are computed on the hull evaluation set.\n'
            y_true, y_pred = y_test, clf.predict(x_test)
            print classification_report(y_true, y_pred)
    return type_classifier


def learningtoclassify(type_dataset, n_iter=1, data_set=[]):
    features_space = data_set
    np.random.shuffle(features_space)
    min_max_scaler = MinMaxScaler()

    new_params = searchinghparameters(data_set)
    for i_iter in range(n_iter):
        print '\titeration: {}'.format(i_iter + 1)
        #  training_set = features_space[:int(number_rows * .8)]
        #  valid_set = features_space[int(number_rows*.5)+1:int(number_rows*.8)]
        #  test_set = features_space[int(number_rows * .8) + 1:]

        type_classifier = {'multi': None, 'binary': None}

        x = min_max_scaler.fit_transform(features_space[:, :features_space.shape[1] - 1])

        kf_total = cross_validation.KFold(len(x), n_folds=10)
        for type_clf in type_classifier.keys():
            classifiers = {'Poly-2 Kernel': svm.SVC(**new_params[type_clf]), }
            general_metrics = {'Poly-2 Kernel': [[], [], [], []], }
            if type_clf == 'binary':
                y = np.array(binarizearray(features_space[:, features_space.shape[1] - 1:].ravel()))
            else:
                y = features_space[:, features_space.shape[1] - 1:].ravel()

            for train_ind, test_ind in kf_total:
                scaled_test_set = x[test_ind]
                for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                    pr.enable()
                    inst_clf = clf.fit(x[train_ind], y[train_ind])
                    pr.disable()
                    s = StringIO.StringIO()
                    sortby = 'cumulative'
                    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                    tt = round(ps.total_tt, 6)
                    print '------------------------------------>>>>   {} \n'.format(clf_name)
                    y_pred = clf.predict(scaled_test_set)
                    # y_true = y[test_ind]
                    if type_clf == 'multi':
                        y_true = np.random.random_integers(1, 3, test_ind.shape[0])
                    else:
                        y_true = np.random.random_integers(0, 1, test_ind.shape[0])
                    inst_clf.score(x[test_ind], y[test_ind])
                    general_metrics[clf_name][0].append(accuracy_score(y_true, y_pred))
                    general_metrics[clf_name][1].append(
                        np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
                    last_metric = '-'.join([str(elem) for elem in confusion_matrix(y_true, y_pred).ravel()])
                    general_metrics[clf_name][2].append(tt)
                    general_metrics[clf_name][3].append(last_metric)

            for clf_name in general_metrics.keys():
                array_a = np.expand_dims(np.array(general_metrics[clf_name][0]), axis=1)
                array_b = np.array(general_metrics[clf_name][1])
                array_c = np.expand_dims(np.array(general_metrics[clf_name][2]), axis=1)
                array_d = np.expand_dims(np.array(general_metrics[clf_name][3]), axis=1)
                try:
                    results = np.concatenate((array_a, array_b, array_c, array_d), axis=1)
                except ValueError as e:
                    print 'ERROR whilst saving {}_{}_hparamt_{}_{} metrics: {}'.\
                        format(type_dataset, type_clf, clf_name, i_iter, str(e))
                    pass
                else:
                    guardar_csv(results, 'recursos/resultados/experiment_tfidf/{}_{}_hparamt_{}_{}.csv'.
                                format(type_dataset, type_clf, clf_name, i_iter + 1))
                    print 'saved {}_{}_hparamt_{}_{} metrics'.format(type_dataset, type_clf, clf_name, i_iter + 1)
        # End of the classifier type iterations
    # End of the ith iterations


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

    print '\n--------------------------------------->>>>   SEARCHING HYPERPARAMETERS   ' \
          '<<<<-------------------------------------------'
    learningtoclassify(type_set.replace('rand_', ''), 30, data)
