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
            guardar_csv(results, 'recursos/resultados/{}/{}_eensemble_{}.csv'.format(type_dataset, type_clf, clf_name))


def plotmetric():
    data = {'multi':
                ((0.31609195402298851, 0.30747126436781608, 0.33908045977011492, 0.29022988505747127, 0.34482758620689657, 0.33620689655172414, 0.32183908045977011, 0.30747126436781608, 0.33620689655172414, 0.29597701149425287, 0.27873563218390807, 0.28448275862068967, 0.3045977011494253, 0.35057471264367818, 0.30747126436781608, 0.31034482758620691, 0.31896551724137934, 0.33045977011494254, 0.30172413793103448, 0.35344827586206895, 0.32471264367816094, 0.35057471264367818, 0.33620689655172414, 0.33045977011494254, 0.28160919540229884, 0.3045977011494253, 0.34195402298850575, 0.33908045977011492, 0.30172413793103448, 0.33333333333333331),
                 (0.32471264367816094, 0.5431034482758621, 0.5545977011494253, 0.31896551724137934, 0.57471264367816088, 0.47413793103448276, 0.50574712643678166, 0.58045977011494254, 0.58908045977011492, 0.58908045977011492, 0.50574712643678166, 0.57471264367816088, 0.56896551724137934, 0.57758620689655171, 0.51724137931034486, 0.56321839080459768, 0.56609195402298851, 0.52873563218390807, 0.49137931034482757, 0.60632183908045978, 0.46264367816091956, 0.54022988505747127, 0.55747126436781613, 0.51724137931034486, 0.50862068965517238, 0.55747126436781613, 0.56034482758620685, 0.50862068965517238, 0.52011494252873558, 0.53735632183908044),
                 (0.60344827586206895, 0.55747126436781613, 0.56321839080459768, 0.62643678160919536, 0.53735632183908044, 0.58620689655172409, 0.60632183908045978, 0.51436781609195403, 0.58908045977011492, 0.5, 0.53448275862068961, 0.51436781609195403, 0.58333333333333337, 0.56321839080459768, 0.46839080459770116, 0.50574712643678166, 0.53735632183908044, 0.55172413793103448, 0.53448275862068961, 0.46839080459770116, 0.47413793103448276, 0.59195402298850575, 0.57183908045977017, 0.5316091954022989, 0.52873563218390807, 0.51436781609195403, 0.52011494252873558, 0.49137931034482757, 0.54885057471264365, 0.4885057471264368),
                 (0.64367816091954022, 0.39655172413793105, 0.40229885057471265, 0.64367816091954022, 0.60919540229885061, 0.60344827586206895, 0.64367816091954022, 0.59195402298850575, 0.57183908045977017, 0.35919540229885055, 0.57471264367816088, 0.5, 0.40517241379310343, 0.55172413793103448, 0.45689655172413796, 0.63218390804597702, 0.61206896551724133, 0.25, 0.63218390804597702, 0.47413793103448276, 0.43678160919540232, 0.53448275862068961, 0.58620689655172409, 0.32183908045977011, 0.50287356321839083, 0.46551724137931033, 0.56321839080459768, 0.46264367816091956, 0.43390804597701149, 0.4942528735632184),
                 (0.31321839080459768, 0.58333333333333337, 0.62643678160919536, 0.28160919540229884, 0.62643678160919536, 0.60632183908045978, 0.64367816091954022, 0.5977011494252874, 0.62356321839080464, 0.59482758620689657, 0.56896551724137934, 0.60344827586206895, 0.61781609195402298, 0.63218390804597702, 0.61781609195402298, 0.60919540229885061, 0.57758620689655171, 0.5977011494252874, 0.60632183908045978, 0.62931034482758619, 0.5316091954022989, 0.61781609195402298, 0.61781609195402298, 0.60344827586206895, 0.52873563218390807, 0.59195402298850575, 0.5977011494252874, 0.55172413793103448, 0.52298850574712641, 0.63505747126436785)),
            'binary':
                ((0.66954022988505746, 0.73275862068965514, 0.7068965517241379, 0.71264367816091956, 0.63793103448275867, 0.66091954022988508, 0.67241379310344829, 0.63505747126436785, 0.6954022988505747, 0.70402298850574707, 0.7183908045977011, 0.69252873563218387, 0.64942528735632188, 0.66091954022988508, 0.67528735632183912, 0.73275862068965514, 0.67816091954022983, 0.7183908045977011, 0.72988505747126442, 0.74425287356321834, 0.72413793103448276, 0.67241379310344829, 0.71551724137931039, 0.6522988505747126, 0.70402298850574707, 0.7183908045977011, 0.69827586206896552, 0.70402298850574707, 0.68678160919540232, 0.69252873563218387),
                 (0.58908045977011492, 0.60919540229885061, 0.66091954022988508, 0.66379310344827591, 0.61781609195402298, 0.62068965517241381, 0.59195402298850575, 0.58333333333333337, 0.64942528735632188, 0.64942528735632188, 0.60057471264367812, 0.44827586206896552, 0.56034482758620685, 0.57471264367816088, 0.62931034482758619, 0.69827586206896552, 0.66954022988505746, 0.59195402298850575, 0.51724137931034486, 0.6954022988505747, 0.70114942528735635, 0.58620689655172409, 0.67816091954022983, 0.34770114942528735, 0.66954022988505746, 0.62931034482758619, 0.68103448275862066, 0.67816091954022983, 0.61494252873563215, 0.62931034482758619),
                 (0.66954022988505746, 0.8045977011494253, 0.79885057471264365, 0.77298850574712641, 0.77011494252873558, 0.66091954022988508, 0.67241379310344829, 0.63505747126436785, 0.6954022988505747, 0.70402298850574707, 0.7816091954022989, 0.69252873563218387, 0.75862068965517238, 0.80747126436781613, 0.77586206896551724, 0.7816091954022989, 0.7068965517241379, 0.77011494252873558, 0.8045977011494253, 0.74425287356321834, 0.78735632183908044, 0.67241379310344829, 0.71551724137931039, 0.6522988505747126, 0.75862068965517238, 0.7614942528735632, 0.69827586206896552, 0.79022988505747127, 0.76436781609195403, 0.69252873563218387),
                 (0.7183908045977011, 0.79885057471264365, 0.79597701149425293, 0.81034482758620685, 0.74712643678160917, 0.77298850574712641, 0.72126436781609193, 0.71551724137931039, 0.73275862068965514, 0.74425287356321834, 0.7931034482758621, 0.6954022988505747, 0.76724137931034486, 0.80747126436781613, 0.76436781609195403, 0.77873563218390807, 0.77586206896551724, 0.77298850574712641, 0.79597701149425293, 0.77873563218390807, 0.77586206896551724, 0.7183908045977011, 0.74712643678160917, 0.66091954022988508, 0.76436781609195403, 0.7614942528735632, 0.72988505747126442, 0.7816091954022989, 0.76436781609195403, 0.72988505747126442),
                 (0.67241379310344829, 0.70402298850574707, 0.68103448275862066, 0.71264367816091956, 0.63218390804597702, 0.56609195402298851, 0.67241379310344829, 0.62356321839080464, 0.68965517241379315, 0.66954022988505746, 0.70977011494252873, 0.35344827586206895, 0.56896551724137934, 0.58333333333333337, 0.67241379310344829, 0.7068965517241379, 0.69252873563218387, 0.62356321839080464, 0.59482758620689657, 0.73275862068965514, 0.72413793103448276, 0.65804597701149425, 0.70402298850574707, 0.32471264367816094, 0.68678160919540232, 0.72126436781609193, 0.68965517241379315, 0.69252873563218387, 0.67528735632183912, 0.68390804597701149),
                 (0.26724137931034481, 0.28448275862068967, 0.2557471264367816, 0.25, 0.29022988505747127, 0.2471264367816092, 0.27298850574712646, 0.27586206896551724, 0.37931034482758619, 0.42528735632183906, 0.43678160919540232, 0.39942528735632182, 0.25, 0.23850574712643677, 0.4454022988505747, 0.29310344827586204, 0.36206896551724138, 0.52586206896551724, 0.32183908045977011, 0.2614942528735632, 0.27011494252873564, 0.33620689655172414, 0.27586206896551724, 0.38793103448275862, 0.3045977011494253, 0.31034482758620691, 0.3045977011494253, 0.4885057471264368, 0.31321839080459768, 0.48563218390804597))
    }
    #  stats = cbook.boxplot_stats(np.array(data['multi']).T)
    plt.subplot()
    plt.boxplot(np.array(data['binary']).T)
    plt.show()


def machinelearning(type_set):
    data = contenido_csv('recursos/{}.csv'.format(type_set))
    print '\n--------------------------------------->>>>   EASY ENSEMBLE UNDERSAMPLING   ' \
          '<<<<-------------------------------------------'
    learningtoclassify(type_set, 30, np.array(data, dtype='f'))
