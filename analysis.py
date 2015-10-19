# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from utiles import contenido_csv, guardar_csv

__author__ = 'Juan David Carrillo López'

if __name__ == '__main__':
    type_class, improve, class_name = ('binary', 'multi'), ('kfolds',), \
                                      ('Poly-2 Kernel', 'AdaBoost', 'GradientBoosting')

    for t_clf in type_class:
        for t_imprv in improve:
            for clf_name in class_name:
                '''new_arr = []
                for ith_iter in range(1, 31):
                    csvfile_path = 'recursos/resultados/{}_{}_{}.csv'.\
                        format(t_clf, t_imprv, clf_name)
                    print 'Renderizing: {}'.format(csvfile_path)

                    results = contenido_csv(csvfile_path)
                    results = np.array(results, dtype='f')

                    statistics = tuple([results[:, col].ravel().mean() for col in range(7)])
                    new_arr.append(statistics)
                new_arr = np.array(new_arr)
                '''
                csvfile_path = 'recursos/resultados/{}_{}_{}.csv'.format(t_clf, t_imprv, clf_name)
                print 'Renderizing: {}'.format(csvfile_path)

                results = contenido_csv(csvfile_path)
                results = np.array(results, dtype='f')

                plt.subplot()
                plt.boxplot(results[:, :7])
                plt.title('{}_{}_{}'.format(t_clf, t_imprv, clf_name))
                plt.xticks(np.arange(1, 8), ('accuracy', 'precision\nclass 1', 'precision\nclass 2', 'recall\nclass 1',
                                             'recall\nclass 2', 'f1-score\class 1', 'f1-score\nclass 2'))
                plt.show()
                #  plt.savefig('recursos/charts/{}_{}_{}.jpeg'.format(t_clf, t_imprv, clf_name))
                plt.close()
                '''guardar_csv(new_arr, 'recursos/resultados/{}_{}_{}_joined.csv'.
                            format(t_clf, t_imprv, clf_name))
                '''