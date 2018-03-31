# -*- coding: utf-8 -*-
'''from math import log
import cProfile, pstats, StringIO
import random
import re

from pyexcel_xlsx import get_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn import svm, cross_validation
import numpy as np
'''

from utiles import leerarchivo, guardar_csv, contenido_csv, binarizearray

from mining import readexceldata, preprocessdataset, getfeaturessample
from kfolds import machinelearning

__author__ = 'Juan David Carrillo López'


if __name__ == '__main__':
    sample = readexceldata('recursos/ponderacion/conjuntos.xlsx')
    # guardar_csv(sample, 'recursos/ponderacion/tweet_weight.csv')
    # preprocessdataset()
    for t_data in ('rand_nongrams', 'rand_ngrams', 'nongrams', 'ngrams'):
        try:
            machinelearning(t_data)
        except IOError as e:
            print 'El archivo {} no fue encontrado'.format(t_data)
    #getfeaturessample()
