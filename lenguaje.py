# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

import numpy as np
from nltk import FreqDist
from nltk.corpus import udhr2


class Gramas(object):

    def __init__(self, path_texto='spa.txt'):
        if path_texto != 'spa.txt':
            archivo = open(path_texto)
            self.texto_plano = archivo.read()
            archivo.close()
        else:
            self.texto_plano = udhr2.raw('spa.txt')
        self.vector_frecuencias = None
        self.frecuencia_total = None
        self.calcularfrecuencias()

    def calcularfrecuencias(self):
        distribucion_frecuencias = FreqDist(ch for ch in self.texto_plano)
        self.vector_frecuencias = np.array(distribucion_frecuencias.most_common())
        self.frecuencia_total = distribucion_frecuencias.N()

    def agregargrama(self, valor_pareado=('~', 1)):
        self.vector_frecuencias = np.append(self.vector_frecuencias, np.array([valor_pareado]), axis=0)
        self.frecuencia_total += 1

    def getfrecuencias(self):
        return self.vector_frecuencias, self.frecuencia_total