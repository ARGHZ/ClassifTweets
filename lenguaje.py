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


def contarvocales(string):
    vowels = "aeiouáéíóú"
    string = string.lower()
    return sum(letter in vowels for letter in string)


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s1, s2)
    if levenshtein(s2) == 0:
        return len(s1)
    fila_anterior = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        fila_actual = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = fila_anterior[j + 1] + 1
            deletions = fila_actual[j] + 1
            substitutions = fila_anterior[j] + (c1 != c2)
            fila_actual.append(min(insertions, deletions, substitutions))
        fila_anterior = fila_actual
    return fila_anterior[-1]