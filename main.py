# -*- coding: utf-8 -*-
__author__ = 'Argvz'
from ngram import NGram
import numpy as np

from codigoaritm import *
from utiles import leerarchivo


if __name__ == '__main__':
    muestreo = leerarchivo('resources/muestreo_caracteres.txt')
    frecuencia_total = float(muestreo[len(muestreo)-1].split(':')[1])
    muestreo = muestreo[1:len(muestreo)-1]
    muestreo = np.array([par.split(' | ') for par in muestreo])
    alfabeto = muestreo[:, 0].ravel()
    probabilidades = muestreo[:, 1].ravel()
    alfabeto = ' | '.join(alfabeto)
    probabilidades = ' | '.join([str(int(i)/frecuencia_total) for i in probabilidades])

    muestreo = leerarchivo('resources/lexico.txt')
    lexico = [par.split(',') for par in muestreo]

    try:
        inst = CodigoAritm(alfabeto, probabilidades)
    except SimbProbsError as e:
        print(e)
    except ItemVacioError as e:
        print(e)
    else:
        mensajes = ('no ests chngndo', 'cmo chgas', 'vt a l hingada', 'nga tv modr', 'chngn',
                    'chngda mdre', 'p  to', 'pv t05', 'stpid', 'stpdo', 'indiota', 'Piche',
                    'inche m dre', 'q nches haces!?', 'q pichs!?', 'ndjo', 'ndja', 'ch...',
                    'chxgx xx madxe')
        resultados = []
        for mensaje in mensajes:
            caracteres = NGram(mensaje.split(' '))
            print('\nEntropía de \'{0}\': {1}'.format(mensaje, str(inst.entropiadelmensaje(mensaje))))
            inst.precodmsj(mensaje+'~')
            for palabrota in lexico:
                minusculas = palabrota[0].lower()
                query = caracteres.search(minusculas)
                coincidencias = [match for match in query if match[1] > 0.22]
                if len(coincidencias) > 0:
                    print('\tBuscando >> {0}: {1}'.format(minusculas, coincidencias))
    finally:
        print('\nTerminando ejecución del programa')