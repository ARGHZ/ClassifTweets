# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'
from ngram import NGram
from TwitterAPI import TwitterAPI
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

    consumer_key = 'kxfJjFCXjkRySLkW2aHGeAXxN'
    consumer_secret = 'VKalY6au6029H5uqo63VHH1VWcYwaBmlJ36EPulYUBmThyvDUi'
    access_key = '1576798795-MJcRA8Yu8nfgDWbIQjshgio6bOoBCBOGZbSOF06'
    access_secret = 'jPVa8ELVIDT2StlNJvts6UmZASllsliVdvHg7VikT88ew'

    api = TwitterAPI(consumer_key, consumer_secret, access_key, access_secret)
    respuesta = api.request('search/tweets', {'q': 'yamecanse', 'count': '100', 'lang': 'es'})
    muestreo = []
    for item in respuesta.get_iterator():
        texto = item['text'].encode('latin-1', 'ignore')
        muestreo.append(texto.decode('latin-1'))

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
        mensajes = tuple(muestreo)

        for mensaje in mensajes:
            caracteres = NGram(mensaje.split(' '))
            try:
                print('\nEntropía de \'{0}\': {1}'.format(mensaje, str(inst.entropiadelmensaje(mensaje))))
                inst.precodmsj(mensaje+'~')
            except ExistSimbError as e:
                print('{0} \t Ignorando mensaje'.format(e))
            else:
                for palabrota in lexico:
                    minusculas = palabrota[0].lower()
                    query = caracteres.search(minusculas)
                    coincidencias = [match for match in query if match[1] > 0.29]
                    if len(coincidencias) > 0:
                        print('\tBuscando >> {0}: {1}'.format(minusculas, coincidencias[0]))
    finally:
        print('\nTerminando ejecución del programa')