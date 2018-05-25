# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'
from ngram import NGram
from TwitterAPI import TwitterAPI, TwitterError
from nltk import word_tokenize
import numpy as np
import warnings

from lenguaje import contarvocales
from codigoaritm import *
from utiles import leerarchivo, guardararchivo, contenido_csv


if __name__ == '__main__':
    '''muestreo = leerarchivo('resources/muestreo_caracteres.txt')
    frecuencia_total = float(muestreo[len(muestreo)-1].split(':')[1])
    muestreo = muestreo[1:len(muestreo)-1]
    muestreo = np.array([par.split(' | ') for par in muestreo])
    alfabeto = muestreo[:, 0].ravel()
    probabilidades = muestreo[:, 1].ravel()
    alfabeto = ' | '.join(alfabeto)
    probabilidades = ' | '.join([str(int(i)/frecuencia_total) for i in probabilidades])

    muestreo = leerarchivo('resources/lexico.txt')
    lexico = [par.split(',') for par in muestreo]
    '''
    id_tweets = []
    metdat_tweets = np.array(contenido_csv('resources/data.csv'))
    for row in metdat_tweets:
        if row[2] == '"y"':
            id_tweets.append(row[0])
    id_tweets = tuple(id_tweets)

    consumer_key = 'kxfJjFCXjkRySLkW2aHGeAXxN'
    consumer_secret = 'VKalY6au6029H5uqo63VHH1VWcYwaBmlJ36EPulYUBmThyvDUi'
    access_key = '1576798795-MJcRA8Yu8nfgDWbIQjshgio6bOoBCBOGZbSOF06'
    access_secret = 'jPVa8ELVIDT2StlNJvts6UmZASllsliVdvHg7VikT88ew'

    api = TwitterAPI(consumer_key, consumer_secret, access_key, access_secret)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        muestro = []
        for id_t in id_tweets:
            respuesta = api.request('statuses/show/:%d' % int(id_t))
            try:
                for item in respuesta.get_iterator():
                    muestro.append((id_t, item['text']))
            except TwitterError.TwitterRequestError as e:
                print(e)
        muestro = tuple(muestro)
        guardararchivo(muestro, 'resources/bully_trace.txt')
    '''respuesta = api.request('search/tweets', {'q': 'yamecanse', 'count': '100', 'lang': 'es'})
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
        mensajes = tuple(muestreo)
        results = []
        iteracion = 0
        for mensaje in mensajes:
            iteracion += 1
            caracteres = NGram(mensaje.split(' '))
            try:
                entropia_msj = inst.entropiadelmensaje(mensaje)
                n_vocales = contarvocales(mensaje)
                n_palabras = len(word_tokenize(mensaje))
                data_set = ','.join((str(entropia_msj), str(n_vocales), str(n_palabras)))

                print('\nMensaje {4}: \'{3}\' \nEntropía: {0} \tTotal de vocales: {1} \t Total de palabras: {2}'
                      .format(str(entropia_msj), n_vocales, n_palabras, mensaje, iteracion))

                results.append(data_set)
            except ExistSimbError as e:
                pass
            else:
                for palabrota in lexico:
                    minusculas = palabrota[0].lower()
                    query = caracteres.search(minusculas)
                    coincidencias = [match for match in query if match[1] > 0.29]
                    if len(coincidencias) > 0:
                        # print('\tBuscando >> {0}: {1}'.format(minusculas, coincidencias[0]))
                        pass
    finally:
        results = tuple(results)
        guardararchivo(results, 'resources/entrenamiento.txt', 'a')
        print('\nFinalizando ejecución del programa...')
    '''