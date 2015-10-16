# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'
import _mysql
import numpy as np
from nltk import word_tokenize
from utiles import leerarchivo
from threadqueue import ThreadQueue


def buscar_coincidencias(text_set, lexico, n_hilo):
    contador = 0
    for t_usuario, t_text in text_set:
        if [palabra for palabra in lexico if palabra in t_text]:
            contador += 1
    print(contador)
    return contador


if __name__ == '__main__':
    # Getting the lexicon of offensive/profane words
    vocabulario = leerarchivo('recursos/offensive_profane_lexicon.txt')
    vocabulario = tuple(word_tokenize(' '.join(vocabulario)))  # Cleaning the words and performing access to them

    # Getting the entire dataset of tweets
    base_datos = _mysql.connect(user='zacatecas', passwd='yomero', db='nlpresearch')
    base_datos.query('SELECT screen_name, tweet_text FROM tweets')
    results = base_datos.use_result()
    text_tweets = results.fetch_row(maxrows=0)

    text_tweets = np.asarray(text_tweets, np.str)  # Using ctypes in numpy array
    groups_tweets = np.array_split(text_tweets, 20)  # Making subsets due to large data

    task_queue = ThreadQueue(7)
    num_thread = 0
    for tweets_group in groups_tweets:
        num_thread += 1
        '''This will block if the queue is full
        and will enqueue the thread as soon as space become available
        '''
        task_queue.enqueue(buscar_coincidencias, tweets_group, vocabulario, num_thread)
    task_queue.join()  # Will block untill all threads are done and the queue is empty