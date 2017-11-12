# -*- coding: utf-8 -*-
from math import log
from pyexcel_xlsx import get_data
from ngram import NGram
from nltk import word_tokenize, pos_tag, bigrams, PorterStemmer, LancasterStemmer, FreqDist
from nltk.corpus import stopwords

from utiles import leerarchivo, guardar_csv, contenido_csv

import re
import random
import numpy as np


__author__ = 'Juan David Carrillo López'


class TextAnalysis:
    def __init__(self, raw_tweets, ortony_lexicon, list_profane_words):
        self.raw_tweets = raw_tweets
        self.ortony_list = ortony_lexicon
        self.profane_words = list_profane_words
        self.punctual_signs = ('.', ':', ';', ',', '¿', '?', '!', "'", "\"", '/', '|', '-', '_', '*',
                               '(', ')', '[', ']', '{', '}', '<', '>', '=', '^', '°', '･', '£')
        self.caracteres = ('RT', '@', '#', 'http', 'htt', '&')
        self.unimprt_sequences = ('lol', 'hah', 'amp', 'idk', 'omg', 'wtf', 'wth' 'tf', 'rft', 'ctfu', 'ogt', 'lmao',
                                  'rmft', 'pe', 'tp', 'gpa', 'jk', 'asdf', 'plz', 'pls', 'wbu', 'cpe', 'kms', 'ffs',
                                  'ah', 'aw', 'stfu')

        self.new_tweetset = None
        self.features_space = None
        self.training_set = None
        self.valid_set = None
        self.test_set = None

    @staticmethod
    def replacepronounscontractiosn(whole_txt):
        contractions = (("'m", ' am'), ("'r", ' are'), ("'s", ' is'))
        new_txt = whole_txt
        for old, new in contractions:
            new_txt = new_txt.replace(old, new)
        return new_txt

    @staticmethod
    def removestopwords(word_list, lang_words='english'):
        stop_words = tuple([word.encode('latin-1', errors='ignore') for word in stopwords.words(lang_words)])[29:]
        filtered_words = tuple([w for w in word_list if not (w.lower() in stop_words)])
        return filtered_words

    @staticmethod
    def stemmingword(word_list, stemtype='porter'):
        if stemtype == 'porter':
            stemengine = PorterStemmer()
        else:
            stemengine = LancasterStemmer()
        try:
            filtered_words = [stemengine.stem(token).encode('latin-1', errors='ignore') for token in word_list]
        except UnicodeDecodeError, e:
            print 'Error en el tipo de caracteres descartando texto "{}"'.format(' '.join(word_list))
        else:
            return filtered_words

    @staticmethod
    def stemaposphe(single_word):
        regexp = r'^(.*?)(\'s)?$'
        stem, suffix = re.findall(regexp, single_word)[0]
        return stem

    @staticmethod
    def termfrequency(count_word, total_words):
        result = count_word / float(total_words)
        return result

    def inversdocfreq(self, word):
        tweets_match = 0
        for tweet_tokens, weight, features in self.new_tweetset:
            if word in tweet_tokens:
                tweets_match += 1
        result = log((self.new_tweetset.shape[0] / tweets_match), 10)
        return result

    @staticmethod
    def numpytotuple(arr):
        try:
            return tuple(TextAnalysis.numpytotuple(i) for i in arr)
        except TypeError:
            return arr

    @staticmethod
    def posbigrams(list_words):
        pair_tags = (('PRP', 'VBP'), ('JJ', 'DT'), ('VB', 'PRP'))
        pair_words = tuple(bigrams(list_words))
        res = pos_tag(list_words)
        ptag_counter = [0, 0, 0]
        for word_tokens in pair_words:
            tagged_pair = np.array(pos_tag(word_tokens))
            tagp_1, tagp_2 = tuple(tagged_pair[:, 1])
            #  matches = tuple([(t_1, t_2) for t_1, t_2 in pair_tags if tagp_1 == t_1 and tagp_2 == t_2])
            ind_count = 0
            for t_1, t_2 in pair_tags:
                if tagp_1 == t_1 and tagp_2 == t_2:
                    ptag_counter[ind_count] += 1
                ind_count += 1

        return tuple(ptag_counter)

    def wordsoccurrences(self, words_list, option='ortony'):
        frequencies = FreqDist(words_list)
        ordered_unigrams = frequencies.most_common()
        if option == 'ortony':
            lexicon = self.ortony_list
        else:
            lexicon = self.profane_words
        count = 0
        for t_word, count_w in ordered_unigrams:
            lower_word = t_word.lower()
            three_grams = NGram(lexicon)
            likely_words = three_grams.search(lower_word)
            if len(likely_words) > 0:
                # if lower_word in lexicon:
                count += 1 * count_w

            if lower_word in lexicon:
                count += 1
        return count

    def removepunctuals(self, tweet_t):
        new_tweet = []
        for charact in tweet_t:
            if charact in self.punctual_signs:
                new_tweet.append(' ')
            else:
                new_tweet.append(charact)

        return ''.join(new_tweet)

    def removeunimportantseq(self, list_word):
        try:
            for unimp_seq in self.unimprt_sequences:
                if unimp_seq in list_word:
                    list_word.remove(unimp_seq)
        except TypeError:
            raise ValorNuloError
        else:
            return list_word

    def hashtagsdirectedrtweets(self):
        process_tweets, new_set, text_data = [], [], []
        for text, weight in self.raw_tweets:
            raw_text, class_text = text, weight
            process_tweets.append('Raw text: {}'.format(raw_text))
            text = text.split(' ')
            features = [0, 0]
            for caracter in self.caracteres:
                match = [w_token for w_token in text if caracter in w_token]
                len_match = len(match)
                if len_match > 0 and (caracter == 'http' or caracter == '#' or caracter == '&'):
                    try:
                        text.remove(match[0])
                    except ValueError:
                        print '\telement {} is not present'.format(match)
                    except IndexError:
                        print '\tindex out of range in {}'.format(match)
                elif len_match > 0 and caracter == '@':
                    for char_found in match:
                        try:
                            text.remove(char_found)
                        except ValueError:
                            print '\telement {} is not present'.format(char_found)
                        except IndexError:
                            print '\tindex out of range in {}'.format(char_found)
                    features[1] = 1
                elif len_match > 0:
                    index_match = text.index(match[0])
                    text.remove(match[0])
                    text.insert(index_match, '')
                    features[0] = 1
            text = self.replacepronounscontractiosn(' '.join(text))
            process_tweets.append('\treplace pronouns/contractions: {}'.format(text))
            word_list = word_tokenize(self.removepunctuals(text))
            #  print '{} \n\t{}'.format(' '.join(word_list), pos_tag(word_list))
            process_tweets.append('\tremove punctuals: {}'.format(word_list))
            word_list = [self.stemaposphe(w) for w in word_list]
            process_tweets.append('\tremove aposthrophe: {}'.format(word_list))
            stopw_tweet = self.removestopwords(word_list)
            process_tweets.append('\tremove stopwords: {}'.format(stopw_tweet))
            stemw_tweet = self.stemmingword(stopw_tweet)
            process_tweets.append('\tstemming words: {}'.format(stemw_tweet))
            try:
                # stemw_tweet = self.removeunimportantseq(stemw_tweet)
                process_tweets.append('\tremove unimportant sequences: {}\n'.format(stemw_tweet))
                text_data.append((raw_text, class_text))
                new_set.append((tuple(stemw_tweet), features, int(weight)))
            except ValorNuloError, e:
                print 'Descartando tweet -> {}'.format(e)
            except TypeError:
                print '{}. Descartando tweet -> {}'.format(ValorNuloError(), raw_text)
        # guardar_csv(text_data, 'recursos/resultados/tweets_and_classes.csv')
        # guardararchivo(process_tweets, 'recursos/resultados/processing_tweets_b.txt')
        self.new_tweetset = np.array(new_set)

    def featuresextractor(self, set_name='featurespace.csv'):
        new_set = []
        for tweet_tokens, features, weight in self.new_tweetset:  # features are pending till performance methodoly
            #  print '\n{}'.format(tweet_tokens)
            total_tokens = len(tweet_tokens)
            frequencies = FreqDist(tweet_tokens)
            words_tfidf = [(t_word, round(self.termfrequency(count_w, total_tokens) * self.inversdocfreq(t_word), 2))
                           for t_word, count_w in frequencies.most_common()]
            tfidf_vector = tuple([value for unigram, value in words_tfidf])

            feat_bigrams = self.posbigrams(tweet_tokens)
            ortony_occur = self.wordsoccurrences(tweet_tokens)
            profane_occur = self.wordsoccurrences(tweet_tokens, option='profane')
            preprocessed_twits = self.new_tweetset[:, 0]
            # guardar_csv(preprocessed_twits, 'recursos/processed_twits_slang.csv')
            new_set.append((sum(tfidf_vector), ortony_occur, profane_occur, sum(feat_bigrams), weight))
        guardar_csv(new_set, 'recursos/{}'.format(set_name))
        self.features_space = np.array(new_set)


def preprocessdataset():
    first_filter = np.array(readexceldata('recursos/ponderacion/conjuntos.xlsx'))
    prof_word = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])
    ortony_words = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/ortony_lexicon.txt')])

    anlys = TextAnalysis(first_filter, ortony_words, prof_word)
    anlys.hashtagsdirectedrtweets()
    anlys.featuresextractor('ngrams.csv')


class ValorNuloError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'El valor hace una referencia nula o vacia'


class InvalidOption(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'Invalid argument'


def readexceldata(path_file):
    book = get_data(path_file)
    data_set = np.array(book['filtro'])
    filtro = np.array([row for row in data_set if row[6] <= 2])
    n_filas = filtro.shape[0]

    rangos, filtro2 = [0, 0, 0], []
    for row in filtro[:n_filas - 4]:
        if row[6] == 2:
            valor_selecc = int((row[1] + row[2]) / 2)
        else:
            valor_selecc = int(random.choice(row[1:3]))
        if valor_selecc < 4:
            rangos[0] += 1
            valor_selecc = 1
        elif valor_selecc > 6:
            rangos[2] += 1
            valor_selecc = 3
        else:
            rangos[1] += 1
            valor_selecc = 2

        row[0] = row[0].encode('latin-1', errors='ignore').replace('<<=>>', '')
        filtro2.append((row[0], valor_selecc))
    return filtro2


def gettfidfvectors(path_file='recursos/tfidf_vectors.csv'):
    tfidf_vects = np.array(contenido_csv(path_file), dtype='f')
    return tfidf_vects


def getfeaturessample(type_set):
    data = np.array(contenido_csv('recursos/{}.csv'.format(type_set)), dtype='f')
    if 'rand' in type_set:
        data = np.delete(data, data.shape[1] - 2, 1)  # removing the examiner gradeing
    else:
        data = np.delete(data, 0, 1)  # replacing tfidf vectorial sum by each tfidf vector
        tfidf_vects = gettfidfvectors('recursos/tfidf_vectors.csv')
        data = np.concatenate((tfidf_vects, data), axis=1)

    return data
