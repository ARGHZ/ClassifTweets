# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

from math import log
import random
import re
import json

from pyexcel_xlsx import XLSXBook, save_data
from ngram import NGram
from Levenshtein import distance
from nltk import word_tokenize, pos_tag, bigrams, PorterStemmer, LancasterStemmer, FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm, cross_validation
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from utiles import leerarchivo, contenido_csv, guardar_csv


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
    def binarizearray(temp_array):
        new_array = []
        for elem in temp_array:
            if elem == 3:
                elem = 1
            else:
                elem = 0
            new_array.append(elem)
        return new_array

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
            likely_words = three_grams.search(lower_word, 0.5)
            #  levens_words = [(similar_w, distance(lower_word, similar_w)) for similar_w, ratio in likely_words]
            if len(likely_words) > 0:
            #if lower_word in lexicon:
                count += 1 * count_w
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
        new_set = []
        for text, weight in self.raw_tweets:
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

            word_list = word_tokenize(self.removepunctuals(text))
            word_list = [self.stemaposphe(w) for w in word_list]
            stopw_tweet = self.removestopwords(word_list)

            stemw_tweet = self.stemmingword(stopw_tweet)

            try:
                stemw_tweet = self.removeunimportantseq(stemw_tweet)
                new_set.append((tuple(stemw_tweet), features, int(weight)))
            except ValorNuloError, e:
                print 'Descartando tweet -> {}'.format(e)

        self.new_tweetset = np.array(new_set)

    def featuresextr(self):
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
            #  print (tfidf_vector, ortony_occur, profane_occur, feat_bigrams, weight)
            new_set.append((sum(tfidf_vector), ortony_occur, profane_occur, sum(feat_bigrams), weight))
        guardar_csv(new_set, 'recursos/conjuntos.csv')
        self.features_space = np.array(new_set)

    def learningtoclassify(self, data_set=[]):
        if len(data_set) > 0:
            self.features_space = data_set
        number_rows = self.features_space.shape[0]
        np.random.shuffle(self.features_space)
        min_max_scaler = MinMaxScaler()

        self.training_set = self.features_space[:int(number_rows * .8)]
        #  self.valid_set = self.features_space[int(number_rows*.5)+1:int(number_rows*.8)]
        self.test_set = self.features_space[int(number_rows * .8) + 1:]

        c, gamma, cache_size = 1000.0, 0.001, 300
        x = min_max_scaler.fit_transform(self.features_space[:, :4])
        y, = self.features_space[:, 4:5].ravel(),
        bin_y = np.array(self.binarizearray(y))

        poly_clf = svm.SVC(kernel='poly', degree=2, C=c, gamma=gamma, cache_size=cache_size)  # , class_weight='auto'
        rbf_clf = svm.SVC(kernel='rbf', degree=2, gamma=gamma, C=c)
        ovr_clf = OneVsRestClassifier(poly_clf)
        #  one_class = svm.OneClassSVM(kernel='poly', gamma=gamma, cache_size=cache_size)
        #  per_clf = Perceptron(n_iter=250)
        nu_svc = svm.NuSVC(kernel='rbf', degree=2, gamma=gamma, cache_size=cache_size)
        adab_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1),
                                      learning_rate=0.5, n_estimators=100, algorithm='SAMME')
        gradb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0)

        kf_total = cross_validation.KFold(len(x), n_folds=10)
        #  kf_total = cross_validation.ShuffleSplit(len(x), n_iter=10, test_size=0.3)

        general_metrics = {'One-vs-Rest': [], 'NuSVC RBF': [],
                           'MultiClass Poly-2 SVM': [], 'MultiClass RBF SVM': [],
                           'AdaBoost DecisionTree': [], 'GradientBoosting': []}

        ciclo, target_names = 1, ('class 1', 'class 2', 'class 3')

        results = []
        for train_ind, test_ind in kf_total:
            target_kfold = len(kf_total)
            scaled_test_set = min_max_scaler.fit_transform(x[test_ind])
            #  print ciclo
            if ciclo == target_kfold:
                results.append(np.insert(y[test_ind], 0, 11))
                results.append(np.insert(bin_y[test_ind], 0, 12))

            ovr_clf.fit(x[train_ind], bin_y[train_ind])
            y_pred = ovr_clf.predict(scaled_test_set)
            y_true = bin_y[test_ind]
            ind_score = ovr_clf.fit(x[train_ind], bin_y[train_ind]).score(x[test_ind], bin_y[test_ind])
            if ciclo == target_kfold:
                results.append(np.insert(y_pred, 0, 13))
                #  print '{} \n{}'.format('OVR', classification_report(y_true, y_pred))
            general_metrics['One-vs-Rest'].append((ind_score, 0))

            nu_svc.fit(x[train_ind], bin_y[train_ind])
            y_pred = nu_svc.predict(scaled_test_set)
            ind_score = nu_svc.fit(x[train_ind], bin_y[train_ind]).score(x[test_ind], bin_y[test_ind])
            if ciclo == target_kfold:
                results.append(np.insert(y_pred, 0, 14))
                #  print '{} \n{}'.format('NuSVC', classification_report(y_true, y_pred))
            general_metrics['NuSVC RBF'].append((ind_score, 0))

            poly_clf.fit(x[train_ind], bin_y[train_ind])
            y_pred = poly_clf.predict(scaled_test_set)
            y_true = bin_y[test_ind]
            ind_score = poly_clf.fit(x[train_ind], bin_y[train_ind]).score(x[test_ind], bin_y[test_ind])
            if ciclo == target_kfold:
                results.append(np.insert(y_pred, 0, 15))
                #  print '{} \n{}'.format('POLY', classification_report(y_true, y_pred, target_names=target_names))
            general_metrics['MultiClass Poly-2 SVM'].append((ind_score, 0))

            rbf_clf.fit(x[train_ind], bin_y[train_ind])
            y_pred = rbf_clf.predict(scaled_test_set)
            y_true = bin_y[test_ind]
            ind_score = rbf_clf.fit(x[train_ind], bin_y[train_ind]).score(x[test_ind], bin_y[test_ind])
            if ciclo == target_kfold:
                results.append(np.insert(y_pred, 0, 16))
                #  print '{} \n{}'.format('RBF', classification_report(y_true, y_pred, target_names=target_names))
            general_metrics['MultiClass RBF SVM'].append((ind_score, 0))

            adab_clf.fit(x[train_ind], bin_y[train_ind])
            y_pred = adab_clf.predict(scaled_test_set)
            y_true = bin_y[test_ind]
            ind_score = adab_clf.fit(x[train_ind], bin_y[train_ind]).score(x[test_ind], bin_y[test_ind])
            if ciclo == target_kfold:
                results.append(np.insert(y_pred, 0, 17))
                #  print '{} \n{}'.format('RBF', classification_report(y_true, y_pred, target_names=target_names))
            general_metrics['AdaBoost DecisionTree'].append((ind_score, 0))

            gradb_clf.fit(x[train_ind], bin_y[train_ind])
            y_pred = gradb_clf.predict(scaled_test_set)
            y_true = bin_y[test_ind]
            ind_score = gradb_clf.fit(x[train_ind], bin_y[train_ind]).score(x[test_ind], bin_y[test_ind])
            if ciclo == target_kfold:
                results.append(np.insert(y_pred, 0, 18))
                #  print '{} \n{}'.format('RBF', classification_report(y_true, y_pred, target_names=target_names))
            general_metrics['GradientBoosting'].append((ind_score, 0))

            ciclo += 1

        #  results = np.array(results)
        #  guardar_csv(results.T, 'recursos/KFolds.csv')

        for clf_name, metrics in general_metrics.items():
            metrics = np.array(metrics)
            mean_accuray = round(metrics[:, 0].ravel().mean(), 4)
            general_metrics[clf_name] = mean_accuray
            # print '{} - {}'.format(clf_name, mean_accuray)
        return general_metrics

    def getoriginalset(self):
        return self.raw_tweets

    def getfilteredset(self):
        return self.new_tweetset

    def getfeatures(self):
        return self.features_space


class ValorNuloError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return 'El valor hace una referencia nula o vacia'


def readexceldata(path_file):
    book = XLSXBook(path_file)
    content = book.sheets()
    data_set = np.array(content['filtro'])[:2326, :7]
    filtro = np.array([row for row in data_set if row[6] <= 2])
    n_filas, n_columnas = filtro.shape

    rangos, filtro2 = [0, 0, 0], []
    for row in filtro[:n_filas - 4, :]:
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


def plotmetric():
    labels = ('Poly-2', 'AdaBoost', 'GradientB', 'O-v-R', 'RBF', 'NuSVC')
    data = {'multi':
                ((0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5832, 0.5832, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5838, 0.5831, 0.5831),
                 (0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.676, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.676, 0.6766, 0.6766, 0.6766, 0.6767, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766),
                 (0.6789, 0.6823, 0.6777, 0.68, 0.6812, 0.6789, 0.676, 0.6755, 0.6795, 0.6766, 0.6789, 0.6726, 0.6789, 0.6765, 0.68, 0.6812, 0.6737, 0.6783, 0.6737, 0.6737, 0.6738, 0.676, 0.6703, 0.6777, 0.6761, 0.6794, 0.6755, 0.6749, 0.6783, 0.6743),
                 (0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5832, 0.5832, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5831),
                 (0.6216, 0.6198, 0.6187, 0.6227, 0.6221, 0.6193, 0.6227, 0.6238, 0.621, 0.6226, 0.6199, 0.6192, 0.6233, 0.6176, 0.6215, 0.6187, 0.6221, 0.621, 0.6215, 0.6199, 0.6211, 0.621, 0.6209, 0.621, 0.6176, 0.6198, 0.6221, 0.6158, 0.6192, 0.6221)),
            'binary':
                ((0.7339, 0.7339, 0.734, 0.7339, 0.734, 0.7339, 0.734, 0.7339, 0.7339, 0.7339, 0.7339, 0.734, 0.734, 0.7339, 0.734, 0.734, 0.7339, 0.734, 0.7339, 0.7339, 0.7339, 0.734, 0.7339, 0.7339, 0.734, 0.7339, 0.734, 0.7339, 0.734, 0.734),
                 (0.7764, 0.7752, 0.7746, 0.7741, 0.7741, 0.7758, 0.7724, 0.7724, 0.774, 0.7741, 0.7729, 0.7741, 0.7753, 0.7763, 0.7736, 0.7752, 0.7758, 0.7718, 0.7758, 0.7775, 0.7734, 0.7753, 0.7741, 0.7741, 0.7746, 0.7752, 0.7746, 0.7729, 0.7781, 0.7735),
                 (0.7827, 0.7804, 0.781, 0.7798, 0.777, 0.7804, 0.781, 0.7781, 0.7792, 0.7827, 0.7798, 0.7798, 0.7804, 0.7798, 0.777, 0.7787, 0.7815, 0.7804, 0.781, 0.7769, 0.7734, 0.7804, 0.7764, 0.777, 0.7827, 0.7815, 0.7816, 0.7775, 0.781, 0.777),
                 (0.7339, 0.7339, 0.734, 0.7339, 0.734, 0.7339, 0.734, 0.7339, 0.7339, 0.7339, 0.7339, 0.734, 0.734, 0.7339, 0.734, 0.734, 0.7339, 0.734, 0.7339, 0.7339, 0.7339, 0.734, 0.7339, 0.7339, 0.734, 0.7339, 0.734, 0.7339, 0.734, 0.734),
                 (0.7494, 0.7448, 0.7449, 0.7465, 0.7437, 0.7443, 0.7466, 0.7465, 0.7431, 0.7448, 0.746, 0.7471, 0.75, 0.7437, 0.7489, 0.7448, 0.746, 0.742, 0.7425, 0.7454, 0.7448, 0.7466, 0.7471, 0.7437, 0.7437, 0.7466, 0.7512, 0.7459, 0.7455, 0.7477),
                 (0.7689, 0.7723, 0.7684, 0.7718, 0.7713, 0.7689, 0.7713, 0.7718, 0.7689, 0.7706, 0.7712, 0.7718, 0.7701, 0.7683, 0.773, 0.7661, 0.7724, 0.7706, 0.7678, 0.7677, 0.7723, 0.7707, 0.77, 0.7689, 0.7712, 0.7712, 0.7735, 0.7689, 0.7684, 0.7695))
    }

    data2 = {'multi':
                 ((0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5832, 0.5831, 0.5832, 0.5832, 0.5832, 0.5832, 0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831),
                  (0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6761, 0.6766, 0.6767, 0.6766, 0.6766, 0.6767, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6767, 0.6766, 0.6766, 0.6766, 0.6766, 0.676, 0.6766, 0.6766, 0.6766, 0.6766, 0.6766, 0.6765, 0.6766, 0.6766),
                  (0.676, 0.6737, 0.6778, 0.6755, 0.6749, 0.6772, 0.6766, 0.6795, 0.6755, 0.6749, 0.6778, 0.6766, 0.6783, 0.6789, 0.6738, 0.6761, 0.6784, 0.6749, 0.6817, 0.6732, 0.6778, 0.6714, 0.6766, 0.6749, 0.6778, 0.6795, 0.6766, 0.6765, 0.6754, 0.6778),
                  (0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5832, 0.5831, 0.5832, 0.5832, 0.5832, 0.5832, 0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5831, 0.5832, 0.5831, 0.5832, 0.5831, 0.5831, 0.5831, 0.5831),
                  (0.6324, 0.6324, 0.6273, 0.6325, 0.6279, 0.6262, 0.6296, 0.6267, 0.6325, 0.6279, 0.6325, 0.6324, 0.6284, 0.6314, 0.629, 0.629, 0.6285, 0.6324, 0.6319, 0.6324, 0.6267, 0.6336, 0.6324, 0.6279, 0.6221, 0.6325, 0.6296, 0.6301, 0.6324, 0.6324)),
             'binary':
                 ((0.734, 0.7339, 0.734, 0.734, 0.734, 0.734, 0.734, 0.7339, 0.7339, 0.7339, 0.7339, 0.7339, 0.7339, 0.734, 0.734, 0.734, 0.734, 0.734, 0.734, 0.7339, 0.734, 0.7339, 0.734, 0.734, 0.734, 0.734, 0.7339, 0.7339, 0.734, 0.7339),
                  (0.7712, 0.7746, 0.7758, 0.7741, 0.7735, 0.7723, 0.7718, 0.7769, 0.7758, 0.7746, 0.7741, 0.7746, 0.7718, 0.7723, 0.7741, 0.7718, 0.773, 0.7764, 0.7747, 0.7752, 0.7747, 0.7747, 0.7747, 0.773, 0.7741, 0.7764, 0.7741, 0.7735, 0.7735, 0.7752),
                  (0.781, 0.7821, 0.7821, 0.7827, 0.7793, 0.7775, 0.7741, 0.7786, 0.7815, 0.7775, 0.7798, 0.7815, 0.7798, 0.7787, 0.7787, 0.7787, 0.7764, 0.7804, 0.777, 0.7787, 0.7816, 0.7816, 0.7798, 0.7753, 0.7798, 0.7793, 0.7787, 0.7798, 0.781, 0.7827),
                  (0.734, 0.7339, 0.734, 0.734, 0.734, 0.734, 0.734, 0.7339, 0.7339, 0.7339, 0.7339, 0.7339, 0.7339, 0.734, 0.734, 0.734, 0.734, 0.734, 0.734, 0.7339, 0.734, 0.7339, 0.734, 0.734, 0.734, 0.734, 0.7339, 0.7339, 0.734, 0.7339),
                  (0.7672, 0.7672, 0.7672, 0.7672, 0.7673, 0.7672, 0.7609, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7672, 0.7621, 0.7672, 0.7672, 0.7672, 0.7672, 0.7678, 0.7672, 0.7672),
                  (0.7684, 0.7695, 0.7724, 0.7604, 0.7409, 0.7586, 0.7333, 0.7569, 0.762, 0.7603, 0.7511, 0.7678, 0.7689, 0.7396, 0.7569, 0.7678, 0.7574, 0.7598, 0.738, 0.7672, 0.7494, 0.7586, 0.7534, 0.7804, 0.7143, 0.7063, 0.7689, 0.7408, 0.7684, 0.7626))
             }
    #  stats = cbook.boxplot_stats(np.array(data['multi']).T)
    plt.subplot()
    plt.boxplot(np.array(data['multi']).T, labels=labels[:5])
    plt.show()


if __name__ == '__main__':
    first_filter = np.array(readexceldata('recursos/conjuntos.xlsx'))

    prof_word = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])
    ortony_words = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])

    '''with open('recursos/bullyingV3/tweet.json') as json_file:
        for line in json_file:
            json_data = (json.loads(line)['id'], str(json.loads(line)['text']))
    '''
    anlys = TextAnalysis(first_filter, ortony_words, prof_word)
    anlys.hashtagsdirectedrtweets()
    anlys.featuresextr()
    '''
    data = contenido_csv('recursos/conjuntos.csv')

    general_accuracy = {'One-vs-Rest': [], 'NuSVC RBF': [], 'MultiClass Poly-2 SVM': [], 'MultiClass RBF SVM': [],
                        'AdaBoost DecisionTree': [], 'GradientBoosting': []}
    for cicle in range(30):
        accuracies = anlys.learningtoclassify(np.array(data, dtype='f'))
        print 'Iter {} results {}'.format((cicle + 1), accuracies)
        for clf_name, mean_acc in accuracies.items():
            general_accuracy[clf_name].append(mean_acc)
    for clf_name, vect_accu in general_accuracy.items():
        mean_acc = np.array(vect_accu).mean()
        print '{} {} -> {}'.format(clf_name, mean_acc, vect_accu)

    plotmetric()
    filtro = first_filter[:, 1].ravel()
    print rangos
    print np.bincount(np.array(filtro, dtype=np.int32))

    n, bins, patches, = plt.hist(np.array(filtro, dtype=np.int32), normed=1, facecolor='green', alpha=0.5)
    plt.xlabel('Weighting')
    plt.ylabel('Probability')
    plt.title('Histogram')
    plt.show()
    '''
