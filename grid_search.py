# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

from math import log
import random
import re

from pyexcel_xlsx import XLSXBook, save_data
from nltk import word_tokenize, pos_tag, bigrams, PorterStemmer, LancasterStemmer, FreqDist
from nltk.corpus import stopwords
from sklearn import svm, cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from skll.metrics import kappa
from statsmodels.stats.inter_rater import cohens_kappa
import numpy as np
import matplotlib.pyplot as plt

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

    def hasgtagsdirectedrtweets(self):
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
        # guardar_csv(new_set,'recursos/conjuntos.csv')
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

        c, gamma, cache_size = 1000, 0.001, 300
        X = min_max_scaler.fit_transform(self.features_space[:, :4])
        y = self.features_space[:, 4:5].ravel()

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

        tuned_parameters = [{'kernel': ['rbf'], 'degree': [2], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000],
                             'cache_size': [cache_size]},
                            {'kernel': ['poly'], 'degree': [2], 'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4],
                             'cache_size': [cache_size]}]
        scores = ['precision', 'recall']
        for score in scores:
            print '# Tuning hyper-parameters for %s\n' % score
            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=10, scoring='%s_weighted' % score)
            clf.fit(X_train, y_train)

            print 'Best parameters set found on development set: \n'
            print '{}\n'.format(clf.best_params_)
            print 'Grid scores on development set:\n'
            for params, mean_score, scores in clf.grid_scores_:
                print '%0.3f (+/-%0.03f) for %r\n' % (mean_score, scores.std() * 2, params)

            print 'Detailed classifications report:\n'
            print 'The model is trained on the full development set.'
            print 'The scores are computed on the hull evaluation set.\n'
            y_true, y_pred = y_test, clf.predict(X_test)
            print classification_report(y_true, y_pred)

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


def makekappatable(eval_1, eval_2, subjects):
    table = np.zeros((subjects, subjects))
    for eval_i in range(len(eval_1)):
        selected = (eval_1[eval_i], eval_2[eval_i])


def plotmetric():
    data = contenido_csv('recursos/conjuntos.csv')
    data = np.array(data, dtype='f')

    true_score = data[1:, 1].astype(float)
    pred_score = data[1:, 3].astype(float)
    # Compute ROC curve and ROC area for each class
    '''n_classes = 3
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for cicle in range(n_classes):
        fpr[cicle], tpr[cicle], _ = roc_curve(true_score, pred_score)
        roc_auc[cicle] = auc(fpr[cicle], tpr[cicle])
    # Compute micro-avergage ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(true_score, pred_score)
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    '''
    cm = confusion_matrix(true_score, pred_score)
    np.set_printoptions(precision=2)
    title, cmap, target_names = 'Confusion matrix', plt.cm.Blues, ('class 1', 'class 2')
    print 'Confusion matrix, without normalization'
    print cm
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    print data


if __name__ == '__main__':
    first_filter = np.array(readexceldata('recursos/conjuntos.xlsx'))

    prof_word = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])
    ortony_words = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])

    anlys = TextAnalysis(first_filter, ortony_words, prof_word)
    # anlys.hasgtagsdirectedrtweets()
    #  anlys.featuresextr()
    data = contenido_csv('recursos/conjuntos.csv')
    anlys.learningtoclassify(np.array(data, dtype='f'))