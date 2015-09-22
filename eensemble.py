# -*- coding: utf-8 -*-
__author__ = 'Juan David Carrillo López'

from math import log
import random
import re
import json

from pyexcel_xlsx import XLSXBook
from nltk import word_tokenize, pos_tag, bigrams, PorterStemmer, LancasterStemmer, FreqDist
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from unbalanced_dataset.ensemble_sampling import EasyEnsemble
from scipy.stats import itemfreq
from ngram import NGram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from utiles import leerarchivo, contenido_csv, guardar_csv


class TextAnalysis:

    def __init__(self, raw_tweets, ortony_lexicon,list_profane_words):
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
        filtered_words = tuple([w for w in word_list if not(w.lower() in stop_words)])
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
        '''for elem in temp_array:
            if elem == 3:
                elem = 1
            else:
                elem = 0
            new_array.append(elem)
        '''
        binarizer = Binarizer(threshold=2.0)
        new_array = binarizer.transform(temp_array)[0]

        return new_array

    @staticmethod
    def votingoutputs(temp_array):
        index_outputs = []
        for col_index in range(temp_array.shape[1]):
            item_counts = itemfreq(temp_array[:, col_index])
            max_times = 0
            for class_label, n_times in item_counts:
                if n_times > max_times:
                    last_class, max_times = class_label, n_times
            #print 'feature {} class voted {} - {}'.format(col_index, class_label, n_times)
            index_outputs.append((col_index, class_label))
        return np.array(index_outputs)

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
        #  guardar_csv(new_set,'recursos/conjuntos.csv')
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

        c, gamma, cache_size = 1.0, 0.1, 300

        classifiers = {'Poly-2 Kernel': svm.SVC(kernel='poly', degree=2, C=c, cache_size=cache_size),
                       'AdaBoost': AdaBoostClassifier(
                           base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1), learning_rate=0.5,
                           n_estimators=100, algorithm='SAMME'),
                       'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                                                            max_depth=1, random_state=0)}
        prediction = {'Poly-2 Kernel': [], 'AdaBoost': [], 'GradientBoosting': []}
        general_metrics = {'Poly-2 Kernel': [[], []], 'AdaBoost': [[], []], 'GradientBoosting': [[], []]}
        type_classifier = {'multi': None, 'binary': None}

        x = min_max_scaler.fit_transform(self.training_set[:, :4])
        '''y = self.training_set[:, 4:5].ravel()
        y_true = self.test_set[:, 4:5].ravel()

        easyens = EasyEnsemble(verbose=True)
        '''
        for type_clf in type_classifier.keys():
            if type_clf == 'multi':
                y = self.training_set[:, 4:5].ravel()
                y_true = self.test_set[:, 4:5].ravel()
            else:
                y = self.binarizearray(self.training_set[:, 4:5].ravel())
                y_true = self.binarizearray(self.test_set[:, 4:5].ravel())
            easyens = EasyEnsemble(verbose=True)
            eex, eey = easyens.fit_transform(x, y)

            ciclo, target_names = 0, ('class 1', 'class 2', 'class 3')
            #  for train_ind, test_ind in kf_total:
            for i_ee in range(len(eex)):
                scaled_test_set = min_max_scaler.fit_transform(self.test_set[:, :4])
                #  print 'Subset {}'.format(ciclo)
                for i_clf, (clf_name, clf) in enumerate(classifiers.items()):
                    clf.fit(eex[i_ee], eey[i_ee])
                    y_pred = clf.predict(scaled_test_set)
                    prediction[clf_name].append(y_pred)
                ciclo += 1

            for clf_name, output in prediction.items():
                all_ypred = np.array(output, dtype=int)
                y_pred = self.votingoutputs(all_ypred)[:, 1].ravel()
                mean_accuracy = accuracy_score(y_true, y_pred)
                general_metrics[clf_name][0] = mean_accuracy
                general_metrics[clf_name][1].append(np.array(precision_recall_fscore_support(y_true, y_pred)).ravel())
            #guardar_csv(results.T, 'recursos/EE_sampling.csv')
            type_classifier[type_clf] = general_metrics
        return type_classifier


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
    for row in filtro[:n_filas-4, :]:
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
    data = {'multi':
                ((0.31609195402298851, 0.30747126436781608, 0.33908045977011492, 0.29022988505747127, 0.34482758620689657, 0.33620689655172414, 0.32183908045977011, 0.30747126436781608, 0.33620689655172414, 0.29597701149425287, 0.27873563218390807, 0.28448275862068967, 0.3045977011494253, 0.35057471264367818, 0.30747126436781608, 0.31034482758620691, 0.31896551724137934, 0.33045977011494254, 0.30172413793103448, 0.35344827586206895, 0.32471264367816094, 0.35057471264367818, 0.33620689655172414, 0.33045977011494254, 0.28160919540229884, 0.3045977011494253, 0.34195402298850575, 0.33908045977011492, 0.30172413793103448, 0.33333333333333331),
                 (0.32471264367816094, 0.5431034482758621, 0.5545977011494253, 0.31896551724137934, 0.57471264367816088, 0.47413793103448276, 0.50574712643678166, 0.58045977011494254, 0.58908045977011492, 0.58908045977011492, 0.50574712643678166, 0.57471264367816088, 0.56896551724137934, 0.57758620689655171, 0.51724137931034486, 0.56321839080459768, 0.56609195402298851, 0.52873563218390807, 0.49137931034482757, 0.60632183908045978, 0.46264367816091956, 0.54022988505747127, 0.55747126436781613, 0.51724137931034486, 0.50862068965517238, 0.55747126436781613, 0.56034482758620685, 0.50862068965517238, 0.52011494252873558, 0.53735632183908044),
                 (0.60344827586206895, 0.55747126436781613, 0.56321839080459768, 0.62643678160919536, 0.53735632183908044, 0.58620689655172409, 0.60632183908045978, 0.51436781609195403, 0.58908045977011492, 0.5, 0.53448275862068961, 0.51436781609195403, 0.58333333333333337, 0.56321839080459768, 0.46839080459770116, 0.50574712643678166, 0.53735632183908044, 0.55172413793103448, 0.53448275862068961, 0.46839080459770116, 0.47413793103448276, 0.59195402298850575, 0.57183908045977017, 0.5316091954022989, 0.52873563218390807, 0.51436781609195403, 0.52011494252873558, 0.49137931034482757, 0.54885057471264365, 0.4885057471264368),
                 (0.64367816091954022, 0.39655172413793105, 0.40229885057471265, 0.64367816091954022, 0.60919540229885061, 0.60344827586206895, 0.64367816091954022, 0.59195402298850575, 0.57183908045977017, 0.35919540229885055, 0.57471264367816088, 0.5, 0.40517241379310343, 0.55172413793103448, 0.45689655172413796, 0.63218390804597702, 0.61206896551724133, 0.25, 0.63218390804597702, 0.47413793103448276, 0.43678160919540232, 0.53448275862068961, 0.58620689655172409, 0.32183908045977011, 0.50287356321839083, 0.46551724137931033, 0.56321839080459768, 0.46264367816091956, 0.43390804597701149, 0.4942528735632184),
                 (0.31321839080459768, 0.58333333333333337, 0.62643678160919536, 0.28160919540229884, 0.62643678160919536, 0.60632183908045978, 0.64367816091954022, 0.5977011494252874, 0.62356321839080464, 0.59482758620689657, 0.56896551724137934, 0.60344827586206895, 0.61781609195402298, 0.63218390804597702, 0.61781609195402298, 0.60919540229885061, 0.57758620689655171, 0.5977011494252874, 0.60632183908045978, 0.62931034482758619, 0.5316091954022989, 0.61781609195402298, 0.61781609195402298, 0.60344827586206895, 0.52873563218390807, 0.59195402298850575, 0.5977011494252874, 0.55172413793103448, 0.52298850574712641, 0.63505747126436785)),
            'binary':
                ((0.66954022988505746, 0.73275862068965514, 0.7068965517241379, 0.71264367816091956, 0.63793103448275867, 0.66091954022988508, 0.67241379310344829, 0.63505747126436785, 0.6954022988505747, 0.70402298850574707, 0.7183908045977011, 0.69252873563218387, 0.64942528735632188, 0.66091954022988508, 0.67528735632183912, 0.73275862068965514, 0.67816091954022983, 0.7183908045977011, 0.72988505747126442, 0.74425287356321834, 0.72413793103448276, 0.67241379310344829, 0.71551724137931039, 0.6522988505747126, 0.70402298850574707, 0.7183908045977011, 0.69827586206896552, 0.70402298850574707, 0.68678160919540232, 0.69252873563218387),
                 (0.58908045977011492, 0.60919540229885061, 0.66091954022988508, 0.66379310344827591, 0.61781609195402298, 0.62068965517241381, 0.59195402298850575, 0.58333333333333337, 0.64942528735632188, 0.64942528735632188, 0.60057471264367812, 0.44827586206896552, 0.56034482758620685, 0.57471264367816088, 0.62931034482758619, 0.69827586206896552, 0.66954022988505746, 0.59195402298850575, 0.51724137931034486, 0.6954022988505747, 0.70114942528735635, 0.58620689655172409, 0.67816091954022983, 0.34770114942528735, 0.66954022988505746, 0.62931034482758619, 0.68103448275862066, 0.67816091954022983, 0.61494252873563215, 0.62931034482758619),
                 (0.66954022988505746, 0.8045977011494253, 0.79885057471264365, 0.77298850574712641, 0.77011494252873558, 0.66091954022988508, 0.67241379310344829, 0.63505747126436785, 0.6954022988505747, 0.70402298850574707, 0.7816091954022989, 0.69252873563218387, 0.75862068965517238, 0.80747126436781613, 0.77586206896551724, 0.7816091954022989, 0.7068965517241379, 0.77011494252873558, 0.8045977011494253, 0.74425287356321834, 0.78735632183908044, 0.67241379310344829, 0.71551724137931039, 0.6522988505747126, 0.75862068965517238, 0.7614942528735632, 0.69827586206896552, 0.79022988505747127, 0.76436781609195403, 0.69252873563218387),
                 (0.7183908045977011, 0.79885057471264365, 0.79597701149425293, 0.81034482758620685, 0.74712643678160917, 0.77298850574712641, 0.72126436781609193, 0.71551724137931039, 0.73275862068965514, 0.74425287356321834, 0.7931034482758621, 0.6954022988505747, 0.76724137931034486, 0.80747126436781613, 0.76436781609195403, 0.77873563218390807, 0.77586206896551724, 0.77298850574712641, 0.79597701149425293, 0.77873563218390807, 0.77586206896551724, 0.7183908045977011, 0.74712643678160917, 0.66091954022988508, 0.76436781609195403, 0.7614942528735632, 0.72988505747126442, 0.7816091954022989, 0.76436781609195403, 0.72988505747126442),
                 (0.67241379310344829, 0.70402298850574707, 0.68103448275862066, 0.71264367816091956, 0.63218390804597702, 0.56609195402298851, 0.67241379310344829, 0.62356321839080464, 0.68965517241379315, 0.66954022988505746, 0.70977011494252873, 0.35344827586206895, 0.56896551724137934, 0.58333333333333337, 0.67241379310344829, 0.7068965517241379, 0.69252873563218387, 0.62356321839080464, 0.59482758620689657, 0.73275862068965514, 0.72413793103448276, 0.65804597701149425, 0.70402298850574707, 0.32471264367816094, 0.68678160919540232, 0.72126436781609193, 0.68965517241379315, 0.69252873563218387, 0.67528735632183912, 0.68390804597701149),
                 (0.26724137931034481, 0.28448275862068967, 0.2557471264367816, 0.25, 0.29022988505747127, 0.2471264367816092, 0.27298850574712646, 0.27586206896551724, 0.37931034482758619, 0.42528735632183906, 0.43678160919540232, 0.39942528735632182, 0.25, 0.23850574712643677, 0.4454022988505747, 0.29310344827586204, 0.36206896551724138, 0.52586206896551724, 0.32183908045977011, 0.2614942528735632, 0.27011494252873564, 0.33620689655172414, 0.27586206896551724, 0.38793103448275862, 0.3045977011494253, 0.31034482758620691, 0.3045977011494253, 0.4885057471264368, 0.31321839080459768, 0.48563218390804597))
    }
    #  stats = cbook.boxplot_stats(np.array(data['multi']).T)
    plt.subplot()
    plt.boxplot(np.array(data['binary']).T)
    plt.show()


def getnewdataset():
    with open('recursos/bullyingV3/tweet.json') as json_file:
        for line in json_file:
            json_data = (json.loads(line)['id'], str(json.loads(line)['text']))
    return json_data


def preprocessdataset():
    first_filter = np.array(readexceldata('recursos/conjuntos.xlsx'))
    prof_word = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])
    ortony_words = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])

    anlys = TextAnalysis(first_filter, ortony_words, prof_word)
    anlys.hashtagsdirectedrtweets()
    anlys.featuresextr()


def machinelearning():
    first_filter = np.array(readexceldata('recursos/conjuntos.xlsx'))

    prof_word = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])
    ortony_words = tuple([str(word.rstrip('\n')) for word in leerarchivo('recursos/offensive_profane_lexicon.txt')])

    anlys = TextAnalysis(first_filter, ortony_words, prof_word)
    data = contenido_csv('recursos/conjuntos.csv')

    general_accuracy = {'MultiClass Poly-2 SVM': [], 'AdaBoost DecisionTree': [], 'GradientBoosting': []}
    for cicle in range(1):
        result_info = anlys.learningtoclassify(np.array(data, dtype='f'))
        print 'Iter {} results {}'.format((cicle + 1), result_info)


if __name__ == '__main__':
    machinelearning()