# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.text import TextCollection

from utiles import contenido_csv, guardar_csv

n_samples = 1744


# Loading own corpus
print("Loading preprocessed corpus")
t0 = time()
corpus = contenido_csv('recursos/processed_twits_slang.csv')
corpus = [' '.join(document) for document in corpus]
print("\tdone in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features with default parameters")
tfidf_vectorizer = TfidfVectorizer()
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(corpus)
print("\tdone in %0.3fs." % (time() - t0))

set_docs = TextCollection(corpus)

weights = []
for row in tfidf.todense().astype(dtype='S'):
    weights.append(row.getA()[0])
guardar_csv(weights, 'recursos/tfidf_vectors.csv')
