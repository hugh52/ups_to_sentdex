

# this is an example of the module called 'sentiment_mod' (here called 'sent_mod') in the sentdex tutorials;
# if you are using an older version of Python (I think 3.2 or 3.4 and under?), do not forget to add an empty '__init__.py' file
# to the directory containing this file, in order to make in importable. some examples are described at the bottom:

# imports:

import nltk
import random

from statistics import mode

from nltk.tokenize import word_tokenize

from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI

import warnings
warnings.simplefilter('ignore')


# build 'VoteClassifier' class:

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# read 'documents', 'word_features', and 'featuresets' pickled files,
# again assuming they are stored in directory named 'data_etc/'::

documents_f = open("data_etc/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("data_etc/word_features.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

featuresets_f = open("data_etc/featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()


# build 'find_features' function to return features from document

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# radomize featuresets, split for training and testing algos:

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]


# read in algos from saved pickled files, 
# again assuming they are stored in directory named 'data_etc/':

open_file = open("data_etc/pickled_algo_originalnaivebayes5k.pickle","rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("data_etc/pickled_algo_MNB_classifier5k.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("data_etc/pickled_algo_BernoulliNB_classifier5k.pickle","rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("data_etc/pickled_algo_LogisticRegression_classifier5k.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("data_etc/pickled_algo_LinearSVC_classifier5k.pickle","rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("data_etc/pickled_algo_SGDC_classifier5k.pickle","rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()


# assign algo combination to 'voted_classifier' and build function to return classification and confidence

voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


# breif examples:

# 'import sent_mod as s'
## or, assuming 'sent_mod.py' file is located in folder named 'ghub', which resides in folder named 'nlp', then use:
## 'from nlp.ghub import sent_mod as s'
# 
# print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
# print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, #  0/10"))

# running the above should produce "('pos', 1.0)" after the first statement and "('neg', 1.0)" after the next.
#
