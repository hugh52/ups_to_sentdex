
#======================================================================================================================================#
import nltk
import numpy as np
import pandas as pd
import random
import pickle
import sklearn

from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original NaiveBayes algo accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

# as seen in the imports above, a few additional classification algos are brought in from sklearn for comparison purposes;
# and maybe to see if they might be of any help when mixed with the two or three we are already using...

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
#======================================================================================================================================#

# the next approach we will try, since the remaining "scores" are relatively close, is to use all algos at once; we will create another
# classifier, and make the result of that classifier based on what the other algorithms said; similar to a voting system, so we will
# just need an odd number of algorithms...

# combining classifier algorithms is a common technique, accomplished by creating a sort of voting system, where each algorithm gets one
# vote, and the classification that has the most votes is the chosen one; to do this, we want our new classifier to act like a typical
# nltk classifier, with all of the methods; so we just need to use object oriented programming and ensure to inherit from the nltk
# classifier class.

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers
		
		#




































