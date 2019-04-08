

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



# I will have to study these in depth and am not certain of the abbreviations until then, but from my experience I will attempt to 
# tell what these terms stand for; unfortunately, even a surface level conceptual exaplanation of how they work and the particulars of 
# each is beyond the scope of this project, so I will try not to get into that, as I would never complete this if I did.

# starting at top and working down... ending 'NB': of course 'NaiveBayes'; 'SGD': stochastic gradient descent; 
# 'svm' from imports: support vector machines; 'SVC': support vector classification; 'LinearSVC': linear support vector classification; # 'NuSVC': still support vector classification, but instead of using variable C, like traditional svm approaches, Nu is used;
# the main difference is C is unbounded to the upside...

# ok so first the maximum-margin hyperplane is a hyperplane that seperates, say two seperate "clouds" of points in space, so that the 
# distance from each "cloud" of points to this hyperplane is equal; then the margin from each cloud and hyperplane is at a maximum.
# so in traditional svm used for classification, there is a parameter 'C' that has no upper bound but is greater than 0;
# rather reluctantly, I will say this parameter decides the amount of error we are willing to accept in our model; its more comlicated

# one explanation: we look for a solution that will either fail often but the errors will be small, or it fails rarely but when it does,
# the errors are quite large;
# another explanation: large C means model searches for a small margin, meaning most points are included, while a small C seeks a large
# margin, which may not include all or many variables used for training.  
# in general, C says how much or how little we care about the model overfitting the data.

# Nu can control the number of support vectors and training errors and must lie between 0 and 1.
# it is considered an upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors -
# so when using .1, then at most 10% of training examples will be misclassified while at least 10% of the training examples will 
# be support vectors; support vectors are samples that lie on the margins of the maximum-margin hyperplane 

# in general, support vector classification works by finding seperating hyperplanes in higher dimentsions while balancing the inclusion
# of training variables with the chance of over-fitting.

# as for C and Nu, they are really about the same, except C must be (or will ideally be) optimized, requiring additional steps.

# and that right there is why I will stop providing explanations or even guesses; please note the info above may be way off and 
# COMPLETELY INCORRECT, and it is also my personal ramblings from readings about SVMs.  so, moving along now.....

















