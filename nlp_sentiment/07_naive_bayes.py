
#======================================================================================================================================#
# previous code is included at the top when needed for this exercise as well (if this is essentially a continuation of the last)
import nltk
import pandas as pd
import random
from nltk.corpus import movie_reviews
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
#======================================================================================================================================#

# naive bayes classification

# choose algo (here naive bayes) and split data into train and test;
# serious bias issues can come about if you train and test on same, so avoid this;

# since dataset is shuffled, we will assign first 2,000 shuffled reviews for training set, which includes positive and negative reviews;
# we will test against the last 1,000 to see how accurate we are

# NOTE: the tutorial says to train with 1,900 and test with the "last 100" but the dataset has 3,000 not 2,000.  I assume this was just 
#       a typo so I have adjusted the numbers.  Once/if I hear back, I will update here. 

# this is considered supervised machine learning, because we are showing the machine data and telling it "this data is positive" or"this
# data is negative"; after training is complete, we show the machine new data and ask the computer, based on what was taught before, 
# what the computer thinks the category of the data is.

# first we have to split the data

# set we will train classifier with:
training_set = featuresets[:2000]

# set we will test against:
testing_set = featuresets[2000:]

# define and train classifier:
classifier = nltk.NaiveBayesClassifier.train(training_set)

# next, we test it:
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


















