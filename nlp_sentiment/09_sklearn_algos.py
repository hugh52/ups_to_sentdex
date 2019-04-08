
#======================================================================================================================================#
# previous code is included at the top when needed for this exercise as well (if this is essentially a continuation of the last)
import nltk
import pandas as pd
import random
import pickle
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

training_set = featuresets[:1900]
testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
#======================================================================================================================================#

# scikit-learn algorithms

# as for those other algos mentioned at the end of last section, the scikit-learn or sklearn module can provide a multitude of them;
# the gods/titans that maintain the nltk module noticed the value of sklearn and created an API of sorts to incorporate the two;
# this tool is called SklearnClassifier, which we must import; with this, we can also import almost all sklearn classifiers and then 
# use them with nltk.  
# I am not even going to try to describe how awesome this is, and if you do not think so, then learn more, start paying attention, or
# maybe just get over your rediculously jaded self.  (of course, that last sentence was for sure NOT from sentdex tutorials)
# anyway, on with the show....

import sklearn
from nltk.classift.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


# the classification algos are very simple to use, but then again, that is in no way supposed to be the difficult part...
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:", nltk.classify.accuracy(MNB_classifier, testing_Set))















































