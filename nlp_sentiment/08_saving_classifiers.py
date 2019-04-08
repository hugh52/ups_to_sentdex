

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

training_set = featuresets[:1900]
testing_set = featuresets[1900:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
#======================================================================================================================================#

# saving classifiers

# the process of training classifiers can take very long time periods, especially when using larger datasets;
# instead of training the model each time we use it, the Pickle module allows us to go ahead and serialize our classifier object, so
# that all we need to do is load that file (obviously, a very quick process).

# first we save the object; typically, after train() is called to train the classifier, we can use process below to save it.

import pickle


# here, we open pickle file, prepare to write to it in bytes, then dump the data; 
# first parameter to dump is what we are dumping, second parameter is where we are dumping it; then we can close the file;
# once this is done, we have a pickled, or serialized, object saved in our script's directory.
save_classifier = open("naivebayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

# now, to open and use the classifier...  the .pickle file if a serialized object, so all we need to do is read it into memory; as for
# timing, it should be about the same as any file.

# here, we open the file to read as bytes; then use pickle.load() to load the file; then save the data to the classifier variable; and
# finally, close the file.  now we have the classifier object as before.
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

# now we can use this object and no longer have to train our classifier each time we want to use it to classify; this is more than
# awesome, however, if the accuracy is not where we would like it to be, then other classifiers can be used...

# IMPORTANT NOTE: yes, we have been told 60-75% average, though personally I obtained 79% when testing on the command line and then 89% 
# when testing in a jupyter notebook, both first time scores - probably only due to advances/updates in the module and/or specific 
# models, or some other reason; I must state to please not think I did some magic to get a higher score, I am confident yours will be 
# similar and it is worth mentioning that (as we will see later) an extremely high accuracy may not always be a good thing...

