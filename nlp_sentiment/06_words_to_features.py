

# words to features

# here we build off previous section and compile feature lists of words from positive reviews and words from negative reviews;
# the idea is to hopefully see some trends or patterns in specific types of words from positive or negative reviews

import nltk
import random
from nltk.corpus import movie_reviews


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
  all_words.append(w.lower())
  
all_words.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

# above is the same as before, but now we have added a new variable, 'word_features', which contains the top 3,000 most commmon words;
# next, we write a function to find these top 3,000 words in our positive and negative documents, 
# marking their presence as either positive or negative.

def find_features(document):
  words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features

# to print a featureset:
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

# we can save feature existence booleans and their respective positive or negative categories by writing the above for all our documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# now we should have features and labels, so it is time to train an algorithm then test it (next section)


