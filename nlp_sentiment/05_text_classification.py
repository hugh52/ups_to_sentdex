

# text classification

# This is a very broad subject; examples include classifying text as to whether it is about politics, about the military, or classifying
# based on the gender of the author who wrote the text; identifying if a body of text is spam or not spam for email filtering is a well
# known classification task; our ultimate goal is to create a sentiment analysis algorithm.

# To create the algo, we will first use a database of movie reviews included with the nltk corpus.  The words will be our features, and
# are part of our positive or negative movie review.  The nltk corpus 'movie_reviews' dataset has the reviews, as well as labels of 
# 'positive' or 'negative'. With the labels included, we can both train and test the data.

import nltk
import random
from nltk.corpus import movie_reviews


# Below, we basically say, in reference to the imported 'movie_reviews' dataset, for each category (positive or negative), take all of 
# the file IDs (each review has its own ID), then store the 'word_tokenized' version for the file ID (which will be a list of words),
# and follow that by the 'positive' or 'negative' label, all in one big list.

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_review.fileids()]

# since we will be training and testing, we now use random to shuffle documents;
# if we did not, we could (or would) end up training on mostly all negatives and a few positives, then testing against all positives
random.shuffle(documents)

# to examine a "small" (relatively) amount of the data we are working with, we examine the first item in documents, 'documents[1]';
# this is still a large list; the first element is a list of the words, second element in the "pos" or "neg" label
print(documents[1])

# now, to collect all words that we find, we use the frequency distribution to find the most common words; 
# punctuation and words unimportant for our purposes like "the" or "a" will of course be at the top of the list. however, since we plan
# on creating a list of thousands of words, this is not a big deal; the actual important words start to show soon.
all_words = []
for w in movie_reviews.words():
	all_words.append(w.lower())
	
all_words = nltk.FreqDist(all_words)

# now to find the most common 15 words:
print(all_words.most_common(15))

# to find the number of occurrences of some given word:
print(all_words["stupid"])


# below is the "Challenge" which provides a method to word with files that are really just a mess and are similar to the below:
# example of bad file is included here entitled 'poorly_delimited.txt'

# now we will open and read the initial file contents and store them to our contents variable
with open('poorly_delimited.txt', 'r') as f:
	contents = f.read()
	# now split and begin to iterate through the lines by splitting apart the classification from the text per line
	lines = contents.split('\n')
	for l in lines:
		# this could also obviously be done with regular expressions
		new_l = l.split(',',1)
	
# now just store to a list or dataframe

import pandas as pd

correcly_split_file = []

with open('poorly_delimited.txt','r') as f:
	contents = contents.read()
	lines = contents.split('\n')
	
	for l in lines:
		new_l = l.split(',',1)
		correctly_split_file.append(new_l)
		
df = pd.DataFrame(correctly_split_file, columns = ['Classification', 'Text'])
									
print(df.head())









