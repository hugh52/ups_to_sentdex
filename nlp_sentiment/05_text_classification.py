

# text classification

# This is a very broad subject; examples include classifying text as to whether it is about politics or about the military, or classifying
# based on the gender of the author who wrote the text; identifying if a body of text is spam or not spam for email filtering is a well
# known classification task; our ultimate goal is to create a sentiment analysis algorithm.

# To create the algo, we will first use a database of movie reviews included with the nltk corpus.  The words will be our features, and
# are part of our positive or negative movie review.  The nltk corpus 'movie_reviews' dataset has the reviews, as well as labels of 
# 'positive' or 'negative'. With the labels included, we can both train and test the data.

import nltk
import random
from nltk.corpus import movie_reviews


# Below, we are basically saying, in reference to the imported 'movie_reviews' dataset, in each category (positive or negative), take all
# of the file IDs (each review has its own ID), then store the 'word_tokenized' version for the file ID (which will be a list of words),
# and follow that by the 'positive' or 'negative' label, all in one big list.












