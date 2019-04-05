# corpora

# the corpora if basicaly a massive data dump of all kinds of natural language data sets that are definately worth taking a look at;
# we can view them manually, they are simply text files, xml files, etc.
# to find where the nltk module is located in the os, use:
# >>> import nltk
# >>> print(nltk.__file__)
# for me, this returns something similar to 'C:/other/files/specific/to/my/computer/Anaconda3/lib/site-packages/nltk/__init__.py'

# hear, we will use some of the methods from nltk ro handle the corpus.  we will open the Gutenberg Bible and read the first few lines.

import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer

from nltk.corpus import gutenberg


# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
	print(tok[x])

	

# wordnet

# wordnet is a collection of words, definitions, examples of their use, synonyms, antonyms, etc; 
# it is one of the more advanced datasets in the corpus

from nltk.corpus import wordnet


# use the term "program" to find synsets
syns = wordnet.synsets("program")

# example of a synset
print(syns[0].name())

# just the word
print(syns[0].lemmas()[0].name())

# definition of first synset
print(syns[0].definition())
# 'a series of steps to be carried out or goals to be accomplished'

# examples of the word in use
print(syns[0].examples())

# to find synonyms and antonyms of a word, the lemmas will be synonyms, while '.antonyms' can be used to find the antonyms to lemmas;
# so we can populate list just like the following
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			antonyms.append(l.antonyms()[0].name())
			
print(set(synonyms))
print(set(antonyms))

# here, there are many more synonyms than antonyms, since we just looked up the antonym for the first lemma;
# this can be balanced by doing the exact same process for the term "bad".

# next, we compare the similarity of two words and their tenses by incorporating the Wu and Palmer method for semantic related-ness. 

# compare the noun "ship" and "boat"
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))




