# corpora

# the corpora is basically a massive data dump of all kinds of natural language data sets that are definately worth taking a look at;
# we can view them manually, they are simply text files, xml files, etc.
# to find where the nltk_data_directory, and the actual nltk module (specifically, its __init__.py), is located in the os, use:
# >>> import nltk
# >>> print(nltk.__file__)
# for me, this returns something similar to 'C:/other/files/specific/to/my/computer/Anaconda3/lib/site-packages/nltk/__init__.py'
# provided with the location of the module's __init__.py, we can navigate to the nltk directory and open the 'data.py' file;
# within the first 100 lines or so of the data.py file, there are listings of common locations on Windows, as well as UNIX & OS X;
# for Windows, corpora will probaly be in local directory in 'AppData'; if the folder is not here, you can... try changing the options
# to show the (possibly) hidden folder, and/or provide yourself or the account in use access to the folder, and/or do as described in
# the sentdex tutorial (by far the easiest route): enter '%APPDATA%' or %appdata% in the top bar of the file browser, this should work;
# next, open the folder titled 'Roaming' and this should have a folder titled 'nltk_data' with the the corpora and other folders inside;
# my full file path here is similar to 'C:/Users/some/local/name/and/then/AppData/Roaming/nltk_data/corpora/' ;

# as previously mentioned, there are many different file types within this folder, so we could surely open some with a simple click or
# maybe a double-click; and of course, for simple text files, we could use normal Python code like 'open()' and/or 'read()' 

# here, we will use some of the methods from nltk to handle the corpus.  we will open the Gutenberg Bible and read the first few lines.

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



