

import nltk
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


# PunktSentenceTokenizer can be trained on any body of text (ok most...), since it is capable of unsupervised machine learning.

# to find file path for corpora, enter 'print(nltk.__file__)' into terminal; output is destination to nltk folder. Next, open 'data.py'.
# within first 100 lines are generic file paths for different operating systems.  
# NOTE: 'nltk.download()' must be entered before these datasets will download; this was explained at the start of the tutorial.
# a list (up-to-date, I think?) of what is contained in the corpora can be found at 'http://www.nltk.org/nltk_data/'

# create training and testing data
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
# train the Punkt tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# then actually tokenize...
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# nltk can do Part of Speech tagging, essentially labelling words in a sentence as nouns, adjectives, verbs, etc. It can also label by
# tense and more. A list of tags, what they mean, and some examples should be located in a separate folder.

# now, we create a function that will run through and tag all of the parts of speech per sentence.

def process_content():
	try:
		for i in tokenized[:5]:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			print(tagged)
	except Exception as e:
		print(str(e))

process_content()


# chunking

# once part of speech is confirmed, use chunking to group words into (hopefully) meaningful chunks. one goal is to group into so-called 
#  "noun-phrases", which are simply phrases containing a noun, plus maybe one or two descriptive parts of speech (verb, adverb, etc.)
#  overall idea is group nouns with words related to or ideally describing the noun. below we combine part of speech tags with regex.

# '+' = match 1 or more
# '?' = match 0 or 1 repetitions
# '*' = match 0 or more repetitions
# '.' = any character except a new line
# part of speech tags are denoted with "<" and ">"; we can place regexs within the tags to account for things like "all nouns" ('<N.*>')


def chunking_one():
  try:
    for i in tokenized[:5]:
	    words = nltk.word_tokenize(i)
	    tagged = nltk.pos_tag(words)
	    chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
	    chunkParser = nltk.RegexpParser(chunkGram)
	    chunked = chunkParser.parse(tagged)
	    chunked.draw()
	    ##print(chunked)
	    ##print(list(chunked))
	except Exception as e:
	  print(str(e))
		
chunking_one()

# breaking down regex line: 'chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""'
# '<RB.?>*' = 0 or more of any tense of adverb
# '<VB.?>*' = 0 or more of any tense of verb
# '<NNP>+' = one or more proper nouns
# '<NN>?' = zero or on singular noun
# can be useful at times to print out the specific chunks
# if we want to access the data via our program and not just visually, what may have been seen in the printed output is that our
#  "chunked" variable is an nltk tree. each "chunk" and non-chunk" is a "subtree" of the tree. these can be referenced using something
#  like 'chunked.subtrees', which will then allow us to iterate through these subtrees like the following methods show.
# 'for subtree in chunked.subtrees():'
# 	'print(subtree)'

def chunking_two():
  try:
    for i in tokenized[:5]:
      words = nltk.word_tokenize(i)
      tagged = nltk.pos_tag(words)
      chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
      chunkParser = nltk.RegexpParser(chunkGram)
      chunked = chunkParser.parse(tagged)
      #chunked.draw()
      #print(chunked)
      #print(list(chunked))
      print(chunked)
      for subtree in chunked.subtrees():
        print(subtree)
      chunked.draw()
  except Exception as e:
    print(str(e))

chunking_two()


# using filter parameter in 'chunked.subtree()' call can return/print only the chunks, ignoring the rest; full implementation below.
# 'for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):'
#     'print(subtree)'

def chunking_three():
  try:
    for i in tokenized[:5]:
      words = nltk.word_tokenize(i)
      tagged = nltk.pos_tag(words)
      chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
      chunkParser = nltk.RegexpParser(chunkGram)
      chunked = chunkParser.parse(tagged)
      
      print(chunked)
      
      for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
        print(subtree)
        
      chunked.draw()
      
  except Exception as e:
    print(str(e))
    
chunking_three()

# note: above, we are filtering to show only subtrees with the label of "Chunk" but not "Chunk" as in the nltk chunk attribute; "Chunk"
#  is used here simply because that is the label we assigned to it when using the function line: 
#         'chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""'
#  so, if instead we used something like 'chunkGram = r"""Pythons: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""' we would filter by label "Pythons"



# chinking

# chinking can provide a way to chunk almost everything, oftentimes leftover elements not chunked with chunking methods.
# chinking basically just removes a chunk from a chunk, and the chunk removed from the chunk is called the chink.
# similar code is used as well, but denote the chink after the chunk with }{ instead of the chunk's {}.

def chinking():
  try:
    for i in tokenized[:5]:
      words = nltk.word_tokenize(i)
      tagged = nltk.pos_tag(words)
      
      chunkGram = r"""Chunk: {<.*>+}
                              }<VB.?|IN|DT|TO>{"""
      
      chunkParser = nltk.RegexpParser(chunkGram)
      chunked = chunkParser.parse(tagged)
      
      chunked.draw()
      
  except Exception as e:
    print(str(e))
    
chinking()
      
# obviously, the main difference is '}<VB.?|IN|DT|TO>+{' 
#  which means remove from the chink one or more verbs, prepositions, determiners, or the word 'to'


# named entity recognition

# this is a form of chunking that comes with nltk, so in that regard it is unlike the custom forms of chunking shown above.
# idea is to have the machine immediately be able to pull our "entities", like people, places, things, locations, monetary figures, etc.

# two major options with nltk's ner:
#  1) recognize all named entities
#  2) recognize entity as its respective type, for example, people, places, things, locations, etc.

def ner_one():
  try:
		for i in tokenized[:5]:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			namedEnt = nltk.ne_chunk(tagged, binary=True)
			namedEnt.draw()
  except Exception as e:
			print(str(e))
			
ner_one()

	
def ner_two():
  try:
		for i words in tokenized[:5]:
			words = nltk.tokenize(i)
			tagged = nltk.pos_tag(words)
			namedEnt = nltk.ne_chunk(tagged, binary=False)
			namedEnt.draw()
			
  except Exception as e:
		print(str(e))

ner_two()


# the second method will split up terms like "White House" into "White" and "House" as if they are different, while the first method
#  correctly returns the combined term "White House"
#  different types of Named Entities that will show up with binary set to False, shown below
#  NE Types and Examples
# ORGANIZATION - Georgia-Pacific Corp., WHO
# PERSON - Eddy Bonte, President Obama
# LOCATION - Murray River, Mount Everest
# DATE - June, 2008-06-29
# TIME - two fifty a m, 1:30 p.m.
# MONEY - 175 million Canadian Dollars, GBP 10.40
# PERCENT - twenty pct, 18.75 %
# FACILITY - Washington Monument, Stonehenge
# GPE - South East Asia, Midlothian



# lemmatizing

# similar to stemming; major difference is stemming can often create non-existant words, whereas lemmas are actual words
#  root stem cannot be looked up in a dictionary whereas a lemma could be
#  sometimes word will be similar while other times it will be very different

from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos='a'))
print(lemmatizer.lemmatize("best", pos='a'))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run", 'v'))

# lemmatize takes a part of speech parameter "pos"
# if "pos" parameter is left blank, default is set to "noun"; can cause problems because machine will attempt to find the nearest nouns


