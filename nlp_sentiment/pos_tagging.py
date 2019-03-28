

import nltk
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


# PunktSentenceTokenizer can be trained on any body of text, since it is capable of unsupervised machine learning

# create training and testing data
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
# train the Punkt tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
# then actually tokenize...
tokenized = custom_sent_tokenizer.tokenize(sample_text)


# nltk can do Part of Speech tagging, essentially labelling words in a sentence as nouns, adjectives, verbs, etc. It can also label by
#  tense and more. A list of tags, what they mean, and some examples should be located in a separate folder.

# now, create a function that will run through and tag all of the parts of speech per sentence.

def process_content():
	try:
    for i in tokenized[:5]:
      words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			print(tagged)
	except Exception as e:
		print(str(e))
process_content()

