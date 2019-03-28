# chunking

# Once we can confirm the part of speech, we can use chunking, which attempts to group words into hopefully meaningful chunks.  
#  One goal is to try to group into so-called "noun-phrases", which are simply phrases containing a noun, and maybe one or two descriptive
#  parts of speech like a verb and/or adverb.  Overall idea is to group nouns with words that relate to or ideally describe the noun.
#  We will combine part of speech tags with regular expressions (some regex shown before code with more generous description(s) after.

#  '+' = match 1 or more
#  '?' = match 0 or 1 repetitions
#  '*' = match 0 or more repetitions
#  '.' = any character except a new line
#  part of speech tags are denoted with "<" and ">"; we can place regexs within the tags to account for things like "all nouns" ('<N.*>')


import nltk
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize


def chunking():
  try:
    for i in tokenized
