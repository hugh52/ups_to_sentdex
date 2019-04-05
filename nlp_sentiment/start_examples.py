

# This assumes NLTK (Natural Language Toolkit) is installed with no issues importing the packing into your current python environment.
# If nltk is not installed, please use 'pip install nltk' or 'conda install nltk' or install it however you want, then continue.
# Once this is confirmed, we will need to download some information that comes with nltk, like various datasets.


import nltk

# please be advised that this download can take up a significant amount of room, depending on options selected, etc.  
# please refer to outside sources if you are having difficulty during this part of the tutorial series.
nltk.download()

# some helpful definitions below, courtesy of the Py-God sentdex at 
#     https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/  
# 'Corpus - Body of text, singular. Corpora is the plural of this. Example: A collection of medical journals.'
# 'Lexicon - Words and their meanings. Example: English dictionary. Consider, however, that various fields will have different lexicons.
# For example: To a financial investor, the first meaning for the word "Bull" is someone who is confident about the market, as compared 
# to the common English lexicon, where the first meaning for the word "Bull" is an animal. As such, there is a special lexicon for 
# financial investors, doctors, children, mechanics, and so on.'
# 'Token - Each "entity" that is a part of whatever was split up based on rules. For examples, each word is a token when a sentence is    # "tokenized" into words. Each sentence can also be a token, if you tokenized the sentences out of a paragraph.'

# now short practice:


from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today?  The weather is great, and Python is awesome.  The sky is pinkish-blue.  You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT))

