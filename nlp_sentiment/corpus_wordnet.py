# corpora

# the corpora if basicaly a massive data dump of all kinds of natural language data sets that are definately worth taking a look at;
# we can view them manually, they are simply text files, xml files, etc.
# to find where the nltk module is located in the os, use:
# >>> import nltk
# >>> print(nltk.__file__)
# for me, this returns something similar to 'C:/Users/other/files/specific/to/my/computer/Anaconda3/lib/site-packages/nltk/__init__.py'

# hear, we will use some of the methods from nltk ro handle the corpus.  we will open the Gutenberg Bible and read the first few lines.

from nltk.corpus import gutenberg


# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
	print(tok[x])
