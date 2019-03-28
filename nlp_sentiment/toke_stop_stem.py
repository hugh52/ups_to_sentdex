# please excuse any issues with best practices, i.e. some modules/libs/etc could be imported that are not used until later or maybe never;
#	for now, for me, better safe than sorry...
 
 
import nltk
import pandas as pd
import numpy as np
 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
 
 
# tokenizing words and sentences
example_text = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish_blue. You shouldn't eat cardboard."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))


# stop words
stop_words = set(stopwords.words("english"))
example_sent = "This is a sample sentence, showing off the stop words filtration."
word_tokens = word_tokenize(example_sent)

	# method one - for loop, if statement
##filtered_sentence = []
##for w in word_tokens:
##	if not w in stop_words:
##		filtered_sentence.append(w)

	# method two - list comp
filt_sent = [w for w in word_tokens if not w in stop_words]
print(word_tokens)
print(filt_sent)
 
 
# stemming words
ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
for w in example_words:
	print(ps.stem(w))
new_text = "It is important to be very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)
for w in words:
	print(ps.stem(w))






 
