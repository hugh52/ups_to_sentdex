

# the next approach we will try, since the remaining "scores" are relatively close, is to use all algos at once; we will create another
# classifier, and make the result of that classifier based on what the other algorithms said; similar to a voting system, so we will
# just need an odd number of algorithms...

# combining classifier algorithms is a common technique, accomplished by creating a sort of voting system, where each algorithm gets one
# vote, and the classification that has the most votes is the chosen one; to do this, we want our new classifier to act like a typical
# nltk classifier, with all of the methods; so we just need to use object oriented programming and ensure to inherit from the nltk
# classifier class.

from nltk.classify import ClassifierI
from statistics import mode







































