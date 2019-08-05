

# the results I produced were higher in all accuracies, as well as higher ratios within the list of most_informative_features; this
# could just be reading it wrong or doing something incorrectly, improvement in modules since tutorial originally came out, just pure
# luck, etc;

# according to the website, SVC had an accuracy percentage around 45%, meaning it is wrong more often than right out of the gate, so
# that one could be dumped; I will go ahead and do this to follow along, although my SVC showed 72%, still the lowest but not by much.



# THE BELOW IS OPTIONAL NOT NECESSARY, JUST MY RAMBLINGS, AND QUITE POSSIBLY CONTAINS INCORRECT INFORMATION.

# I will have to study these in depth and am not certain of the abbreviations until then, but from my experience I will attempt to 
# tell what these terms stand for; unfortunately, even a surface level conceptual exaplanation of how they work and the particulars of 
# each is beyond the scope of this project, so I will try not to get into that, as I would never complete this if I did.

# starting at top and working down... ending 'NB': of course 'NaiveBayes'; 'SGD': stochastic gradient descent; 
# 'svm' from imports: support vector machines; 'SVC': support vector classification; 'LinearSVC': linear support vector classification;
# 'NuSVC': still support vector classification, but instead of using variable C, like traditional svm approaches, Nu is used;
# the main difference is C is unbounded to the upside...

# ok so first the maximum-margin hyperplane is a hyperplane that seperates, say two seperate "clouds" of points in space, so that the 
# distance from each "cloud" of points to this hyperplane is equal; then the margin from each cloud and hyperplane is at a maximum.
# so in traditional svm used for classification, there is a parameter 'C' that has no upper bound but is greater than 0;
# rather reluctantly, I will say this parameter decides the amount of error we are willing to accept in our model; its more comlicated

# one explanation: we look for a solution that will either fail often but the errors will be small, or it fails rarely but when it does,
# the errors are quite large;
# another explanation: large C means model searches for a small margin, meaning most points are included, while a small C seeks a large
# margin, which may not include all or many variables used for training.  
# in general, C says how much or how little we care about the model overfitting the data.

# Nu can control the number of support vectors and training errors and must lie between 0 and 1.
# it is considered an upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors -
# so when using .1, then at most 10% of training examples will be misclassified while at least 10% of the training examples will 
# be support vectors; support vectors are samples that lie on the margins of the maximum-margin hyperplane 

# in general, support vector classification works by finding seperating hyperplanes in higher dimensions while balancing the inclusion
# of training variables with the chance of over-fitting.

# as for C and Nu, they are really about the same, except C must be (or will ideally be) optimized, requiring additional steps.

# and that right there is why I will stop providing explanations or even guesses; please note the info above may be way off and 
# COMPLETELY INCORRECT, and it is also my personal ramblings from readings about SVMs.  so, moving along now.....


