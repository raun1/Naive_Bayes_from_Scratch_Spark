# Naive_bayes
Basic_naive_Bayes code

This was Project 1 of DSP Spring 2018. 

## Problem Definition
Classify the Reuters Corpus, taking into account only labels ending in CAT i.e. CCAT, ECAT, GCAT, MCAT.


## Built With/Things Needed to implement experiments

* [Python](https://www.python.org/downloads/) - Python-2 
* [Numpy](http://www.numpy.org/) - Numpy
* [reuters dataset as url]
* [Spark](https://spark.apache.org/downloads.html) - Apache spark download


## Approach 
Build Basic Naive bayes from scratch in Apache Spark

Pre_Processing Step - Use the stopwords file of Stanford_NLP
Basic Flow of the algorithm - 

```

python2 naive_bayes.py <url_train> <url_test> <vsmall/small/large>"
Step 1 > Obtain the necessary files using Wget.
Step 2 > Preprocess the training set i.e. strip remove punctuations and stopwords etc
Step 3 > Preprocess the lables, if two labels eg CCAT and MCAT duplicate corresponding training data 
Step 4 > Do a join and split into two train and test
Step 5 > Calculate Prior Probabilities of the four classes and broad cast them.
Step 6 > Generate the Vocab and the frequency per class for every unique word in vocab
Step 7 > Broadcast the above Vocab/frequency array
Step 8 > Use the vocab/frequency array + the priors to compute the class of the new document

Accuracy acheived 92% :(

```


## Contributors 

* [Raunak Dey](https://github.com/raun1) - Main Coding of the function and Naive Bayes file 
* [Aishwarya Jagtap](-) - Documentation

## References 

Stopword.txt file used from NLP Stopword - https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
Project definition - https://github.com/dsp-uga/sp18/blob/master/projects/p1/project1.pdf

