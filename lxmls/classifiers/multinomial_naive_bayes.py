import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        print(self.smooth)
        
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # ----------
        # Solution to Exercise 1
        
        # for each class, compute the prior and likelihood
        for i in range(n_classes):
            
            # get the documents which are equal to that class
            docs_of_class, _ = np.nonzero(y == classes[i])  
            
            # the prior is the ratio of documents which are that class
            prior[i] = len(docs_of_class) / n_docs  

            # get the word occurrences in those documents
            word_count_in_class = x[docs_of_class, :].sum(0)
            total_words_in_class = word_count_in_class.sum()  # total_words_in_class = total number of words in documents of class i
            
            # update the likelihood for each word
            likelihood[:, i] = word_count_in_class / total_words_in_class
            
            # if smoothed
            #likelihood[:, i] = (1 + word_count_in_class ) / (1 + total_words_in_class )

            
            
        #raise NotImplementedError("Complete Exercise 1")

        # End solution to Exercise 1
        # ----------

        params = np.zeros((n_words+1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
