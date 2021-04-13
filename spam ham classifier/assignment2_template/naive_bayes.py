# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]
                                 ham                             spam
                        
    word | spam
      a  |  1
      b  |  2
      c  |  3
      d  |  0

    word | ham
      a  |  1
      b  |  2
      c  |  0
      d  |  1

      laplace smoothing

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set
              [[],[],[]]

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1) 
    P(ham) is positive
    P(spam) = 1-P(ham)
    """
    # TODO: Write your code here
    # return predicted labels of development set

    predict_label = []
    spam_dict = {}
    ham_dict = {}
    
    pos_prior_ham = pos_prior
    pos_prior_spam = 1 - pos_prior

    #create unigram boW
    for i in range (len(train_set)):
        if train_labels[i] == 1:
            for j in range(len(train_set[i])):
                if train_set[i][j] in ham_dict:
                    ham_dict[train_set[i][j]] = ham_dict[train_set[i][j]] +  1
                else:
                    ham_dict[train_set[i][j]] = 1
        if train_labels[i] == 0:
            for j in range(len(train_set[i])):
                if train_set[i][j] in spam_dict:
                    spam_dict[train_set[i][j]] = spam_dict[train_set[i][j]] + 1
                else:
                    spam_dict[train_set[i][j]] = 1  
    # print(ham_dict)
    #calculate the total word occurance of ham
    ham_word_count = 0
    spam_word_count = 0

    for word in ham_dict:
        ham_word_count = ham_word_count + ham_dict[word]
    for word in spam_dict:
        spam_word_count = spam_word_count + spam_dict[word]

    print("ham_count: ", ham_word_count)
    print("spam_count: ", spam_word_count)
    #calculate the probability of each word : P(word|class)
    spam_dict_prob = {} 
    ham_dict_prob = {}   
    for word in ham_dict:
        ham_dict_prob[word] = float(ham_dict[word] + smoothing_parameter) / (ham_word_count + smoothing_parameter * (len(ham_dict)+1))
    print(ham_dict_prob)
    for word in spam_dict:
        spam_dict_prob[word] = float(spam_dict[word] + smoothing_parameter) / (spam_word_count + smoothing_parameter * (len(spam_dict)+1))
        # print("spam_dict_prob[word] : ", spam_dict_prob[word])
    spam_dict_len = len(spam_dict_prob)
    ham_dict_len = len(ham_dict_prob)
    print("spam_dict_len: ", spam_dict_len)
    print("ham_dict_len: ", ham_dict_len)
    #using log to calculate the bayes, compare the value
    
    
    
    for doc in dev_set:
        y_ham = math.log(pos_prior_ham)
        y_spam = math.log(pos_prior_spam)
        for word in doc:
            if word in ham_dict_prob:
                y_ham = y_ham + math.log(ham_dict_prob[word])
            else:
                y_ham = y_ham + math.log((smoothing_parameter) / float(ham_word_count + smoothing_parameter * (ham_dict_len + 1)))
        
            if word in spam_dict_prob:
                y_spam = y_spam + math.log(spam_dict_prob[word])
            else:
                y_spam =  y_spam + math.log(smoothing_parameter / float(spam_word_count + smoothing_parameter * (spam_dict_len + 1)))
        # print("y_ham: ", y_ham)
        # print("y_spam: ", y_spam)
        if y_ham <= y_spam:
            predict_label.append(0)
        else:
            predict_label.append(1)  
    print(predict_label)  
    return predict_label