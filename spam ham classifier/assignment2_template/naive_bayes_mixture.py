# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""



import math
def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set

    predict_label = []
    w_spam_dict = {}
    w_ham_dict = {}
    b_ham_dict = {}
    b_spam_dict = {}
    pos_prior_ham = pos_prior
    pos_prior_spam = 1 - pos_prior

    #create unigram boW
    for i in range (len(train_set)):
        if train_labels[i] == 1:
            for j in range(len(train_set[i])):
                if train_set[i][j] in w_ham_dict:
                    w_ham_dict[train_set[i][j]] = w_ham_dict[train_set[i][j]] +  1
                else:
                    w_ham_dict[train_set[i][j]] = 1
        if train_labels[i] == 0:
            for j in range(len(train_set[i])):
                if train_set[i][j] in w_spam_dict:
                    w_spam_dict[train_set[i][j]] = w_spam_dict[train_set[i][j]] + 1
                else:
                    w_spam_dict[train_set[i][j]] = 1  
    
    #create bigram boW
    for i in range (len(train_set)):
        if train_labels[i] == 1:
            for j in range(len(train_set[i]) - 1):
                b_word = train_set[i][j] + " " + train_set[i][j + 1]
                if b_word in b_ham_dict:
                    b_ham_dict[b_word] = b_ham_dict[b_word] +  1
                else:
                    b_ham_dict[b_word] = 1
        if train_labels[i] == 0:
            for j in range(len(train_set[i]) - 1):
                b_word = train_set[i][j] + " " + train_set[i][j + 1]
                if b_word in b_spam_dict:
                    b_spam_dict[b_word] = b_spam_dict[b_word] + 1
                else:
                    b_spam_dict[b_word] = 1  

    # for keys,values in b_ham_dict.items():
    #     print(keys)
    #     print(values)
    #calculate the total word occurance of ham
    w_ham_word_count = 0
    w_spam_word_count = 0
    b_ham_word_count = 0
    b_spam_word_count = 0
    for word in w_ham_dict:
        w_ham_word_count = w_ham_word_count + w_ham_dict[word]
    for word in w_spam_dict:
        w_spam_word_count = w_spam_word_count + w_spam_dict[word]
    for word in b_ham_dict:
        b_ham_word_count = b_ham_word_count + b_ham_dict[word]
    for word in b_spam_dict:
        b_spam_word_count = b_spam_word_count + b_spam_dict[word]
    #总字数
    print("w_ham_count: ", w_ham_word_count)
    print("w_spam_count: ", w_spam_word_count)
    print("b_ham_count: ", b_ham_word_count)
    print("b_spam_count: ", b_spam_word_count)
    #calculate the probability of each word : P(word|class)
    w_spam_dict_prob = {} 
    w_ham_dict_prob = {}  
    b_ham_dict_prob = {}
    b_spam_dict_prob = {}
    for word in w_ham_dict:
        w_ham_dict_prob[word] = float(w_ham_dict[word] + unigram_smoothing_parameter) / (w_ham_word_count + unigram_smoothing_parameter * (len(w_ham_dict)+1))
        # print("w_ham_dict_prob[word] : ", w_ham_dict_prob[word])
    for word in w_spam_dict:
        w_spam_dict_prob[word] = float(w_spam_dict[word] + unigram_smoothing_parameter) / (w_spam_word_count + unigram_smoothing_parameter * (len(w_spam_dict)+1))
        # print("w_spam_dict_prob[word] : ", w_spam_dict_prob[word])
    for word in b_ham_dict:
        b_ham_dict_prob[word] = float(b_ham_dict[word] + bigram_smoothing_parameter) / (b_ham_word_count + bigram_smoothing_parameter * (len(b_ham_dict)+1))
        # print("b_ham_dict_prob[word] : ", b_ham_dict_prob[word])
    for word in b_spam_dict:
        b_spam_dict_prob[word] = float(b_spam_dict[word] + bigram_smoothing_parameter) / (b_spam_word_count + bigram_smoothing_parameter * (len(b_spam_dict)+1))
        # print("b_spam_dict_prob[word] : ", b_spam_dict_prob[word])

    w_spam_dict_len = len(w_spam_dict_prob)
    w_ham_dict_len = len(w_ham_dict_prob)
    print("w_spam_dict_len: ", w_spam_dict_len)
    print("w_ham_dict_len: ", w_ham_dict_len)
    b_spam_dict_len = len(b_spam_dict_prob)
    b_ham_dict_len = len(b_ham_dict_prob)
    print("b_spam_dict_len: ", b_spam_dict_len)
    print("b_ham_dict_len: ", b_ham_dict_len)
    #using log to calculate the bayes, compare the value
    for doc in dev_set:
        w_y_ham =  math.log(pos_prior_ham) 
        w_y_spam = math.log(pos_prior_spam)
        b_y_ham =  math.log(pos_prior_ham) 
        b_y_spam = math.log(pos_prior_spam)  
        for word in doc:
            if word in w_ham_dict_prob:
                w_y_ham = w_y_ham + math.log(w_ham_dict_prob[word]) 
            else:
                w_y_ham = w_y_ham + math.log((unigram_smoothing_parameter) / float(w_ham_word_count + unigram_smoothing_parameter * (w_ham_dict_len + 1)))
            if word in w_spam_dict_prob:
                w_y_spam = w_y_spam + math.log(w_spam_dict_prob[word]) 
            else:
                w_y_spam = w_y_spam + math.log(unigram_smoothing_parameter / float(w_spam_word_count + unigram_smoothing_parameter * (w_spam_dict_len + 1)))
        w_y_ham = w_y_ham * (1-bigram_lambda)
        w_y_spam = w_y_spam * (1-bigram_lambda)
        for i in range(len(doc)-1):
            b_word = doc[i] + " " + doc[i+1]
            if b_word in b_ham_dict_prob:
                b_y_ham = b_y_ham + math.log(b_ham_dict_prob[b_word])
            else:
                b_y_ham = b_y_ham + math.log((bigram_smoothing_parameter) / float(b_ham_word_count + bigram_smoothing_parameter * (b_ham_dict_len + 1)))
            if b_word in b_spam_dict_prob:
                b_y_spam = b_y_spam + math.log(b_spam_dict_prob[b_word])
            else:
                b_y_spam = b_y_spam + math.log((bigram_smoothing_parameter) / float(b_spam_word_count + bigram_smoothing_parameter * (b_spam_dict_len + 1)))
        b_y_ham = b_y_ham * bigram_lambda
        b_y_spam = b_y_spam * bigram_lambda
        y_ham = b_y_ham + w_y_ham
        y_spam = b_y_spam + w_y_spam
        if y_ham <= y_spam:
            predict_label.append(0)
        else:
            predict_label.append(1)  
    return predict_label