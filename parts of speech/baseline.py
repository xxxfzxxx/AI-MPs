# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
                test data (list of sentences, no tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        # {word:{tag1:0, tag2:1}}
        
        count_dict = {}
        word_dict = {}
        for sentence in train:
                for (word,tag) in sentence:
                        if tag not in count_dict:
                                count_dict[tag]=1
                        else:
                                count_dict[tag] += 1
                        if word not in word_dict:
                                word_dict[word] = {}
                                word_dict[word][tag] = 1
                        else:
                                if tag not in word_dict[word]:
                                        word_dict[word][tag] = 1
                                else:
                                        word_dict[word][tag] += 1

        max_tag = max(count_dict, key=lambda x:count_dict[x])

        predict = []
        for sentence in test:
                curr_list = []
                for word in sentence:
                        if word in word_dict:
                                max_key = max(word_dict[word], key=lambda x:word_dict[word][x])
                                curr_tup = (word,max_key)
                                curr_list.append(curr_tup)
                        else:
                                curr_list.append((word,max_tag)) 
                predict.append(curr_list)
                
        return predict
