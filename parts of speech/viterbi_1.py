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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math
import numpy as np
def viterbi_1(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
                [[word1, word2, word3], [word3, word4]]
        output: list of sentences with tags on the words
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        word_for_t = {}
        for sentence in train:
                for (word, tag) in sentence:
                        if tag not in word_for_t:
                                word_for_t[tag] = set()
                                word_for_t[tag].add(word)
                        if tag in word_for_t:
                                word_for_t[tag].add(word)
        word_set = set()
        tag_set = set()
        for sentence in train:
                for (word, tag) in sentence:
                        word_set.add(word)
                        tag_set.add(tag)
        tag_list = []
        for tag in tag_set:
                tag_list.append(tag)
        
        tags_count_dict = {} 
        tag_pair_count_dict = {} 
        tag_word_count_dict = {} 

        for sentence in train:
                for i in range(len(sentence)):
                        tag = sentence[i][1]
                        if tag not in tags_count_dict:
                                tags_count_dict[tag] = 1
                        else:
                                tags_count_dict[tag] = tags_count_dict[tag] + 1

                        tag_word = (sentence[i][1], sentence[i][0])
                        if tag_word not in tag_word_count_dict:
                                tag_word_count_dict[tag_word] = 1
                        else:
                                tag_word_count_dict[tag_word] = tag_word_count_dict[tag_word] + 1
        
        for sentence in train:
                for i in range(len(sentence) - 1):
                        tag_pair = (sentence[i][1], sentence[i+1][1])
                        if tag_pair not in tag_pair_count_dict:
                                tag_pair_count_dict[tag_pair] = 1
                        else:
                                tag_pair_count_dict[tag_pair] = tag_pair_count_dict[tag_pair] + 1
        
                        
        

        init_prob_dict = {}
        transition_prob_dict = {}
        emission_prob_dict = {}
        
        smooth = 1e-5
        for key, value in tags_count_dict.items():
                init_prob_dict[key] = math.log(value / sum(tags_count_dict.values()) )
        for key, value in tag_pair_count_dict.items():
                transition_prob_dict[key] = math.log((value + smooth)/ (tags_count_dict[key[0]] + smooth * len(tag_list)))
        for key, value in tag_word_count_dict.items():
                emission_prob_dict[key] = math.log((value + smooth) / (tags_count_dict[key[0]] + smooth * len(word_for_t[key[0]])))
 

        predict = []
        
        for sentence in test:
                coor_dict = {}
                trellis = np.zeros((len(tag_list), len(sentence)))
                for s in range(len(tag_list)):
                        if (tag_list[s], sentence[0]) not in emission_prob_dict:
                                emission_prob_dict[(tag_list[s], sentence[0])] = math.log(smooth / (tags_count_dict[key[0]] + smooth * (len(word_for_t[key[0]])+1)))
                        trellis[s, 0] = init_prob_dict[tag_list[s]] + emission_prob_dict[(tag_list[s], sentence[0])]
                for o in range(1, len(sentence)):
                        for s in range(len(tag_list)):
                                f = []
                                for k in range(len(tag_list)):
                                        if (tag_list[k], tag_list[s]) not in transition_prob_dict:
                                                transition_prob_dict[(tag_list[k], tag_list[s])] = math.log(smooth / (tags_count_dict[key[0]] + 1 + smooth * len(tag_list)))
                                        if (tag_list[s], sentence[o]) not in emission_prob_dict:
                                                emission_prob_dict[(tag_list[s], sentence[o])] = math.log(smooth / (tags_count_dict[key[0]] + smooth * (len(word_for_t[key[0]])+1)))
                                f = [(trellis[k, o-1] + transition_prob_dict[(tag_list[k], tag_list[s])] + emission_prob_dict[(tag_list[s], sentence[o])]) for k in range(len(tag_list))]
                                k = np.argmax(f)
                                trellis[s, o] = max(f)
                                coor_dict[(s,o)] = (k ,o-1)
                best_path = []
                f = [(trellis[k, len(sentence) - 1]) for k in range(len(tag_list))]
                k = np.argmax(f)
                best_path.insert(0, (sentence[len(sentence) - 1], tag_list[k]))
                curr_pos = (k,len(sentence) - 1)
                for i in range(len(sentence) - 1):
                        curr_pos =  coor_dict[curr_pos]
                        k = curr_pos[0]
                        best_path.insert(0, (sentence[len(sentence)- 2 - i], tag_list[k]))
                predict.append(best_path)  
              
        return predict
