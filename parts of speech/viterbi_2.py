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

import math
import numpy as np

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

def viterbi_2(train, test):
        smooth = 1e-5

        tag_set = set()
        word_set = set()
        
        tag_count_dict = {}
        tag_pair_count_dict = {}
        word_tag_count_dict = {}

        START_tag_count_dict = {}
        END_tag_count_dict = {}
        
        
        word_count_dict = {}
        hapax_tag_count = {}

        for sentence in train:
                for k in range(len(sentence)):
                        word, tag = sentence[k]
                        tag_set.add(tag)
                        if tag not in tag_count_dict:
                                tag_count_dict[tag] = 1
                        else:
                                tag_count_dict[tag] += 1
                        if k == 0:
                                if tag not in START_tag_count_dict:
                                        START_tag_count_dict[tag] = 1
                                else:
                                        START_tag_count_dict[tag] += 1
                        else:
                                if k == len(sentence) - 1:
                                        if tag not in END_tag_count_dict:
                                                END_tag_count_dict[tag] = 1
                                        else:
                                                END_tag_count_dict[tag] += 1
                                prev_tag = sentence[k - 1][1]
                                if (prev_tag, tag) not in tag_pair_count_dict:
                                        tag_pair_count_dict[(prev_tag, tag)] = 1
                                else:
                                        tag_pair_count_dict[(prev_tag, tag)] += 1
                        if word in word_tag_count_dict:
                                count = word_tag_count_dict[word].get(tag, 0)
                                word_tag_count_dict[word][tag] = count + 1
                        else:
                                word_tag_count_dict[word] = {tag: 1}
                        word_set.add(word)
                        if word not in word_count_dict:
                                word_count_dict[word] = 1
                        else:
                                word_count_dict[word] += 1
        tag_list = []
        for tag in tag_set:
                tag_list.append(tag)
        
        hapax_tag_prob_dict = {}
        init_prob_dict = {}
        transition_prob_dict = {}
        emission_prob_dict = {}


        for (word, count) in word_count_dict.items():
                if count == 1:
                        tag = list(word_tag_count_dict[word].keys())[0]
                        count = hapax_tag_count.get(tag, 0)
                        hapax_tag_count[tag] = count + 1
        for tag in tag_list:
                hapax_tag_prob_dict[tag] = (hapax_tag_count.get(tag, 0) + smooth) / (
                        sum(hapax_tag_count.values()) + smooth * len(tag_list))


        
        num_of_starting_position = sum(START_tag_count_dict.values())
        for tag in tag_list:
                init_prob_dict[tag] = math.log(
                (START_tag_count_dict.get(tag, 0) + smooth) / (
                        num_of_starting_position + smooth * len(tag_list)))


        
        for tag_prev in tag_list:
                for tag_curr in tag_list:
                        transition_prob_dict[(tag_prev, tag_curr)] = math.log(
                                (tag_pair_count_dict.get((tag_prev, tag_curr), 0) + smooth) / (
                                        tag_count_dict[tag_prev] - END_tag_count_dict.get(tag_prev, 0) + smooth * len(tag_list)))

        emission_prob_dict = {}
        unseen = set()
        for sentence in test:
                for word in sentence:
                        if word not in word_set:
                                unseen.add(word)
        for tag in tag_list:
                total_occurrence_unseen = hapax_tag_prob_dict[tag] * len(unseen)
                for word in word_set:
                        emission_prob_dict[(tag, word)] = math.log(
                                (word_tag_count_dict[word].get(tag, 0) + smooth) / (
                                        tag_count_dict[tag] + smooth * (len(word_set) + total_occurrence_unseen)))
                for word in unseen:
                        emission_prob_dict[(tag, word)] = math.log(
                                smooth * hapax_tag_prob_dict[tag] / (
                                        tag_count_dict[tag] + smooth * (len(word_set) + total_occurrence_unseen)))

        predicts = []
        for sentence in test:
                coor_dict = {}
                trellis = np.zeros((len(tag_list), len(sentence)))
                for s in range(len(tag_list)):
                        trellis[s, 0] = init_prob_dict[tag_list[s]] + emission_prob_dict[(tag_list[s], sentence[0])]
                for o in range(1, len(sentence)):
                        for s in range(len(tag_list)):
                                f = [(trellis[k, o-1] + transition_prob_dict[(tag_list[k], tag_list[s])] +
                                        emission_prob_dict[(tag_list[s], sentence[o])]) for k in range(len(tag_list))]
                                k = np.argmax(f)
                                trellis[s, o] = max(f)
                                coor_dict[(s, o)] = (k, o-1)
                best_path = []
                f = [(trellis[k, len(sentence) - 1]) for k in range(len(tag_list))]
                k = np.argmax(f)
                
                best_path.insert(0, (sentence[len(sentence) - 1], tag_list[k]))
                curr_pos = (k,len(sentence) - 1)
                for i in range(len(sentence) - 1):
                        curr_pos =  coor_dict[curr_pos]
                        k = curr_pos[0]
                        best_path.insert(0, (sentence[len(sentence)- 2 - i], tag_list[k]))
                predicts.append(best_path)

        return predicts