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
Extra Credit: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

def viterbi_ec(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    laplace_smooth = 1e-5
    tags = set()
    cnt_tag = dict()
    cnt_tag_start = dict()
    cnt_tag_end = dict()
    cnt_tag_pair = dict()
    cnt_word_tag = dict()
    vocabulary = set()
    cnt_word = dict()
    cnt_tag_hapax = dict()
    for sentence in train:
        for k in range(len(sentence)):
            word, tag = sentence[k]
            tags.add(tag)
            val = cnt_tag.get(tag, 0)
            cnt_tag[tag] = val + 1
            if k == 0:
                val = cnt_tag_start.get(tag, 0)
                cnt_tag_start[tag] = val + 1
            else:
                if k == len(sentence) - 1:
                    val = cnt_tag_end.get(tag, 0)
                    cnt_tag_end[tag] = val + 1
                prev_tag = sentence[k - 1][1]
                val = cnt_tag_pair.get((prev_tag, tag), 0)
                cnt_tag_pair[(prev_tag, tag)] = val + 1
            if word in cnt_word_tag:
                val = cnt_word_tag[word].get(tag, 0)
                cnt_word_tag[word][tag] = val + 1
            else:
                cnt_word_tag[word] = {tag: 1}
            vocabulary.add(word)
            val = cnt_word.get(word, 0)
            cnt_word[word] = val + 1

    # Hapax probability
    hapax_p = dict()
    for (word, times) in cnt_word.items():
        if times == 1:
            tag = list(cnt_word_tag[word].keys())[0]
            val = cnt_tag_hapax.get(tag, 0)
            cnt_tag_hapax[tag] = val + 1
    for tag in tags:
        hapax_p[tag] = (cnt_tag_hapax.get(tag, 0) + laplace_smooth) / (
                sum(cnt_tag_hapax.values()) + laplace_smooth * len(tags))

    # Initial probability
    log_initial_p = dict()
    num_of_starting_position = sum(cnt_tag_start.values())
    for tag in tags:
        log_initial_p[tag] = math.log(
            (cnt_tag_start.get(tag, 0) + laplace_smooth) / (
                    num_of_starting_position + laplace_smooth * len(tags)))

    # Transition probability
    log_transition_p = dict()
    for tag_prev in tags:
        for tag_curr in tags:
            log_transition_p[(tag_prev, tag_curr)] = math.log(
                (cnt_tag_pair.get((tag_prev, tag_curr), 0) + laplace_smooth) / (
                        cnt_tag[tag_prev] - cnt_tag_end.get(tag_prev, 0) + laplace_smooth * len(tags)))

    # Emission probability
    log_emission_p = dict()
    unseen = set()
    for sentence in test:
        for word in sentence:
            if word not in vocabulary:
                unseen.add(word)
    for tag in tags:
        total_occurrence_unseen = hapax_p[tag] * len(unseen)
        for word in vocabulary:
            log_emission_p[(tag, word)] = math.log(
                (cnt_word_tag[word].get(tag, 0) + laplace_smooth) / (
                        cnt_tag[tag] + laplace_smooth * (len(vocabulary) + total_occurrence_unseen)))
        for word in unseen:
            log_emission_p[(tag, word)] = math.log(
                laplace_smooth * hapax_p[tag] / (
                        cnt_tag[tag] + laplace_smooth * (len(vocabulary) + total_occurrence_unseen)))

    predicts = []
    for sentence in test:
        trellis = []
        nodes_edges = dict()
        path = dict()
        for k, word in enumerate(sentence):
            if k == 0:
                curr = dict()
                for tag_curr in tags:
                    curr[tag_curr] = log_initial_p[tag_curr] + log_emission_p[(tag_curr, word)]
                trellis.append(curr)
            else:
                prev = trellis[-1]
                curr = dict()
                for tag_curr in tags:
                    for tag_prev in tags:
                        nodes_edges[(tag_prev, tag_curr)] = prev[tag_prev] + log_transition_p[
                            (tag_prev, tag_curr)] + log_emission_p[(tag_curr, word)]
                    select_tag_prev = max(tags, key=lambda x: nodes_edges[(x, tag_curr)])
                    curr[tag_curr] = nodes_edges[(select_tag_prev, tag_curr)]
                    path[(k, tag_curr)] = select_tag_prev
                trellis.append(curr)
        tag = max(tags, key=lambda x: trellis[-1][x])
        res = [(sentence[-1], tag)]
        for k in range(len(sentence) - 1, 0, -1):
            tag = path[(k, tag)]
            res.insert(0, (sentence[k - 1], tag))
        predicts.append(res[:])

    return predicts