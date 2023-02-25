"""Named Entity Recognition as a sequence tagging task.

Author: Kristina Striegnitz and <YOUR NAME HERE>

<HONOR CODE STATEMENT HERE>

Complete this file for part 2 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import math
import numpy as np
import sys
from memm import MEMM

global name_list

#################################
#
# Word classifier
#
#################################

def getfeats(word, o):
    """Take a word and its offset with respect to the word we are trying
    to classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        (o + 'all alphanumeric', __all_alphanumeric(word)),
        (o + 'capitalized', __capitalized(word)),
        (o + 'all caps', __all_caps(word)),
        (o + 'has hyphen', __has_hyphen(word)),
        (o + 'has apostrophe', __has_apostrophe(word))
    ]
    return features

def __has_apostrophe(word):
    for char in word:
        if char == "'":
            return True
    return False

def __has_equals_sign(word):
    for char in word:
        if char == '=':
            return True
    return False

def __all_alphanumeric(word):
    return word.isalnum()

def __capitalized(word):
    return word[0].isupper()

def __all_caps(word):
    for char in word:
        if char.islower():
            return False
    return True

def __has_hyphen(word):
    for char in word:
        if char == '-':
            return True
    return False


def word2features(sent, i):
    """Generate all features for the word at position i in the
    sentence. The features are based on the word itself as well as
    neighboring words.
    """
    features = []
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features

#################################
#
# Viterbi decoding
#
#################################

def viterbi(obs, memm, pretty_print=False):
    V = [{}]
    path = {}

    initial_observation_features = dict(word2features(obs, 0))
    initial_observation_features['-1label'] = "<S>"
    vectorized_features = memm.vectorize_observations(initial_observation_features)
    initial_state_probs = memm.classifier.predict_log_proba(vectorized_features)

    index = 0
    for state in memm.states:
        V[0][state] = initial_state_probs[0][index]
        path[state] = [state]
        index += 1

    for word_index in range(1, len(obs)):
        V.append({})
        newpath = {}
        for index, state in enumerate(memm.states):
            max_v = float('-inf')
            max_prev_state = None
            for prev_state in memm.states:
                observation_features = dict(word2features(obs, word_index))
                observation_features['-1label'] = prev_state
                vectorized_features = memm.vectorize_observations(observation_features)

                state_probs = memm.classifier.predict_log_proba(vectorized_features)
                v = V[word_index - 1][prev_state] + state_probs[0][index]

                if v > max_v:
                    max_v = v
                    max_prev_state = prev_state
            V[word_index][state] = max_v
            newpath[state] = path[max_prev_state] + [state]
        path = newpath
    if pretty_print:
        pretty_print_trellis(V)
    (prob, state) = max([(V[len(obs) - 1][state], state) for state in memm.states])
    return path[state]


def pretty_print_trellis(V):
    """Prints out the Viterbi trellis formatted as a grid."""
    print("    ", end=" ")
    for i in range(len(V)):
        print("%7s" % ("%d" % i), end=" ")
    print()

    for y in V[0].keys():
        print("%.5s: " % y, end=" ")
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=" ")
        print()

if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    single_sent = train_sents[5]
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []


    for sent in train_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            train_labels.append(sent[i][-1])
            if i == 0:
                feats['-1label'] = "<S>"
            else:
                feats['-1label'] = train_labels[-2]
            train_feats.append(feats)
    labels = []
    feats = []
    for word in range(len(single_sent)):
        single_feats = dict(word2features(single_sent, word))
        labels.append(sent[word][-1])
        if word == 0:
            single_feats['-1label'] = "<S>"
        else:
            single_feats['-1label'] = labels[-2]
        feats.append(single_feats)


    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    yes = feats[5]
    yes_v = vectorizer.transform(yes)

    #sys.exit()

    print("\nTesting ...")
    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_memm.txt. That is the results_memm.txt you should hand
    # in.

    y_pred = []
    states = model.classes_ # 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O'

    vocabulary = 0
    memm = MEMM(states, vocabulary, vectorizer, model)


    for sent in test_sents[:100]:
        y_pred.append(viterbi(sent, memm, False))

    print("Writing to results_memm.txt")
    # format is: word gold pred
    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in test_sents[:100]:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j][i]
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
            j += 1
        out.write("\n")

        print("Now run: python3 conlleval.py results_memm.txt")
