"""Named Entity Recognition as a sequence tagging task.

Author: Kristina Striegnitz and Hope Crisafi

I affirm that I have carried out the attached academic endeavors with
full academic honesty, in accordance with the Union College Honor Code and the course
syllabus.

Complete this file for part 2 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import math
import numpy as np

import memm
# TODO (optional): Complete the class MEMM
from memm import MEMM


def viterbi(obs, memm, pretty_print=False):
    V = [{}]
    path = {}

    initial_observation_features = getfeats(obs[0], 0)
    initial_observation_features.append("<S>")
    initial_state_probs = memm.classifier.predict_log_proba(initial_observation_features)

    index = 0
    for state in memm.states:
        V[0][state] = initial_state_probs[index]
        path[state] = [state]
        index += 1

    for word in range(1, len(obs)):
        V.append({})
        newpath = {}
        for state in memm.states:
            max_v = float('-inf')
            max_prev_state = None
            for prev_state in memm.states:
                state_probs = memm.classifier.predict_log_proba([word2features(obs, word)], prev_state)
                for probability in state_probs:
                    if probability > max_v:
                        max_v = probability
                        max_prev_state = prev_state
                V[word][state] = max_v
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

#get first word in sentence features
#append start token to beginning of sentennce, so this token becomes the previous token to get prob with first work
#get list of probabilities from the classifier for each state

#loop through each state to go through each part of the colus for the first word, and append the corresponding prob to each cell

#V[0] is first column, states are keys, so use as index to get proper cell

#V[0] is first co.umn,  then go through each for in that first column and retreive the corresponding prob for that state from the list of probs got from classifier in previous step
#next go through rest of words in sentenc, for each word in that sentence, go through each row,
#fill each cell with the max probability that you find, start by setting max_v to negative infinity.
#then go through each state, and for each state, get the prob from the classifier, and multiply it by the prob from the previous column that corresponds to the state that you are currently in.
#
#then compare that to the max_v, if it is greater than max_v, then set max_v to that value, and set the state that you are currently in to the state that you are currently in.
#go through each row in the column, find the max_v for each word, and set the cell to the max_v that you found for that column
#find the prob of that feature representation of that state, being in that state, and multiply it by the prob of the previous state being in that state
#then find the max of all the states, and put the prob of that state in that cell
#then repeat that process for each row in the column, and then move on to the next column
#################################
#
# Word classifier
#
#################################




def getfeats(word, o):
    """Take a word its offset with respect to the word we are trying to
    classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        (o + 'word.istitle', word.istitle()),
        (o + 'word.islower', word.islower()),
        (o + 'word.isidentifier', word.isidentifier()),
    ]
    return features
    

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




if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = dict(word2features(sent,i))
            # TODO: training needs to take into account the label of
            # the previous word. And <S> if i is the first words in a
            # sentence.
            train_labels.append(sent[i][-1])
            if i == 0:
                feats['-1label'] = "<S>"
            else:
                feats['-1label'] = train_labels[-2]
            train_feats.append(feats)
            #train on words, get features for each word.  One of the features is the previous label
            #for each sentence, for each word, getting the test label and putting it in list of train_labels
            #use the previous words label as a feature in our training for each word
            #if word is first word in sentence, then use <S> as the previous label, else: then the feature is the label for the word before the considered word
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.

    model = LogisticRegression(max_iter=600)
    model.fit(X_train, train_labels)

    print("\nTesting ...")
    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_memm.txt. That is the results_memm.txt you should hand
    # in.
    y_pred = []
    states = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'O']
    vocabulary = 0
    memm = MEMM(states, vocabulary, vectorizer, model)
    for sent in dev_sents[:100]:
        for i, word in enumerate(sent):
            features = dict(word2features(sent, i))
            y_pred.append(viterbi(word, memm, pretty_print=True))
    #go through each word in the sentence, get the features for that word, and then use the classifier to get the prob for each state

    print("Writing to results_memm.txt")
    # format is: word gold pred
    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_memm.txt")






