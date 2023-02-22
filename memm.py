import numpy as np

class MEMM:
    def __init__(self, states, vocabulary, vectorizer, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.states = states
        self.vocabulary = vocabulary
        self.vectorizer = vectorizer
        self.classifier = classifier

    def get_state_probabilities(self, features, previous_tag):
        """Given a dictionary of features representing a word and the tag chosen for the
        previous word, return the probabilities of each of the MEMM's states.
        """
        #TODO FIll in this code (optional)
        #return state_probabilities

