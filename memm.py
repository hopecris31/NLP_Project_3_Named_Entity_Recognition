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

    def vectorize_observations(self, features):
        """Vectorize the observation features."""
        return self.vectorizer.transform(features)

