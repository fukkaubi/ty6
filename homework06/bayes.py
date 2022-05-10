import itertools
import math
import operator
import typing as tp
from collections import Counter, defaultdict
from copy import deepcopy


class NaiveBayesClassifier:
    def __init__(self, alpha: float) -> None:
        if not (alpha <= 1.0 and alpha > 0.0):
            raise ValueError("Smoothing parameter must be between 0.0 and 1.0")
        self.priors: tp.Counter[str] = Counter()
        self.alpha: float = alpha
        self.doc_count: int = 0
        self.unique_words: tp.List[str] = []
        self.words_per_class: tp.DefaultDict[str, tp.DefaultDict[str, float]] = defaultdict(
            defaultdict
        )
        self.class_lengths: tp.DefaultDict[str, float] = defaultdict(float)
        self.word_probs: tp.DefaultDict[str, tp.DefaultDict[str, float]] = defaultdict(defaultdict)

    def fit(self, X: tp.List[str], y: tp.List[str]) -> None:
        """ Fit Naive Bayes classifier according to X, y. """
        if not X or not y or not (len(X) == len(y)):
            raise ValueError("The training set is either unmarked or empty.")
        self.doc_count = len(X)
        self.priors = Counter(y)
        for e in self.priors:
            self.priors[e] /= len(y)  # type: ignore
        unique_words_prepare = [i.split(" ") for i in X]
        self.unique_words = list(itertools.chain.from_iterable(unique_words_prepare))
        self.unique_words = sorted(list(set(self.unique_words)))
        self.words_per_class = defaultdict(defaultdict)
        self.class_lengths = defaultdict(float)
        for i, string in enumerate(X):
            words = string.split(" ")
            c = y[i]
            self.class_lengths[c] += len(words)
            for word in words:
                if not word in self.words_per_class.keys():
                    self.words_per_class[word] = defaultdict(
                        float,
                        {
                            key: value
                            for (key, value) in zip(list(set(y)), [0 for _ in list(set(y))])
                        },
                    )
                self.words_per_class[word][c] += 1
        self.word_probs = deepcopy(self.words_per_class)
        for word, value in self.word_probs.items():
            for c, _ in value.items():
                self.word_probs[word][c] = (self.word_probs[word][c] + self.alpha) / (
                    self.class_lengths[c] + self.alpha * len(self.unique_words)
                )

    def predict(self, X: tp.List[str]) -> tp.List[str]:
        """ Perform classification on an array of test vectors X. """
        if (
            not self.priors
            or not self.doc_count
            or not self.unique_words
            or not self.words_per_class
            or not self.class_lengths
            or not self.word_probs
        ):
            raise ValueError("The model is untrained")
        labels = []
        for document in X:
            words = document.split(" ")
            probabilities: tp.DefaultDict[str, float] = defaultdict(float)
            for c in self.class_lengths.keys():
                word_probs = []
                for word in words:
                    if word in self.unique_words:
                        word_probs.append(math.log(self.word_probs[word][c]))
                word_probs_sum = sum(word_probs)
                probabilities[c] = math.log(self.priors[c]) + word_probs_sum
            predicted_label = max(probabilities.items(), key=operator.itemgetter(1))[0]
            labels.append(predicted_label)
        return labels

    def score(self, X_test: tp.List[str], y_test: tp.List[str]) -> float:
        """ Returns the mean accuracy on the given test data and labels. """
        predicted = self.predict(X_test)
        class_accuracies = defaultdict(float)
        for c in list(set(y_test)):
            if y_test.count(c):
                true_positives = sum(
                    [1 for i, e in enumerate(predicted) if e == c and y_test[i] == c]
                )
                false_negatives = sum(
                    [1 for i, e in enumerate(predicted) if e != c and y_test[i] == c]
                )
                class_accuracies[c] = true_positives / (true_positives + false_negatives)
        score = sum([i for i in class_accuracies.values()]) / len(list(set(y_test)))
        return score
