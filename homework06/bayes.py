from collections import Counter
from math import e, log


class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1e-5) -> None:
        # self.d = 0
        # self.word = defaultdict(lambda: 0)  # type: ignore
        # self.class_words = defaultdict(lambda: 0)  # type: ignore
        # self.y = defaultdict(lambda: 0)  # type: ignore
        self.alpha = alpha
        self.model: dict = {
            "labels": {},
            "words": {},
        }

    def fit(self, X, y) -> None:
        """ Fit Naive Bayes classifier according to X, y."""
        words_list = []
        for sentence, lable in zip(X, y):
            for word in sentence.split():
                words_list.append((word, lable))

        self.words_labels = Counter(words_list)
        print("words_labels", self.words_labels)
        self.counted_labels: dict = dict(Counter(y))
        print("counted_labels", self.counted_labels)
        words = [word for sentence in X for word in sentence.split()]
        self.counted_words = dict(Counter(words))
        print("counted_words", self.counted_words)
        self.model = {
            "labels": {},
            "words": {},
        }

        for var_label in self.counted_labels:
            params = {
                "label_count": self.count_words(var_label),
                "probability": self.counted_labels[var_label] / len(y),
            }
            self.model["labels"][var_label] = params
        for word in self.counted_words:
            params = {}
            for var_label in self.counted_labels:
                params[var_label] = self.smoothing(word, var_label)
            self.model["words"][word] = params

    def predict(self, X) -> str:
        """ Perform classification on an array of test vectors X. """
        words = X.split()
        prob_labels = []
        for cur_label in self.model["labels"]:
            probability = self.model["labels"][cur_label]["probability"]
            total_score = log(probability, e)
            for word in words:
                word_dict = self.model["words"].get(word, None)
                if word_dict:
                    total_score += log(word_dict[cur_label], e)
            prob_labels.append((total_score, cur_label))
        _, answer = max(prob_labels)
        return answer

    def score(self, X_test, y_test) -> float:
        """ Returns the mean accuracy on the given test data and labels. """
        prediction = []
        for one in X_test:
            prediction.append(self.predict(one))
        return sum(0 if prediction[k] != y_test[k] else 1 for k in range(len(X_test))) / len(X_test)

    def smoothing(self, word, cur_label) -> float:
        """ Возвращает сглаженную вероятность со словом и лейблом. """
        nc = self.model["labels"][cur_label]["label_count"]
        nic = self.words_labels.get((word, cur_label), 0)
        counted_len = len(self.counted_words)
        alpha = self.alpha
        return (nic + alpha) / (nc + alpha * counted_len)

    def count_words(self, cur_label) -> int:
        """ Возвращает посчитанные слова с присвоенными лейблами. """
        count = 0
        for word, label_name in self.words_labels:
            if cur_label == label_name:
                count += self.words_labels[(word, cur_label)]
        return count
