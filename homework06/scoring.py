import typing as tp

import bayes
import stemmer
from db import News, session

s = session()
rows = s.query(News).all()
stop_sign = int(0.7 * len(rows))  # train-to-test ratio is usually 70:30
extracts: tp.List[str] = []
labels: tp.List[str] = []
for i in range(len(rows)):
    row = s.query(News).filter(News.id == (i + 1)).first()  # id is unique and enumerated from id
    extracts.append(row.title)
    labels.append(row.label)
extracts = [stemmer.clear(i.lower()) for i in extracts]
X_train, X_test = extracts[:stop_sign], extracts[stop_sign:]
y_train, y_test = labels[:stop_sign], labels[stop_sign:]
model = bayes.NaiveBayesClassifier(alpha=0.4)
model.fit(X_train, y_train)
print("Classifier accuracy: ", end="")
accuracy = model.score(X_test, y_test)
print(accuracy)
