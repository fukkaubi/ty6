import pathlib
import pickle
import typing as tp

import stemmer
from bayes import NaiveBayesClassifier
from bottle import redirect, request, route, run, template
from db import News, session
from scraputils import get_news


@route("/news")  # type: ignore
def news_list() -> tp.Any:
    s = session()
    rows = s.query(News).filter(News.label == None).all()
    return template("news_template", rows=rows)


@route("/add_label/")  # type: ignore
def add_label() -> None:
    params = request.query
    s = session()
    entry = s.query(News).filter(News.id == params["id"]).first()
    entry.label = params["label"]
    s.commit()
    redirect("/news")


@route("/update")  # type: ignore
def update_news() -> None:
    new_arrivals = get_news("https://news.ycombinator.com/newest")
    s = session()
    marker = s.query(News).first()
    batch_size = 30
    for i, e in enumerate(new_arrivals):
        if e["title"] == marker.title and e["author"] == marker.author:
            batch_size = i
    new_arrivals = new_arrivals[:batch_size]
    for _, e in enumerate(new_arrivals):
        obj = News(
            title=e["title"],
            author=e["author"],
            url=e["url"],
            comments=e["comments"],
            points=e["points"],
        )
        s.add(obj)
        s.commit()
    redirect("/news")


@route("/classify")  # type: ignore
def classify_news() -> tp.Any:
    s = session()
    unclassified: tp.List[tp.Tuple[int, str]] = [
        (i.id, stemmer.clear(i.title)) for i in s.query(News).filter(News.label == None).all()
    ]
    X: tp.List[str] = [i[1] for i in unclassified]
    if not pathlib.Path("model/model.pickle").is_file():
        raise ValueError(
            "Классификатор не натренирован! Пожалуйста, разметь новости, чтобы нормально можно было натренировать модель."
        )
    with open("model/model.pickle", "rb") as model_file:
        model = pickle.load(model_file)
    labels = model.predict(X)
    for i, e in enumerate(unclassified):
        extract = s.query(News).filter(News.id == e[0]).first()
        extract.label = labels[i]
        s.commit()
    rows = s.query(News).filter(News.label != None).order_by(News.label).all()
    return template("classified_template", rows=rows)


if __name__ == "__main__":
    run(host="localhost", port=8080)
