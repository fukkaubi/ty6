import db
import scraputils


def save_data(pages: int = 1) -> None:
    news = scraputils.get_news("https://news.ycombinator.com/newest", pages)
    s = db.session()
    for i in news:
        obj = db.News(
            title=i["title"],
            author=i["author"],
            url=i["url"],
            comments=i["comments"],
            points=i["points"],
        )
        s.add(obj)
        s.commit()


save_data(34)
