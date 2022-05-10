import re

import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore


def extract_news(parser):
    """ Extract news from a given web page """
    news_list = []
    list_one = []
    list_two = []

    news = parser.findAll("tr", class_="athing")
    subtexts = parser.findAll("td", class_="subtext")
    for new in news:
        title = new.find("a", class_="storylink").get_text()
        url = (
            new.find("span", class_="sitestr").get_text()
            if new.find("span", class_="sitestr")
            else None
        )
        list_one.append((title, url))
    for subtext in subtexts:
        author = (
            subtext.find("a", class_="hnuser").get_text()
            if subtext.find("a", class_="hnuser")
            else None
        )
        points = (
            subtext.find("span", class_="score").get_text()
            if subtext.find("span", class_="score")
            else None
        )
        comments = (
            subtext.find("a", href=re.compile("item"), recursive=False).get_text()
            if subtext.find("a", href=re.compile("item"), recursive=False)
            else None
        )
        if points != None:
            points = re.findall(r"\d+", points)[0]
        if comments == "discuss":
            comments = None
        if comments != None:
            comments = re.findall(r"\d+", comments)[0]
        list_two.append((author, points, comments))
    for i in range(30):
        news_list.append(
            {
                "author": list_two[i][0],
                "comments": list_two[i][2],
                "points": list_two[i][1],
                "title": list_one[i][0],
                "url": list_one[i][1],
            }
        )
    return news_list


def extract_next_page(parser):
    """ Extract next page URL """
    next_page = parser.find("a", class_="morelink").get("href")
    return next_page


def get_news(url, n_pages=1):
    """ Collect news from a given web page """
    news = []
    while n_pages:
        print("Collecting data from page: {}".format(url))
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        news_list = extract_news(soup)
        next_page = extract_next_page(soup)
        url = "https://news.ycombinator.com/" + next_page
        news.extend(news_list)
        n_pages -= 1
    return news
