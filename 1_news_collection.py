from pynytimes import NYTAPI
import datetime
import pandas as pd
import numpy as np


def get_news(year, month, day):
    """
    get top 10 most relevent finance news headings on each day from NY times
    """
    nyt = NYTAPI("5UI21WrJdSgZtHZpljOncwS0qMuJuOcs", parse_dates=True)
    list = []
    articles = nyt.article_search(
            results = 10,
            dates = {
                "begin": datetime.datetime(year, month, day),
                "end": datetime.datetime(year, month, day)
            },
            options = {
                "sort": "relevance",
                "news_desk": [
                    "Business", "Business Day", "Entrepreneurs", "Financial", "Technology"
                ],
                "section_name" : [
                    "Business", "Business Day", "Technology"
                ]
            }
        )
    for i in range(len(articles)):
        list.append(articles[i]['abstract'].replace(',', ""))
    return list

df = pd.DataFrame()



def generate_news_file():
    """
    store news headings everyday of Q3 2022 in csv
    """
    start = '2020-10-01'
    end = '2022-09-30'
    mydates = pd.date_range(start, end)
    dates = []
    for i in range(len(mydates)):
        dates.append(mydates[i].strftime("%Y-%m-%d"))
    matrix = np.zeros((len(dates) + 1, 11), dtype=object)  
    matrix[0, 0] = "Date"

    for i in range(10):
        matrix[0, i + 1] = f"News {i + 1}"
    for i in range(len(dates)):
        matrix[i + 1, 0] = dates[i]
        y, m, d = dates[i].split("-")
        news_list = get_news(int(y), int(m), int(d))
        for j in range(len(news_list)):
            matrix[i + 1, j + 1] = news_list[j]
    df = pd.DataFrame(matrix)
    df.to_csv("news.csv", index = False)


generate_news_file()