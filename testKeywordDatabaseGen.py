from NewsTab.NewsSearcher import searchPresentNews
from NewsTab.KeyWordSelector import getKeyword
from NewsTab.Summarizer import summarize

'''
display_keyword = [
    'keyword 1' : {
        'importance' : rand
        'summarized_text' : txt
        '' :
        news = [
            'title' : 
            'url' : 
        ]
    },
    'keyword 2' : {

    }
]
'''


def testSearch(stock_list):
    stock = {}

    for stock_name in stock_list.keys():
        news_list = searchPresentNews(stock_name)
        stock[stock_name] = news_list

    stock_keywords = {}

    for stock_name, news_list in stock.items():
        for news in news_list:
            news_keywords = getKeyword(news.title)
            #print(f"news Title: {news.title}")
            for idx, keyword in enumerate(news_keywords):
                if keyword in stock_keywords:
                    stock_keywords[keyword][0] += 1
                    stock_keywords[keyword][1] += idx
                    if stock_name not in stock_keywords[keyword][2]:
                        stock_keywords[keyword][2].append(stock_name)
                    if news not in stock_keywords[keyword][3]:
                        stock_keywords[keyword][3].append(news)
                else:
                    stock_keywords[keyword] = [1, idx, [stock_name], [news]]
                #print(f"keyword: [{keyword}]")
            #print()

    stock_keywords = {k : [v[0], v[1], v[2], v[3]] for k, v in sorted(stock_keywords.items(), key=lambda item: (-item[1][0], item[1][1]))}
    '''
    print()
    print()
    for keyword, info in stock_keywords.items():
        print(f"keyword: [{keyword}], Cnt: {info[0]}, IdxSum: {info[1]}")
    print()
    print()
    '''

    i = 0
    while i < len(stock_keywords):
        keywords_list = list(stock_keywords.keys())
        keyword = keywords_list[i]
        for keyword_entry in keyword.split():
            keywords_list = list(stock_keywords.keys())
            j = i + 1
            while j < len(keywords_list):
                if keyword_entry in keywords_list[j]:
                    del stock_keywords[keywords_list[j]]
                j += 1
        i += 1
    
    stock_keywords = {k : [v[0], v[1], v[2], v[3]] for k, v in stock_keywords.items() if v[0] > 1}

    '''
    for keyword, info in stock_keywords.items():
        print(f"keyword: [{keyword}]")
        print(f"Cnt: {info[0]}")
        print(f"IdxSum: {info[1]}")
        for i, news in enumerate(info[2]):
            print(f"{i} : {news}")
        print()
    '''

    display_keyword = {}
    for keyword, info in stock_keywords.items():
        related_stock = "/".join(info[2])
        importance = 0
        for stock_name in info[2]:
            importance += stock_list[stock_name]
        summarized_text = summarize(info[3][0].title, info[3][0].generateContent())
        display_keyword[keyword] = [importance, related_stock, summarized_text, []]
        for news in info[3][:3]:
            display_keyword[keyword][3].append([news.title, news.url])
    
    return display_keyword



display_keywords = testSearch({"삼성전자" : 10, "카카오" : 5, "네이버" : 2, "현대차" : 7, "SK하이닉스" : 6})

for keyword, info in display_keywords.items():
    print(f'keyword[{keyword}], importance: {info[0]}, Related Stocks: {info[1]}')
    print(f'Summary: {info[2]}')
    for news in info[3]:
        print(f'Title: {news[0]}\nURL: {news[1]}\n')
