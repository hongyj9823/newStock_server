from NewsTab.NewsSearcher import searchPastNews
from Test.testAnnualChart import getAnnualChart
from datetime import date
import pandas as pd
import csv

stock_code_table = pd.read_csv("stock_code_info.csv", dtype = str, encoding = 'EUC-KR')
stock_code_table = stock_code_table[["회사명", "종목코드"]]



f = open('stockName_date_upDown_articleTitle_info.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(['StockName', 'Date', 'UpDown', 'ArticleTitle'])



for j in range(len(stock_code_table['회사명'])):
    stockName = stock_code_table["회사명"][j]
    print("getting ", stockName)
    
    #search anual chart with code
    date_price = getAnnualChart('A' + stock_code_table['종목코드'][j])


    for i in range(1, len(date_price['date'])):
        dates = date_price['date'][i]
        upDown = int(date_price['price'][i] - date_price['price'][i - 1])

        newsList = searchPastNews(stockName, dates)

        for news in newsList:
            articleTitle = str(news.title)
            writer.writerow([stockName,dates,upDown,articleTitle])

