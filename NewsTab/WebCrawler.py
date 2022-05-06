from urllib.request import urlopen
import bs4
from bs4 import BeautifulSoup as bs



def getNewsArticle(url = "https://news.naver.com/main/read.naver?mode=LSD&mid=sec&sid1=101&oid=015&aid=0004680664"):
    if type(url) != str:
        raise ValueError
    
    html = urlopen(url)
    soup = bs(html, "html.parser")
    #soup = soup.find(id = "articleBodyContents")
    soup = soup.find(id = "dic_area")
    content = ""
    if soup == None:
        return None
    for i in soup.contents:
        if type(i) == bs4.element.NavigableString:
            content += i
    content = content.strip()
    if content == "":
        return None
    return content

#print(getNewsArticle())
