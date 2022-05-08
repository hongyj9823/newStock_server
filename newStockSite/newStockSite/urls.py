"""newStockSite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from NewsTab.KeywordGenerator import generateKeywords
from keywordDatabase.models import DisplayKeyword

urlpatterns = [
    path('admin/', admin.site.urls),

    #127.0.0.1:8000/keywordDatabase/
    path('keywordDatabase/', include('keywordDatabase.urls')),
]


print('Clearing Database')
DisplayKeyword.objects.all().delete()

print('Keyword Gen Running')
display_keywords = generateKeywords({"삼성전자" : 670, "카카오" : 100, "네이버" : 240, "현대차" : 120, "SK하이닉스" : 520})

for keyword, info in display_keywords.items():
    importance = info[0]
    related_stocks = info[1]
    summary = info[2]
    news_titles = []
    news_urls = []
    for news in info[3]:
        news_titles.append(news[0])
        news_urls.append(news[1])
    for _ in range(3 - len(info[3])):
        news_titles.append(None)
        news_urls.append(None)
    DisplayKeyword(keywords_text=keyword, related_stocks = related_stocks, 
                    importance=importance, summarized_text=summary, 
                    news_title_1=news_titles[0], news_url_1=news_urls[0],
                    news_title_2=news_titles[1], news_url_2=news_urls[1],
                    news_title_3=news_titles[2], news_url_3=news_urls[2],).save()

print('Keyword Gen Finished')

