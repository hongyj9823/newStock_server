from django.shortcuts import render
from django.http import JsonResponse
from keywordDatabase.models import DisplayKeyword



# Create your views here.
def getKeywordDatabaseJson(request):
    keyword_datas = DisplayKeyword.objects.all()
    data = []
    for keyword_data in keyword_datas:
        single_data = {}
        single_data['keyword'] = keyword_data.keywords_text
        single_data['importance'] = keyword_data.importance
        single_data['stocks'] = keyword_data.related_stocks.split('/')
        single_data['summary'] = keyword_data.summarized_text
        single_data['news'] = []
        if keyword_data.news_title_1 != None and keyword_data.news_url_1 != None:
            single_data['news'].append({'title' : keyword_data.news_title_1, 'url' : keyword_data.news_url_1})
        if keyword_data.news_title_2 != None and keyword_data.news_url_2 != None:
            single_data['news'].append({'title' : keyword_data.news_title_1, 'url' : keyword_data.news_url_2})
        if keyword_data.news_title_3 != None and keyword_data.news_url_3 != None:
            single_data['news'].append({'title' : keyword_data.news_title_1, 'url' : keyword_data.news_url_3})
        data.append(single_data)

    return JsonResponse({'data': data}, json_dumps_params={'ensure_ascii':False})

