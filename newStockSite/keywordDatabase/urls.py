from django.urls import path

from . import views

urlpatterns = [
    # /keywordDatabase/
    path('displaykeyword/', views.getKeywordDatabaseJson, name='getKeywordDatabaseJson')
]
