from django.db import models

# Create your models here.
class DisplayKeyword(models.Model):
    keywords_text = models.CharField(max_length=30)
    importance = models.FloatField()
    related_stocks = models.CharField(max_length=30)
    summarized_text = models.CharField(max_length=200)
    news_title_1 = models.CharField(max_length=100, default=None, blank=True, null=True)
    news_url_1 = models.CharField(max_length=100, default=None, blank=True, null=True)
    news_title_2 = models.CharField(max_length=100, default=None, blank=True, null=True)
    news_url_2 = models.CharField(max_length=100, default=None, blank=True, null=True)
    news_title_3 = models.CharField(max_length=100, default=None, blank=True, null=True)
    news_url_3 = models.CharField(max_length=100, default=None, blank=True, null=True)

