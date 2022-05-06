from NewsTab.WebCrawler import getNewsArticle

class News:
    def __init__(self, title, url):
        self.title = title
        self.url = url
        self.content = None
    
    def generateContent(self):
        self.content = getNewsArticle(self.url)
        return self.content
        
    def __str__(self):
        string = f"Title: {self.title}\n"
        string += f"URL: {self.url}\n"

        '''
        if self.content != None:
            string += f"Article Body: {self.content}\n\n"
        else:
            string += "Article Body Not Created\n\n"
        '''

        return string

    def __eq__(self, obj):
        return isinstance(obj, News) and self.url == obj.url
    