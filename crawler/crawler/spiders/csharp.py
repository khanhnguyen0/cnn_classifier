import scrapy

class Spider(scrapy.Spider):
    name = "CSharp"
    start_urls = ['http://rosettacode.org/wiki/Category:C_sharp']
    def parse(self, response):
        for url in response.xpath('//a[contains(@href,*)]/@href'):
            yield scrapy.Request('http://rosettacode.org'+url.extract(),callback = self.parseURL)



    def parseURL(self, response):
        a = '';
        for code in response.xpath('//pre[contains(@class,"csharp highlighted_source")]/span[contains(@class,"kw")]/text()').extract():
                a+= code+' '
        if len(a)>0:
            yield {
                    'keywords':a
            }
