import scrapy

class Spider(scrapy.Spider):
    name = "Spider"
    start_urls = ['http://rosettacode.org/wiki/Category:Python']
    def parse(self, response):
        for url in response.xpath('//a[contains(@href,*)]/@href'):
            yield {
                'url': 'http://rosettacode.org'+url.extract()
            }
