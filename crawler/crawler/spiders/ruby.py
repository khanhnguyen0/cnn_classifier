import scrapy
import re
import json

class Spider(scrapy.Spider):
    name = "CPP"
    download_delay = 0.5
    start_urls = ['http://rosettacode.org/wiki/Category:C%2B%2B']
    def parse(self, response):
        return scrapy.Request('https://api.github.com/search/repositories?q=node+language:javascript&sort=stars&order=desc', callback = self.parseURL, headers = {'Authentication':'token 36cf8d89dca0f82901f7a42ec6a8397e91252795'})



    def parseURL(self, response):
        jsonresponse = json.loads(response.body_as_unicode())
        for jr in jsonresponse["items"]:
            yield scrapy.Request(jr["html_url"], callback = self.parseRepoURL)


    def parseRepoURL(self, response):
        js = r'.*\.js$'
        folder = r'[^\.]'
        for url in response.xpath('//div[contains(@class,"file-wrap")]/table/tbody/tr[contains(@class,"js-navigation-item")]/td[contains(@class,"content")]/span/a[contains(@href,*)]/@href').extract():
            if re.match(js,url):
                yield scrapy.Request('https://github.com'+url,callback = self.parseFileURL)
            elif re.match(js,url):
                yield scrapy.Request('https://github.com'+url,callback = self.parseRepoURL)


    def parseFileURL(self,response):
        code = ''
        comment = r'^\/\*\* | ^\/\/ | ^\*'
        for line in response.xpath('//div[contains(@itemprop,"text")]/table/tr/td[contains(@class,"blob-code blob-code-inner")]/span/text() | //div[contains(@itemprop,"text")]/table/tr/td[contains(@class,"blob-code blob-code-inner")]/text()').extract():
            if re.match(comment,line):
                continue
            else:
                code+=line
        yield {'code':code}
