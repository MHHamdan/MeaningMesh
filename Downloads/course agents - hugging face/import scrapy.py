import scrapy

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://ici.tou.tv/felix-maude-et-la-fin-du-monde']

    def parse(self, response):
        # Extract text
        text_content = response.xpath('//body//text()').extract()
        with open('text_content.txt', 'w', encoding='utf-8') as text_file:
            text_file.write(' '.join(text_content))

        # Extract image URLs
        img_urls = response.css('img::attr(src)').extract()
        for idx, img_url in enumerate(img_urls, start=1):
            yield scrapy.Request(img_url, callback=self.save_image, meta={'filename': f'image{idx}.jpg'})

    def save_image(self, response):
        filename = response.meta['filename']
        with open(filename, 'wb') as img_file:
            img_file.write(response.body)
