# Import library
import requests
from bs4 import BeautifulSoup

data = {}

# Get content from url
source = requests.get(
    'https://www.tomsguide.com/us/alexa-vs-siri-vs-google,review-4772.html').text

# Content formatting
soup = BeautifulSoup(source, 'lxml')

article = soup.find('article')

# Put title, summary and body all together
title = article.h1.text
summary = article.find('p', class_='strapline').text
content = [title, summary]
article_body = article.find('div', class_='text-copy bodyCopy auto')
for i in article_body.find_all('p'):
    # append all paragraphs within the article body
    content.append(i.text)

data[0] = content
