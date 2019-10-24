# Import library
import requests
from bs4 import BeautifulSoup, Comment

# dictionary to collect all websites' content
data = {}


def scrape01(link='https://www.tomsguide.com/us/alexa-vs-siri-vs-google,review-4772.html'):
    '''
    Scrape content from link 1.
    '''

    # Get content from url
    source = requests.get(link).text

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

    return content


def scrape02(link='https://thewirecutter.com/reviews/amazon-echo-vs-google-home/'):
    '''
    Scrape content from link 2.
    '''

    # Get content from url
    source = requests.get(link)
    print(source)

    # Content formatting
    soup = BeautifulSoup(source, 'lxml')

    article = soup.find('article')

    # Put title, summary and body all together
    title = article.h1.text
    summary = article.find('section', class_='_1f90b2b1').p.text
    content = [title, summary]

    section1 = article.find('section', class_='db3dab13')
    for i in section1.find_all('p'):
        content.append(i.text)

    section2 = article.find('div', class_='c670ff69')
    for sec in section2.find_all('section'):
        for par in sec.find_all('p'):
            content.append(par.text)

    # Comments TODO

    return content
